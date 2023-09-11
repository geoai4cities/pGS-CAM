import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix


class ModelTester:
    def __init__(self, net, chkp_path=None, on_gpu=True):
        # Device selection
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")
        
        
    def slam_segmentation_test(self, net, test_loader, config, num_votes=100, debug=True):
        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)
        self.cls = 0
        self.act_list = np.arange(0, 22)

        # self.test_heatmap = [np.zeros((self.act_list.shape[0], l.shape[0],)) for l in test_loader.dataset.input_labels]
        # Number of classes including ignored labels
        self.nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        test_path = None
        report_path = None
        config.saving_path = ''
        config.validation_size = 100

        seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[0])
        velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[0][0] + '.bin')
        frame_points = np.fromfile(velo_file, dtype=np.float32)
        frame_points = frame_points.reshape((-1, 4))[:, :3]
        # Get frames


        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        self.report_path = report_path

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        all_f_heatmaps = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_heatmaps.append([[np.zeros((0,), dtype=np.float32) for _ in seq_frames] for _ in self.act_list])
               
            
        #####################
        # Network predictions
        #####################
                
        self.predictions = []
        self.targets = []
        test_epoch = 0
        
        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):
                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)


                #pGSCAM core
                stacked_probs = softmax(outputs)
                logits = stacked_probs[:, self.cls]
                logits = torch.sum(logits, axis=0)
                logits = logits.squeeze()
                logits.backward(retain_graph=True)

                act_maps = net.activation_maps
                s_points = batch.points[0].cpu().numpy()

                reproj_idx = {}
                for pt in batch.points:
                    pt = pt.cpu().numpy()
                    tree = KDTree(pt, leaf_size=20)
                    idx_ = tree.query(s_points, k=1, return_distance=False)
                    idx_ = idx_.squeeze()
                    reproj_idx[pt.shape[0]] = idx_


                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                torch.cuda.synchronize(self.device)


                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    if exists(filepath):
                        frame_probs_uint8 = np.load(filepath)
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)
                    
                    
                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':
                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)
                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis, axis=1)].astype(np.int32)
            
                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels


                    # Stack all prediction for this epoch
                    i0 += length

                for act in self.act_list:
                    t2 = time.time()
                    # [N, d]
                    cur_act = act_maps[act]

                    # [N, d]
                    grads = cur_act.grad

                    # [1, d]
                    alpha = torch.sum(grads, axis=0, keepdim=True)

                    stacked_heatmap = torch.matmul(cur_act, alpha.T).squeeze()

                    # Apply ReLU
                    stacked_heatmap = torch.maximum(stacked_heatmap, torch.zeros_like(stacked_heatmap))

                    # Normalize
                    max_val = torch.max(stacked_heatmap, dim=-1, keepdim=True)[0]
                    min_val = torch.min(stacked_heatmap, dim=-1, keepdim=True)[0]
                    stacked_heatmap = (stacked_heatmap - min_val) / (max_val - min_val)
                    stacked_heatmap = stacked_heatmap.cpu().detach().numpy()

                    t4 = time.time()

                    # Reprojection of stacked_heatmap (KNN with 1 neighbour)
                    if stacked_heatmap.shape[0] != s_points.shape[0]:
                        idx_ = reproj_idx[stacked_heatmap.shape[0]]
                        stacked_heatmap = stacked_heatmap[idx_]

                    t7 = time.time()

                    i0 = 0
                    for b_i, length in enumerate(lengths):
                        pgscam_hm = stacked_heatmap[i0:i0 + length]
                        proj_inds = r_inds_list[b_i]
                        proj_mask = r_mask_list[b_i]
                        frame_labels = labels_list[b_i]
                        s_ind = f_inds[b_i, 0]
                        f_ind = f_inds[b_i, 1]

                        # Project predictions on the frame points
                        proj_hm = pgscam_hm[proj_inds]

                        if test_loader.dataset.set == 'validation':
                            folder = 'val_probs'
                            pred_folder = 'val_predictions'
                        else:
                            folder = 'probs'
                            pred_folder = 'predictions'
                        filename = '{:s}_{:d}_{:07d}_hm.npy'.format(seq_name, act, f_ind)
                        filepath = join(test_path, folder, filename)
                        if exists(filepath):
                            frame_hm_ufloat32 = np.load(filepath)
                        else:
                            frame_hm_ufloat32 = np.zeros((proj_mask.shape[0],), dtype=np.float32)

                        frame_hm = frame_hm_ufloat32[proj_mask].astype(np.float32)
                        frame_hm = test_smooth * frame_hm + (1 - test_smooth) * proj_hm
                        frame_hm_ufloat32[proj_mask] = frame_hm.astype(np.float32)
                        np.save(filepath, frame_hm_ufloat32)

                        # keep frame heatmaps in memory
                        all_f_heatmaps[s_ind][act][f_ind] = frame_hm_ufloat32
                        i0 += length

                    t8 = time.time()

                t9 = time.time()

                if i%10 == 0:
                    message = f'e{test_epoch}-i{i}'
                    print(message)
            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            
            # Compute confusion for subsampled clouds
            if last_min + 1 < new_min:
                # Update last min
                last_min += 1
                print(last_min)
                self.compute_on_cloud(test_loader, all_f_preds, all_f_labels, all_f_heatmaps, last_min, save=False)
                

            test_epoch += 1
            
            if last_min > num_votes:
                break
                
        return
            
            
    def compute_on_cloud(self, test_loader, all_f_preds, all_f_labels, all_f_heatmaps, last_min, save=False):
        #####################################
        # Results on the whole validation set
        #####################################
        print("Confusion for whole validations set...")
        # Confusions for our subparts of validation set
        Confs = np.zeros((len(self.predictions), self.nc_tot, self.nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(self.predictions, self.targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)

        # Show vote results
        print('\nCompute confusion')

        val_preds = []
        val_labels = []
        t1 = time.time()
        for i, seq_frames in enumerate(test_loader.dataset.frames):
            val_preds += [np.hstack(all_f_preds[i])]
            val_labels += [np.hstack(all_f_labels[i])]
        val_preds = np.hstack(val_preds)
        val_labels = np.hstack(val_labels)
        t2 = time.time()
        C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
        t3 = time.time()
        print(' Stacking time : {:.1f}s'.format(t2 - t1))
        print('Confusion time : {:.1f}s'.format(t3 - t2))

        s1 = '\n'
        for cc in C_tot:
            for c in cc:
                s1 += '{:7.0f} '.format(c)
            s1 += '\n'

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
            if label_value in test_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        # Compute IoUs
        mIoU = np.mean(val_IoUs)
        s2 = '{:5.2f} | '.format(100 * mIoU)
        for IoU in val_IoUs:
            s2 += '{:5.2f} '.format(100 * IoU)
        print(s2 + '\n')

        # Save a report
        report_file = join(self.report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
        str = 'Report of the confusion and metrics\n'
        str += '***********************************\n\n\n'
        str += 'Confusion matrix:\n\n'
        str += s1
        str += '\nIoU values:\n\n'
        str += s2
        str += '\n\n'
        with open(report_file, 'w') as f:
            f.write(str)

        print("Saving pGSCAM results...")
        s_ind = 0
        f_ind = 0
        seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
        velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
        frame_points = np.fromfile(velo_file, dtype=np.float32)

        frame_points = frame_points.reshape((-1, 4))[:, :3]
        preds = all_f_preds[s_ind][f_ind].astype(np.int32)
        labels = all_f_labels[s_ind][f_ind].astype(np.int32)

        for act in self.act_list:
            pgscam = all_f_heatmaps[s_ind][act][f_ind]
            entities = [frame_points, preds, labels, pgscam]
            write_ply(f'./pgscam_results/KPConv_SemanticKITTI_cls_{self.cls}_act_{act}.ply', entities, ['x', 'y', 'z', 'preds', 'class', 'pgscam'])