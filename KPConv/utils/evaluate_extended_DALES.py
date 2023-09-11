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



    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)
        self.cls = 7
        self.act_list = np.arange(0, 22)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction on test/validation cloud. Dim: (M, 3) (Remember this is subsampled one so not original cloud)
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]
        self.test_heatmap = [np.zeros((self.act_list.shape[0], l.shape[0],)) for l in test_loader.dataset.input_labels]

        self.test_loader = test_loader

        # For validation, label proportions. (used as weights in criterion)
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        self.val_proportions = val_proportions

        # Network predictions
        test_epoch = 0
        last_min = -0.5

        while True:
            print("Initialize workers.")
            for i, batch in enumerate(test_loader):
                ti = time.time()
                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                stacked_probs = softmax(outputs)
                logits = stacked_probs[:, self.cls]
                logits = torch.sum(logits, axis=0)
                logits = logits.squeeze()
                logits.backward(retain_graph=True)
                # logits = torch.sum(logits, axis=0)
                # logits = logits.squeeze()

                act_maps = net.activation_maps
                s_points = batch.points[0].cpu().numpy()

                # Get probs and labels
                stacked_probs = stacked_probs.cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                reproj_idx = {}
                for pt in batch.points:
                    pt = pt.cpu().numpy()
                    tree = KDTree(pt, leaf_size=20)
                    idx_ = tree.query(s_points, k=1, return_distance=False)
                    idx_ = idx_.squeeze()
                    reproj_idx[pt.shape[0]] = idx_

                # Get predictions and labels per instance
                # Also update test_probs for each instance.
                t0 = time.time()
                i0 = 0
                for b_i, length in enumerate(lengths):
                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in subsampled cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (
                            1 - test_smooth) * probs
                    i0 += length

                t1 = time.time()

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
                        points = s_points[i0:i0 + length]
                        heatmap = stacked_heatmap[i0:i0 + length]
                        c_i = cloud_inds[b_i]
                        inds = in_inds[i0:i0 + length]

                        if 0 < test_radius_ratio < 1:
                            mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                            inds = inds[mask]
                            heatmap = heatmap[mask]

                        # Update current heatmap in subsampled cloud
                        self.test_heatmap[c_i][act][inds] = test_smooth * self.test_heatmap[c_i][act][inds] + (
                                    1 - test_smooth) * heatmap
                        i0 += length

                    t8 = time.time()

                t9 = time.time()

                if i % 20 == 0:
                    message = 'e{:03d}-i{:04d}'
                    print(message.format(test_epoch, i))

            # Update min potentials

            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            # Compute confusion for subsampled clouds
            if last_min + 1 < new_min:
                # Update last min
                last_min += 1

            if last_min > num_votes:
                break

        # Voting results on subsampled clouds (NOT ORIGINAL CLOUD)
        self.compute_on_sub_cloud(save=True)
        self.compute_on_full_cloud()


    def compute_on_sub_cloud(self, save=False):
        print("\nConfusion on subsampled clouds...")
        Confs = []
        for i, file_path in enumerate(self.test_loader.dataset.files):
            # Insert false columns for ignored labels
            probs = np.array(self.test_probs[i], copy=True)
            for l_ind, label_value in enumerate(self.test_loader.dataset.label_values):
                if label_value in self.test_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = self.test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

            # Targets
            targets = self.test_loader.dataset.input_labels[i]

            # Subsampled points
            sub_points = self.test_loader.dataset.input_trees[i].data.base

            for act in self.act_list:
                # pGS-CAM heatmap
                pgscam = self.test_heatmap[i][act, :]

                pgscam = np.nan_to_num(pgscam, nan=0.0)

                if save:
                    write_ply(f'./pgscam_results/KPConv_DALES_cls_{self.cls}_act_{act+1}.ply',
                              [sub_points, preds, targets, pgscam], ['x', 'y', 'z', 'preds', 'class', 'pgscam'])

            # Confs
            Confs += [fast_confusion(targets, preds, self.test_loader.dataset.label_values)]

        # Regroup confusions
        C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(self.test_loader.dataset.label_values))):
            if label_value in self.test_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Rescale with the right number of point per class
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Compute IoUs
        IoUs = IoU_from_confusions(C)
        mIoU = np.mean(IoUs)
        s = '{:5.2f} | '.format(100 * mIoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        print(s + '\n')

    def compute_on_full_cloud(self):
        print("Reprojection for full cloud (KNN with 1 neighbor)")
        # REPROJECTION to compute for full cloud (Uses KNN with 1 neighbor for compute for entire cloud from subsampled cloud)
        proj_probs = []

        for i, file_path in enumerate(self.test_loader.dataset.files):
            probs = self.test_probs[i][self.test_loader.dataset.test_proj[i], :]
            proj_probs += [probs]

            for l_ind, label_value in enumerate(self.test_loader.dataset.label_values):
                if label_value in self.test_loader.dataset.ignored_labels:
                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

        # PROJECTION ON FULL CLOUDS
        print("Confusion on full clouds...")
        Confs = []
        for i, file_path in enumerate(self.test_loader.dataset.files):
            # Get the predicted labels
            preds = self.test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

            if self.save_labels:
                print("You have the preds for full cloud. Save label anywhere you want...")

            # Confusion
            targets = self.test_loader.dataset.validation_labels[i]
            Confs += [fast_confusion(targets, preds, self.test_loader.dataset.label_values)]

        # Regroup confusions
        C = np.sum(np.stack(Confs), axis=0)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(self.test_loader.dataset.label_values))):
            if label_value in self.test_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        IoUs = IoU_from_confusions(C)
        mIoU = np.mean(IoUs)
        s = '{:5.2f} | '.format(100 * mIoU)
        for IoU in IoUs:
            s += '{:5.2f} '.format(100 * IoU)
        print('-' * len(s))
        print(s)
        print('-' * len(s) + '\n')




