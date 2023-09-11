import numpy as np 
import torch
from utils.ply import read_ply, write_ply
from torch.autograd import grad
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from drop_dataset import SemanticKITTI
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
import umap

        
        
class PowerCAM:
    """
    Vanilla gradient class activation mapping 
    """
    
    def __init__(self, input_model, batch, config, mask_type="none", mode="normal", norm=False, cls=-1):
        # mode: [normal, counterfactual]
        # mask_type: [none, single, subset]:- none(no mask), single(only single point), subset(collection of points)
        
        self.input_model = input_model
        self.batch = batch
        self.cls = cls
        self.config = config
        self.norm = norm
        self.is_masked = True
        self.threshold = [0.1, 0.3, 0.3]
        self.mode = mode
        self.mask_type = mask_type
        
        # actuall labels starts from unlabelled but we are ignoring it
        self.transform_map = {
          0: "car",
          1: "bicycle",
          2: "motorcycle",
          3: "truck",
          4: "other-vehicle",
          5: "person",
          6: "bicyclist",
          7: "motorcyclist",
          8: "road",
          9: "parking",
          10: "sidewalk",
          11: "other-ground",
          12: "building",
          13: "fence",
          14: "vegetation",
          15: "trunk",
          16: "terrain",
          17: "pole",
          18: "traffic-sign"
      }
        
    def create_mask(self):
        # logits: [1, d, N]
        # points: [1, N, 3]
        # preds: [1, N]
        logits = self.end_points['logits']
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1)
        point_0 = self.end_points['xyz'][0]
        if self.mask_type == 'none':
            logits_mask = torch.ones_like(logits)
            
        elif self.mask_type == 'subset':
            # Create Mask
            pred_mask = (preds == self.cls).int()
            inds_mask = torch.argwhere(pred_mask.squeeze()).squeeze()
            masked_point = point_0[:, inds_mask, :].detach().cpu().numpy()
            
            # Perform DBSCAN clustering to seperate entities of class self.cls
            clustering = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto').fit(masked_point.squeeze())
            
            cluster_labels = clustering.labels_
            
            # Let's extract cluster 0
            inds_clust = np.argwhere((cluster_labels == 0).astype(int)).squeeze()
            clust_point = point_0[:, inds_mask[inds_clust], :]            
            logits_mask = torch.zeros_like(logits)
            logits_mask[:, :, inds_mask[inds_clust]] = 1
            
            preds_mask  = -torch.ones_like(preds)
            preds_mask[:, inds_mask[inds_clust]] = 1
            
            entities = [point_0[0].detach().cpu().numpy(), preds_mask[0].cpu().numpy().astype(np.int32)]
            
            write_ply('vis_cluster.ply', entities, ['x', 'y', 'z', 'preds'])

        elif self.mask_type == 'single':
            # Create Mask
            pred_mask = (preds == self.cls).int()
            inds_mask = torch.argwhere(pred_mask.squeeze()).squeeze()
            
            # First element of inds mask as ROI
            logits_mask = torch.zeros_like(logits)
            
            # Logits 
            logits_mask[:, :, inds_mask[0]] = 1
            
            
        
        return logits_mask
#         return inds_mask[inds_clust]
        
    def getGradients(self):
#         print(f"Current class: {self.transform_map[self.cls]}")
        self.input_model.eval()
        self.end_points = self.input_model(self.batch)
        logits = self.end_points['logits']
                
#         logits = self.create_mask()*logits
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1)
                
#         logits = logits[:, :, self.create_mask()]
        logits = self.create_mask()*logits
        
        logits = softmax(logits)
        mask = ((preds.squeeze(0) == self.cls).unsqueeze(0)).unsqueeze(1)
#         print(f"Number of points for class {self.transform_map[self.cls]}: ", torch.sum(mask.squeeze()).item())

        logits = logits[:, self.cls, :]
        logits = torch.sum(logits, axis=-1)
        
        logits = logits.squeeze()
        
        self.logits = logits

        self.logits.backward(retain_graph=True)
                
        self.activations = self.end_points['activations']
        
        return torch.sum(mask.squeeze()).item()
    
    
    def heatmap(self):
#         self.logits.backward(retain_graph=True)
#         self.logits.backward()
        
        heatmaps_III = []
        heatmaps_III_kdtree = []
        point_0 = self.end_points['xyz'][0].cpu().numpy().squeeze()
#         tree = KDTree(point_0.cpu().numpy().squeeze())
#         print(point_0.shape)
        
        for i, act in enumerate(self.activations):
            grads = act.grad
            if self.mode == 'normal':
                alpha = torch.sum(grads, axis=(2,3))
            elif self.mode == 'counterfactual':
                alpha = -torch.sum(grads, axis=(2,3))
            activation = act.squeeze()
            heatmap = torch.matmul(alpha, activation)
            
            # Apply ReLU
            heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
            
            # Normalize
            max_val = torch.max(heatmap, dim=-1, keepdim=True)[0]
            min_val = torch.min(heatmap, dim=-1, keepdim=True)[0]
            heatmap = (heatmap - min_val) / (max_val - min_val)
            heatmap = heatmap.cpu().detach().numpy()
            
            # Fill NaN values
            heatmap = np.nan_to_num(heatmap)
            
            heatmaps_III.append(heatmap)
            
            heatmap = heatmap.squeeze()
            

            
            if act.shape[2] != point_0.shape[1]:
                for pt in self.end_points['xyz']:
                    if pt.shape[1] == act.shape[2]:
                        tree = KDTree(pt.cpu().numpy().squeeze(), leaf_size=40)
                        idx = tree.query(point_0, return_distance=False).squeeze()
                        heatmaps_III_kdtree.append(np.expand_dims(heatmap[idx], 0))
            else:
                heatmaps_III_kdtree.append(np.expand_dims(heatmap[idx],0))
#             print(act.shape, heatmap.shape)
         
        self.heatmaps_III = heatmaps_III
        self.heatmaps_III_kdtree = heatmaps_III_kdtree
        
        
        
    def refinement(self):
        hm = self.heatmaps_III_kdtree[-1].squeeze()
        logits = self.end_points['logits']
#         logits = self.create_mask()*logits
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1).squeeze().detach().cpu().numpy()
        
        preds_mask = (preds == self.cls).astype(np.int32)
        
        hm_mask = (hm > 0.5).astype(np.int32)
        
        pred_final = hm_mask * preds_mask
        
        print(pred_final.shape, np.unique(hm_mask), np.unique(preds_mask))

        
    def visCAM(self):
        print("Saving visuals...")
        points = self.end_points['xyz']
        labels = self.end_points['labels'].cpu().numpy().astype(np.int32)
        preds = self.end_points['logits'].cpu()
        
        for hm_i, hm in enumerate(self.heatmaps_III):
            for p_i, p in enumerate(points):
                if hm.shape[1] == p.shape[1]:
                    entities = [p[0].detach().cpu().numpy(), hm[0]]
                    break
            
            write_ply(f'./visuals/{self.mask_type}_{self.mode}_{self.transform_map[self.cls]}_{hm_i}_pgscam.ply', entities, ['x', 'y', 'z', 'heatmap'])
            
        
        for hm_i, hm in enumerate(self.heatmaps_III_kdtree):
            for p_i, p in enumerate(points):
                if hm.shape[1] == p.shape[1]:
                    entities = [p[0].detach().cpu().numpy(), hm[0]]
                    break
                    
            write_ply(f'./visuals/{self.mask_type}_{self.mode}_{self.transform_map[self.cls]}_{hm_i}_pgscam_kdtree.ply', entities, ['x', 'y', 'z', 'heatmap'])       
        
#     def visCAM(self):
#         print("Saving visuals...")
#         points = self.end_points['xyz']  # list
#         labels = self.end_points['labels'].cpu().numpy().astype(np.int32)  # [1, N]
#         preds = self.end_points['logits'].cpu()
#         softmax = torch.nn.Softmax(1)
#         preds = softmax(preds).argmax(dim=1).detach().numpy().astype(np.int32) # [1, N]
            
#         for hm_i, hm in enumerate(self.heatmaps_I_II_III):
#             for p_i, p in enumerate(points):
#                 if hm.shape[1] == p.shape[1]:
#                     entities = [p[0].detach().cpu().numpy(), hm[0].detach().cpu().numpy()]
#             if hm.shape[1] == points[0].shape[1]:
#                 entities += [labels[0], preds[0]]    
#                 write_ply(f'./visuals/{self.transform_map[self.cls]}_{hm_i}_vanillaCam.ply', entities, ['x', 'y', 'z', 'heatmap', 'labels', 'preds'])
#                 continue
#             write_ply(f'./visuals/{self.transform_map[self.cls]}_{hm_i}_vanillaCam.ply', entities, ['x', 'y', 'z', 'heatmap'])

    
    
    def runCAM(self):
        num_points = self.getGradients()
        self.heatmap()
#         self.refinement()
#         self.visCAM()
        return num_points
    
class Drop_attack():
    def __init__(self):
        self.transform_map = {
          0: "car",
          1: "bicycle",
          2: "motorcycle",
          3: "truck",
          4: "other-vehicle",
          5: "person",
          6: "bicyclist",
          7: "motorcyclist",
          8: "road",
          9: "parking",
          10: "sidewalk",
          11: "other-ground",
          12: "building",
          13: "fence",
          14: "vegetation",
          15: "trunk",
          16: "terrain",
          17: "pole",
          18: "traffic-sign"
      }
        
    @staticmethod
    def my_worker_init_fn(worker_id):
#         np.random.seed(np.random.get_state()[1][0] + worker_id)
        np.random.seed(0)
    
    def drop(self, input_model, batch, config, n_drop=[0, 1000, 2000, 3000, 4000, 5000], cls=-1, drop_type='high'):
        self.cls = cls
        cam = PowerCAM(input_model, batch, config, norm=False, cls=cls)
        _ = cam.runCAM()
        heatmaps = cam.heatmaps_III
        end_points = cam.end_points
        
        hm = heatmaps[-1]
        point_0 = batch['xyz'][0].detach().cpu().numpy()
        label_0 = batch['labels'].detach().cpu().numpy()
        preds = batch['logits'].detach().cpu().numpy()
        
        point_collect = []
        miou_collect = []
        ciou_collect = []
        
        for drop_ in n_drop:
            # Sorts in descending order
#             print(f"Drop: {drop_}")
            if drop_type == 'high':
                ind = np.argsort(hm.squeeze())[::-1]
            else:
                ind = np.argsort(hm.squeeze())
                
            point_sort = point_0[:, ind, :]
            label_sort = label_0[:, ind]
            hm_sort = hm[:, ind]
            
            drop_points = point_sort[:, :drop_, :].squeeze()
            drop_labels = label_sort[:, :drop_].squeeze()
            drop_hm = hm_sort[:, :drop_].squeeze()
            
            point_collect.append((drop_points, drop_labels, drop_hm))
            
            ent_ = [drop_points.squeeze(), drop_labels.squeeze().astype(np.int32), drop_hm.squeeze()]
#             write_ply(f'drop_{drop_}', ent_, ['x', 'y', 'z', 'cls', 'hm'])
            
            rem_points = point_sort[:, drop_:, :].squeeze()
            rem_labels = label_sort[:, drop_:].squeeze()
            
            DATASET = SemanticKITTI(rem_points, rem_labels)
    
            DATALOADER = DataLoader(DATASET, batch_size=1, shuffle=True, num_workers=20, worker_init_fn=self.my_worker_init_fn, collate_fn=DATASET.collate_fn)
    
            for batch_ in DATALOADER:
                for key in batch_:
                    if type(batch_[key]) is list:
                        for i in range(len(batch_[key])):
                            batch_[key][i] = batch_[key][i].cuda()
                    else:
                        batch_[key] = batch_[key].cuda()
                continue
                
            end_points_ = input_model(batch_)
            mean_iou_, iou_list_, loss_ = self.compute_iou_(end_points_, config)
            miou_100, ciou_100 = self.display_iou(mean_iou_, iou_list_, end_points_['xyz'][0].squeeze())
            
            miou_collect.append(miou_100)
            ciou_collect.append(ciou_100)
            
            
            
            cam_ = PowerCAM(input_model, batch_, config, norm=False, cls=cls)
            _ = cam_.runCAM()
            heatmaps_ = cam_.heatmaps_III_kdtree
            
            point_0 = np.expand_dims(rem_points, axis=0)
            label_0 = np.expand_dims(rem_labels, axis=0)
            hm = heatmaps_[-1]
            
#             entities = [batch_['xyz'][0].squeeze().detach().cpu().numpy(), batch_['labels'].squeeze().cpu().numpy().astype(np.int32), hm.squeeze()]
            
#             write_ply(f'./accumulated_piecewise/{self.transform_map[self.cls]}_SemanticKITTI_drop_{drop_}', entities, ['x', 'y', 'z', 'cls', 'hm'])
            
            
#         pt_full = np.empty(shape=(0,3))
#         hm_full = np.empty(shape=(0,))
#         label_full = np.empty(shape=(0,))
#         for entity in point_collect:
#             pt_, lb_, hm_ = entity
#             pt_full = np.concatenate((pt_full, pt_.squeeze()), axis=0)
#             hm_full = np.concatenate((hm_full, hm_.squeeze()), axis=0)
#             label_full = np.concatenate((label_full, lb_.squeeze()), axis=0)
        
#         pt_full = np.concatenate((pt_full, point_0.squeeze()), axis=0)
#         hm_full = np.concatenate((hm_full, hm.squeeze()), axis=0)
#         label_full = np.concatenate((label_full, label_0.squeeze()), axis=0)
        
#         ent_ = [pt_full, label_full.astype(np.int32), hm_full]
#         write_ply(f'./accumulated_piecewise/pt_full.ply', ent_, ['x', 'y', 'z', 'label', 'pgscam'])
#         print(pt_full.shape, hm_full.shape, label_full.shape)
            
            
            
            
#             end_points_ = input_model(batch_)
            
#             mean_iou_, iou_list_, loss_ = self.compute_iou_(end_points_, config)
#             self.display_iou(mean_iou_, iou_list_, end_points_['xyz'][0].squeeze())
            
#             logits = end_points_['logits']
        
#             softmax = torch.nn.Softmax(1)
#             preds = softmax(logits).argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.int32)
            
#             entities = [end_points_['xyz'][0].squeeze().detach().cpu().numpy(), end_points_['labels'].squeeze().detach().cpu().numpy().astype(np.int32), preds]
            
#             write_ply(f'./drop_vis/{self.transform_map[self.cls]}_SemanticKITTI_drop_{drop_}', entities, ['x', 'y', 'z', 'cls', 'preds'])
        return miou_collect, ciou_collect
        
        
    def compute_iou_(self, end_points, config):
        loss, end_points = compute_loss(end_points, config)
        acc, end_points = compute_acc(end_points)
        
        iou_calc = IoUCalculator(config)
        iou_calc.add_data(end_points)
        mean_iou, iou_list = iou_calc.compute_iou()
        
        return mean_iou, iou_list, loss
        
    def display_iou(self, mean_iou, iou_list, points):
#         print(f"Points shape: {points.shape}\n")
#         print("Class Wise IoU: \n")
#         for i in range(len(iou_list)):
#             print(f'{self.transform_map[i]}: {iou_list[i]*100}')
            
#         print(f"\nMean IoU: {mean_iou * 100}\n")
        
#         print(f"Class IoU: {iou_list[self.cls]*100}\n")
        
        return (mean_iou * 100, iou_list[self.cls]*100)
        
 

class piecewise_pGSCAM():
    def __init__(self, input_model, batch, config, mask_type="subset", mode="normal", norm=False, cls=-1):
        # mode: [normal, counterfactual]
        # mask_type: [none, single, subset]:- none(no mask), single(only single point), subset(collection of points)
        
        self.input_model = input_model
        self.batch = batch
        self.cls = cls
        self.config = config
        self.norm = norm
        self.is_masked = True
        self.threshold = [0.1, 0.3, 0.3]
        self.mode = mode
        self.mask_type = mask_type
        self.partial_heatmaps = []
        # actuall labels starts from unlabelled but we are ignoring it
        self.transform_map = {
          0: "car",
          1: "bicycle",
          2: "motorcycle",
          3: "truck",
          4: "other-vehicle",
          5: "person",
          6: "bicyclist",
          7: "motorcyclist",
          8: "road",
          9: "parking",
          10: "sidewalk",
          11: "other-ground",
          12: "building",
          13: "fence",
          14: "vegetation",
          15: "trunk",
          16: "terrain",
          17: "pole",
          18: "traffic-sign"
      }
        
    @staticmethod
    def my_worker_init_fn(worker_id):
#         np.random.seed(np.random.get_state()[1][0] + worker_id)
        np.random.seed(0)
        
    def runCAM(self):
        net = self.input_model
        net.eval()
        batch = self.batch
        config = self.config
        
#         while True:
        cam = PowerCAM(net, batch, config, norm=False, cls=self.cls)
        _ = cam.runCAM()
        vanilla_heatmaps = cam.heatmaps_III_kdtree
        activations = cam.activations

        # I am interested in act -1
        act = activations[-1]
        hm_orig = vanilla_heatmaps[-1]
        point_orig = self.batch['xyz'][0].detach().cpu().numpy()
        label_orig = self.batch['labels'].cpu().numpy()
        
        point = point_orig
        label = label_orig
        hm  = hm_orig
        point_temp = []
#         partial_ind_delete = ['dummy']
        itr = 5
        
        while itr:
                partial_ind_delete = np.argwhere(hm.squeeze() >= 0.8).squeeze()
                partial_ind_accept = np.argwhere(hm.squeeze() <= 0.8).squeeze()
                point_ = point[:, partial_ind_delete, :]
                label_ = label[:, partial_ind_delete]
                hm_ = hm[:, partial_ind_delete]
                point_temp.append((point_, label_, hm_))
                
                point = point[:, partial_ind_accept, :]
                label = label[:, partial_ind_accept]
                
                print(point.shape, label.shape, partial_ind_delete)

                DATASET = SemanticKITTI(point.squeeze(), label.squeeze())

                DATALOADER = DataLoader(DATASET, batch_size=1, shuffle=True, num_workers=20, worker_init_fn=self.my_worker_init_fn, collate_fn=DATASET.collate_fn)

                for batch_ in DATALOADER:
                    for key in batch_:
                        if type(batch_[key]) is list:
                            for i in range(len(batch_[key])):
                                batch_[key][i] = batch_[key][i].cuda()
                        else:
                            batch_[key] = batch_[key].cuda()
                    continue


                cam_ = PowerCAM(net, batch_, config, norm=False, cls=self.cls)
                _ = cam_.runCAM()
                heatmaps_ = cam_.heatmaps_III_kdtree
                hm = heatmaps_[-1]
                
                ent_ = [point.squeeze(), label.squeeze().astype(np.int32), hm.squeeze()]
                write_ply(f'./accumulated_piecewise/{point.shape[1]}.ply', ent_, ['x', 'y', 'z', 'label', 'pgscam'])
                itr = itr - 1
                
#             except:
#                 break
            
        pt_full = np.empty(shape=(0,3))
        hm_full = np.empty(shape=(0,))
        label_full = np.empty(shape=(0,))
        for entity in point_temp:
            pt_, lb_, hm_ = entity
            pt_full = np.concatenate((pt_full, pt_.squeeze()), axis=0)
            hm_full = np.concatenate((hm_full, hm_.squeeze()), axis=0)
            label_full = np.concatenate((label_full, lb_.squeeze()), axis=0)
        
        pt_full = np.concatenate((pt_full, point.squeeze()), axis=0)
        hm_full = np.concatenate((hm_full, hm.squeeze()), axis=0)
        label_full = np.concatenate((label_full, label.squeeze()), axis=0)
        
        ent_ = [pt_full, label_full.astype(np.int32), hm_full]
        write_ply(f'./accumulated_piecewise/pt_full.ply', ent_, ['x', 'y', 'z', 'label', 'pgscam'])
        print(pt_full.shape, hm_full.shape, label_full.shape)
        
            
            
            

#         print(point_.shape)
            
            
        
        
        
        
        
        
        
        
        
        
#     def getmIoU(self):
# #         num_points = self.getGradients()
#         hm_IoU = []
#         hm_1_list, hm_2_list, hm_3_list = self.heatmaps_I, self.heatmaps_I_II, self.heatmaps_I_II_III
# #         hm_1_list, hm_2_list, hm_3_list = self.heatmap()
#         hm_1 = hm_1_list[7].squeeze()
#         hm_2 = hm_2_list[7].squeeze()
#         hm_3 = hm_3_list[7].squeeze()
# #         print(hm_1.shape, hm_2.shape, hm_3.shape)
#         logits = self.end_points['logits']
#         softmax = torch.nn.Softmax(1)
#         preds = softmax(logits).argmax(dim=1).squeeze()

        
#         pred_mask = (preds == self.cls).type(torch.int32)
#         hm_1 = (hm_1 > self.threshold[0]).type(torch.int32)
#         hm_2 = (hm_2 > self.threshold[1]).type(torch.int32)
#         hm_3 = (hm_3 > self.threshold[2]).type(torch.int32)
#         iou_metric = IoUCalculator()
#         iou_metric.add_data(hm_1, pred_mask)
#         mean_iou, iou_list = iou_metric.compute_iou()
#         cls_iou = iou_list[1]*100
#         hm_IoU.append(cls_iou)
        
# #         iou_metric = IoUCalculator()
#         iou_metric.add_data(hm_2, pred_mask)
#         mean_iou, iou_list = iou_metric.compute_iou()
#         cls_iou = iou_list[1]*100
#         hm_IoU.append(cls_iou)
        
        
# #         iou_metric = IoUCalculator()
#         iou_metric.add_data(hm_3, pred_mask)
#         mean_iou, iou_list = iou_metric.compute_iou()
#         cls_iou = iou_list[1]*100
#         hm_IoU.append(cls_iou)

#         return hm_IoU
#         self.visCAM()


class TSNE_cls():
    def __init__(self, input_model, batch, config, norm=False, cls=-1):
        self.input_model = input_model
        self.batch = batch
        self.cls = cls
        self.config = config
        self.norm = norm
        self.is_masked = True
        self.threshold = [0.1, 0.3, 0.3]
        
        # actuall labels starts from unlabelled but we are ignoring it
        self.transform_map = {
          0: "car",
          1: "bicycle",
          2: "motorcycle",
          3: "truck",
          4: "other-vehicle",
          5: "person",
          6: "bicyclist",
          7: "motorcyclist",
          8: "road",
          9: "parking",
          10: "sidewalk",
          11: "other-ground",
          12: "building",
          13: "fence",
          14: "vegetation",
          15: "trunk",
          16: "terrain",
          17: "pole",
          18: "traffic-sign"
      }
        
    def getActivations(self):
        print(f"Current class: {self.transform_map[self.cls]}")
        self.input_model.eval()
        self.end_points = self.input_model(self.batch)
        logits = self.end_points['logits']
        
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1)
        
        logits = softmax(logits)
        mask = ((preds.squeeze(0) == self.cls).unsqueeze(0)).unsqueeze(1)
        print(f"Number of points for class {self.transform_map[self.cls]}: ", torch.sum(mask.squeeze()).item())
        
        self.activations = self.end_points['activations']
        
        print(len(self.activations))
        
        return torch.sum(mask.squeeze()).item()
    
    
    def t_sne(self, i_act):
        act = self.activations[i_act]
        
        labels = self.end_points['labels'].cpu().numpy().astype(np.int32).squeeze()
        logits = self.end_points['logits']
        softmax = torch.nn.Softmax(1)
        preds = softmax(logits).argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
        
        act = act.squeeze().T
#         tsne = TSNE()
        reducer = PCA(n_components=2)
        act_reduce = reducer.fit_transform(act.detach().cpu().numpy())
        np.save(f'./tsne_vis/pca_act_{i_act}', act_reduce)
        
        print("Computing umap:")
        reducer = TSNE()
        act_reduce = reducer.fit_transform(act.detach().cpu().numpy())
        np.save(f'./tsne_vis/tsne_act_{i_act}', act_reduce)
        
        np.save(f'./tsne_vis/preds', preds)
        np.save(f'./tsne_vis/labels', labels)
        palette = sns.color_palette('bright', 10)
#         sns.scatterplot(act_reduce[:, 0], act_reduce[:, 1], hue=labels, legend='full', palette=palette)

        
        
    def runCAM(self):
        cam = PowerCAM(self.input_model, self.batch, self.config, norm=True, cls=self.cls, mode='normal', mask_type='none')
        num_points = cam.runCAM()
        print(num_points)
        
        num_points = self.getActivations()
        for act_i in [0, 7, 8, 9, 10]:
            self.t_sne(act_i)
        return num_points
        
    


class IoUCalculator_heatmaps:
    def __init__(self):
        self.num_classes = 2
        self.gt_classes = [0 for _ in range(self.num_classes)]
        self.positive_classes = [0 for _ in range(self.num_classes)]
        self.true_positive_classes = [0 for _ in range(self.num_classes)]
       

    def add_data(self, hm, pred_mask):
        hm_valid = hm.detach().cpu().numpy()
        pred_mask_valid = pred_mask.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(hm_valid == pred_mask_valid)
        val_total_correct += correct
        val_total_seen += len(pred_mask_valid)

        conf_matrix = confusion_matrix(pred_mask_valid, hm_valid, labels=np.arange(0, self.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        return mean_iou, iou_list
