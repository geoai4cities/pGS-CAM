# Code for point-drop experiment.

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
from datasets.NPM3D import NPM3DDataset, NPM3DSampler, NPM3DCollate
from torch.utils.data import DataLoader


def PointDrop(file_path=None, model=None, dataset=None, cls=None):
        if not file_path:
            raise ValueError("No file_path specified.")
        if not model in ['kpconv', 'randlanet']:
            raise ValueError("Unknown model type specified.")
        if not dataset in ['npm3d', 'semantickitti', 'dales']:
            raise ValueError("Unknown dataset type specified.")

        if dataset == 'npm3d':
            label_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.int64)
            ignored_labels = np.array([0]).astype(np.int64)
            
        cloud = read_ply(file_path)
        
        points = cloud[['x', 'y', 'z']]
        labels = cloud['class']
        preds = cloud['preds']
        pgscam = cloud['pgscam']
        
        # No. of points
        N = pgscam.shape[0]
        
        # Sorted heatmap indices in descending order
        desc_idx = np.argsort(pgscam)[::-1]

        desc_pgscam = pgscam[desc_idx]

        # Percentage wise mIoU drop
        drop_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        for drop in drop_list:
            num_drop = int((drop * N)/100)
            left_idx = desc_idx[num_drop:]
            left_labels = labels[left_idx]
            left_preds = preds[left_idx]

            Confs = [fast_confusion(left_labels, left_preds, label_values)]

            C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

            # Remove ignored labels from confusions
            for l_ind, label_value in reversed(list(enumerate(label_values))):
                if label_value in ignored_labels:
                    C = np.delete(C, l_ind, axis=0)
                    C = np.delete(C, l_ind, axis=1)

            IoUs = IoU_from_confusions(C)
            mIoU = np.mean(IoUs)
            s = '{:5.2f} | '.format(100 * mIoU)
            for IoU in IoUs:
                s += '{:5.2f} '.format(100 * IoU)
            print(s + '\n')

            continue

        return
        
        
if __name__ == "__main__":
    pd = PointDrop(file_path='./pgscam_results/0_cloud.ply', model='kpconv', dataset='npm3d', cls=7)