import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from datasets.voc_dataset import VOCDataset
from model import YOLOv2
from rpn_utils import k_means_cluster_anchor_box

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    batch_size = 16
    learning_rate = .001
    num_epoch = 10

    root_dir = 'C://DeepLearningData/VOC2012/'
    train_dset = VOCDataset(root_dir, to_categorical=False, get_bbox_only=True)
    # val_dset = VOCDataset(root_dir, is_validation=True, to_categorical=False)

    train_data_loader = DataLoader(train_dset, batch_size=len(train_dset))
    bbox = None
    for i, anns in enumerate(train_data_loader):
        bbox = anns['bbox']
        break
    bbox = bbox.numpy()
    centroids, clusters = k_means_cluster_anchor_box(5, bbox)
    print(centroids)
    print(clusters.shape)

    # model = YOLOv2().to(device)

