import torch.nn as nn
import numpy as np
from pcdet.datasets.augmentor.X_transform import X_TRANS
import torch

class HeightCompression(nn.Module):
    def __init__(self, model_cfg,  voxel_size=None, point_cloud_range=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.x_trans = X_TRANS()
        self.point_cloud_range = point_cloud_range
        self.voxel_size=voxel_size

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        batch_dict['spatial_features'] = spatial_features


        return batch_dict
