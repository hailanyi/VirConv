import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class BEVFeaturesInterpolation(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_frames=1, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0

        if 'temporal_features' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'spatial_features' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        self.output_bev_features = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, points, bev_features, batch_size, bev_stride):

        point_bev_features_list = []
        for k in range(batch_size):

            points_b = points[:,0]

            cur_batch_points = points[points_b==k]

            x_idxs = (cur_batch_points[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
            y_idxs = (cur_batch_points[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
            cur_x_idxs = x_idxs / bev_stride
            cur_y_idxs = y_idxs / bev_stride

            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)

            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features


    def forward(self, batch_dict):

        for i in range(self.num_frames):
            if i==0:
                point_features_list = []
                if 'temporal_features' in self.model_cfg.FEATURES_SOURCE:
                    bev_features = batch_dict['temporal_features']
                    point_bev_features = self.interpolate_from_bev_features(
                        batch_dict['points'], bev_features, batch_dict['batch_size'],
                        bev_stride=batch_dict['spatial_features_stride']
                    )
                    point_features_list.append(point_bev_features)
                if 'spatial_features' in self.model_cfg.FEATURES_SOURCE:
                    bev_features = batch_dict['spatial_features']
                    point_bev_features = self.interpolate_from_bev_features(
                        batch_dict['points'], bev_features, batch_dict['batch_size'],
                        bev_stride=batch_dict['spatial_features_stride']
                    )
                    point_features_list.append(point_bev_features)
                point_features = torch.cat(point_features_list, dim=-1)

                point_features = self.output_bev_features(point_features.view(-1, point_features.shape[-1]))

                batch_dict['point_features'] = point_features  # (BxN, C)
                batch_dict['point_coords'] = batch_dict['points'][:,0:4]  # (BxN, 4)
            elif 'points'+str(-i) in batch_dict:

                points = batch_dict['points'+str(-i)]

                point_features_list = []
                if 'temporal_features' in self.model_cfg.FEATURES_SOURCE:

                    bev_features = batch_dict['temporal_features'+str(-i)]

                    point_bev_features = self.interpolate_from_bev_features(
                        points, bev_features, batch_dict['batch_size'],
                        bev_stride=batch_dict['spatial_features_stride']
                    )
                    point_features_list.append(point_bev_features)
                if 'spatial_features' in self.model_cfg.FEATURES_SOURCE:
                    bev_features = batch_dict['spatial_features'+str(-i)]
                    point_bev_features = self.interpolate_from_bev_features(
                        points, bev_features, batch_dict['batch_size'],
                        bev_stride=batch_dict['spatial_features_stride']
                    )
                    point_features_list.append(point_bev_features)
                point_features = torch.cat(point_features_list, dim=-1)

                point_features = self.output_bev_features(point_features.view(-1, point_features.shape[-1]))

                batch_dict['point_features'+str(-i)] = point_features  # (BxN, C)
                batch_dict['point_coords'+str(-i)] = batch_dict['points'+str(-i)][:, 0:4]  # (BxN, 4)

        return batch_dict