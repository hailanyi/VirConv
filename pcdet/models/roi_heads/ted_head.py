import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from functools import partial
import pickle
import copy

from pcdet.datasets.augmentor.X_transform import X_TRANS

class PositionalEmbedding(nn.Module):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim, pos = True, head = 4):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos_dim = 8
        self.pos = pos

        if self.pos:
            self.pos_en = PositionalEmbedding(self.pos_dim)

            self.Q_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim+self.pos_dim, hidden_dim, bias=False)
        else:

            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.att = nn.MultiheadAttention(hidden_dim, head)


    def forward(self, inputs, Q_in): # N,B,C

        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]

        if self.pos:
            pos_input = torch.from_numpy(np.arange(seq_len)+1).cuda()
            pos_input = self.pos_en(pos_input, batch_size)
            inputs_pos = torch.cat([inputs, pos_input], -1)
            pos_Q = torch.from_numpy(np.array([seq_len])).cuda()
            pos_Q = self.pos_en(pos_Q, batch_size)
            Q_in_pos = torch.cat([Q_in, pos_Q], -1)
        else:
            inputs_pos = inputs
            Q_in_pos = Q_in

        Q = self.Q_linear(Q_in_pos)
        K = self.K_linear(inputs_pos)
        V = self.V_linear(inputs_pos)

        out = self.att(Q, K, V)

        return out[0]

class Attention_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs): # B,K,N


        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)
        V = self.V_linear(inputs)

        alpha = torch.matmul(Q, K)

        alpha = F.softmax(alpha, dim=2)

        out = torch.matmul(alpha, V)

        out = torch.mean(out, -2)

        return out

def gen_sample_grid(rois, grid_size=7, grid_offsets=(0, 0), spatial_scale=1.):
    faked_features = rois.new_ones((grid_size, grid_size))
    N = rois.shape[0]
    dense_idx = faked_features.nonzero()  # (N, 2) [x_idx, y_idx]
    dense_idx = dense_idx.repeat(N, 1, 1).float()  # (B, 7 * 7, 2)

    local_roi_size = rois.view(N, -1)[:, 3:5]
    local_roi_grid_points = (dense_idx ) / (grid_size-1) * local_roi_size.unsqueeze(dim=1) \
                      - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 7 * 7, 2)

    ones = torch.ones_like(local_roi_grid_points[..., 0:1])
    local_roi_grid_points = torch.cat([local_roi_grid_points, ones], -1)

    global_roi_grid_points = common_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)

    x = global_roi_grid_points[..., 0:1]
    y = global_roi_grid_points[..., 1:2]

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(grid_size**2, -1), y.view(grid_size**2, -1)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W        C,K,1,2   C,K,1,1

    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)# 49,K,1,1
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / W)  # normalize to between  0 and 1

    samples[:, :, :, 1] = (samples[:, :, :, 1] / H)  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1  # 49,K,1,2

    #B,C,H,W
    #B,H,W,2
    #B,C,H,W

    return torch.nn.functional.grid_sample(image, samples, align_corners=False)

class TEDMHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range=None, voxel_size=None,  num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class,  model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.pool_cfg_mm = model_cfg.ROI_GRID_POOL_MM
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        LAYER_cfg_mm = self.pool_cfg_mm.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.rot_num = model_cfg.ROT_NUM

        self.x_trans_train = X_TRANS()

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        c_out_mm = 0
        self.roi_grid_pool_layers_mm = nn.ModuleList()
        feat = self.pool_cfg_mm.get('FEAT_NUM', 1)
        for src_name in self.pool_cfg_mm.FEATURES_SOURCE:
            mlps = LAYER_cfg_mm[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]*feat] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg_mm[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg_mm[src_name].NSAMPLE,
                radii=LAYER_cfg_mm[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg_mm[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers_mm.append(pool_layer)

            c_out_mm += sum([x[-1] for x in mlps])



        self.shared_fc_layers = nn.ModuleList()

        for i in range(self.rot_num):
            GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layers.append(nn.Sequential(*shared_fc_list))
            break

        self.shared_fc_layers_mm = nn.ModuleList()

        for i in range(self.rot_num):
            GRID_SIZE = self.model_cfg.ROI_GRID_POOL_MM.GRID_SIZE
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_mm
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layers_mm.append(nn.Sequential(*shared_fc_list))
            break

        self.shared_channel = pre_channel

        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = self.model_cfg.SHARED_FC[-1] * 2 * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2 * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers.append(reg_fc_layers)
            break

        self.cls_layers_P = nn.ModuleList()
        self.reg_layers_P = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers_P.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers_P.append(reg_fc_layers)
            break

        self.cls_layers_PI = nn.ModuleList()
        self.reg_layers_PI = nn.ModuleList()

        for i in range(self.rot_num):
            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers_PI.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1] * 2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers_PI.append(reg_fc_layers)
            break


        if self.model_cfg.get('PART', False):
            self.grid_offsets = self.model_cfg.PART.GRID_OFFSETS
            self.featmap_stride = self.model_cfg.PART.FEATMAP_STRIDE
            part_inchannel = self.model_cfg.PART.IN_CHANNEL
            self.num_parts = self.model_cfg.PART.SIZE ** 2

            self.conv_part = nn.Sequential(
                nn.Conv2d(part_inchannel, part_inchannel, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(part_inchannel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(part_inchannel, self.num_parts, 1, 1, padding=0, bias=False),
            )
            self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets,
                                   spatial_scale=1 / self.featmap_stride)

        self.cross_attention_layers = nn.ModuleList()
        for i in range(self.rot_num):
            this_mo = CrossAttention(self.shared_channel)
            # print(count_parameters(this_mo))
            # input()
            self.cross_attention_layers.append(this_mo)
            break


        self.cross_attention_layers_mm = nn.ModuleList()
        for i in range(self.rot_num):
            this_mo = CrossAttention(self.shared_channel)
            # print(count_parameters(this_mo))
            # input()
            self.cross_attention_layers_mm.append(this_mo)
            break


        self.init_weights()
        self.ious = {0: [], 1: [], 2: [], 3: []}

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                for m in stage_module.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        for module_list in [self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                nn.init.normal_(stage_module[-1].weight, 0, 0.01)
                nn.init.constant_(stage_module[-1].bias, 0)
        for m in self.shared_fc_layers.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes)
                out = bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1, 1)
                confi.append(x)

        confi = torch.cat(confi)

        return confi

    def roi_part_pool(self, batch_dict, parts_feat):
        rois = batch_dict['rois_score'].clone()
        confi_preds = self.obtain_conf_preds(parts_feat, rois)

        return confi_preds

    def roi_grid_pool(self, batch_dict, i):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        if i==0:
            rot_num_id = ''
        else:
            rot_num_id = str(i)

        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                j=i
                while 'multi_scale_3d_features'+rot_num_id not in batch_dict:
                    j-=1
                    rot_num_id = str(j)

                cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]

                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features
    def roi_grid_pool_mm(self, batch_dict, i):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        if i==0:
            rot_num_id = ''
        else:
            rot_num_id = str(i)

        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg_mm.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg_mm.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers_mm[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    if 'multi_scale_3d_features_mm'+rot_num_id in batch_dict:
                        cur_sp_tensors = batch_dict['multi_scale_3d_features_mm'+rot_num_id][src_name]
                    else:
                        cur_sp_tensors = batch_dict['multi_scale_3d_features' + rot_num_id][src_name]
                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg_mm.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def roi_x_trans(self, rois, rot_num_id, transform_param):
        while rot_num_id>=len(transform_param[0]):
            rot_num_id-=1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []


        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_stage_param = bt_transform_param[rot_num_id-1]
            current_stage_param = bt_transform_param[rot_num_id]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': previous_stage_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': current_stage_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def roi_score_trans(self, rois, rot_num_id, transform_param):
        while rot_num_id>=len(transform_param[0]):
            rot_num_id-=1

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_stage_param = bt_transform_param[0]
            current_stage_param = bt_transform_param[rot_num_id]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': current_stage_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': previous_stage_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)

    def pred_x_trans(self, preds, rot_num_id, transform_param):
        while rot_num_id>=len(transform_param[0]):
            rot_num_id-=1

        batch_size = len(preds)
        preds = preds.clone()

        x_transformed_roi = []

        for bt_i in range(batch_size):

            cur_roi = preds[bt_i]
            bt_transform_param = transform_param[bt_i]
            current_stage_param = bt_transform_param[rot_num_id]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': current_stage_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return torch.stack(x_transformed_roi)


    def multi_grid_pool_aggregation(self, batch_dict, targets_dict):


        if self.model_cfg.get('PART', False):
            feat_2d = batch_dict['st_features_2d']

            parts_feat = self.conv_part(feat_2d)

        all_preds = []
        all_scores = []
        all_rois = []

        all_shared_features = []

        all_shared_features_mm = []

        for i in range(self.rot_num):

            rot_num_id = str(i)

            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois'] = self.roi_x_trans(batch_dict['rois'], i, batch_dict['transform_param'])

            if self.training:
                targets_dict = self.assign_targets(batch_dict, i)
                targets_dict['aug_param'] = batch_dict['aug_param']
                targets_dict['image_shape'] = batch_dict['image_shape']
                targets_dict['calib'] = batch_dict['calib']
                batch_dict['rois'] = targets_dict['rois']

                batch_dict['roi_labels'] = targets_dict['roi_labels']


            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois_score'] = self.roi_score_trans(batch_dict['rois'], i, batch_dict['transform_param'])
            else:
                batch_dict['rois_score'] = batch_dict['rois']
            if self.model_cfg.get('PART', False):
                part_scores = self.roi_part_pool(batch_dict, parts_feat)


            if 'transform_param' in batch_dict:
                pooled_features = self.roi_grid_pool(batch_dict, i)
                pooled_features_mm = self.roi_grid_pool_mm(batch_dict, i)
            else:
                pooled_features = self.roi_grid_pool(batch_dict, 0)
                pooled_features_mm = self.roi_grid_pool_mm(batch_dict, 0)

            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            shared_features = self.shared_fc_layers[0](pooled_features)
            shared_features = shared_features.unsqueeze(0)  # 1,B,C
            all_shared_features.append(shared_features)
            pre_feat = torch.cat(all_shared_features, 0)
            cur_feat = self.cross_attention_layers[0](pre_feat, shared_features)
            cur_feat = torch.cat([cur_feat, shared_features], -1)
            cur_feat = cur_feat.squeeze(0)  # B, C*2


            pooled_features_mm = pooled_features_mm.view(pooled_features_mm.size(0), -1)
            shared_features_mm = self.shared_fc_layers_mm[0](pooled_features_mm)
            shared_features_mm = shared_features_mm.unsqueeze(0)  # 1,B,C
            all_shared_features_mm.append(shared_features_mm)
            pre_feat_mm = torch.cat(all_shared_features_mm, 0)
            cur_feat_mm = self.cross_attention_layers_mm[0](pre_feat_mm, shared_features_mm)
            cur_feat_mm = torch.cat([cur_feat_mm, shared_features_mm], -1)
            cur_feat_mm = cur_feat_mm.squeeze(0)  # B, C*2

            final_feat = torch.cat([cur_feat_mm, cur_feat],-1)
            rcnn_cls = self.cls_layers[0](final_feat)
            rcnn_reg = self.reg_layers[0](final_feat)
            rcnn_cls_pi = self.cls_layers_PI[0](cur_feat_mm)
            rcnn_reg_pi = self.reg_layers_PI[0](cur_feat_mm)
            rcnn_cls_p = self.cls_layers_P[0](cur_feat)
            rcnn_reg_p = self.reg_layers_P[0](cur_feat)


            if self.model_cfg.get('PART', False):
                rcnn_cls = rcnn_cls+part_scores
                rcnn_cls_pi = rcnn_cls_pi+part_scores
                rcnn_cls_p = rcnn_cls_p+part_scores

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            outs = batch_box_preds.clone()
            if 'transform_param' in batch_dict:
                outs = self.pred_x_trans(outs, i, batch_dict['transform_param'])
            all_preds.append(outs)
            all_scores.append(batch_cls_preds)

            if self.training:
                targets_dict_pi = copy.deepcopy(targets_dict)
                targets_dict_p = copy.deepcopy(targets_dict)
                targets_dict['rcnn_cls'] = rcnn_cls
                targets_dict['rcnn_reg'] = rcnn_reg
                targets_dict_pi['rcnn_cls'] = rcnn_cls_pi
                targets_dict_pi['rcnn_reg'] = rcnn_reg_pi
                targets_dict_p['rcnn_cls'] = rcnn_cls_p
                targets_dict_p['rcnn_reg'] = rcnn_reg_p

                self.forward_ret_dict['targets_dict' + rot_num_id] = targets_dict
                self.forward_ret_dict['targets_dict_pi' + rot_num_id] = targets_dict_pi
                self.forward_ret_dict['targets_dict_p' + rot_num_id] = targets_dict_p

            batch_dict['rois'] = batch_box_preds
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

        return torch.mean(torch.stack(all_preds), 0), torch.mean(torch.stack(all_scores), 0)

    def forward(self, batch_dict):

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            self.rot_num = trans_param.shape[1]

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        boxes, scores = self.multi_grid_pool_aggregation(batch_dict, targets_dict)
        if not self.training:
            batch_dict['batch_box_preds'] = boxes
            batch_dict['batch_cls_preds'] = scores

        return batch_dict