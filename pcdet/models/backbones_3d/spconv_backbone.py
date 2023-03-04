from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn
import numpy as np
import torch
from pcdet.datasets.augmentor.X_transform import X_TRANS

def index2points(indices, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):
    """
    convert 3D voxel indices to a set of grid points.
    """

    voxel_size = np.array(voxel_size) * stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z

    return new_indices


def index2uv3d(indices, batch_size, calib, stride, x_trans_train, trans_param):
    """
    convert the 3D voxel indices to image pixel indices.
    """
    new_uv = indices.clone().int()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                         'transform_param': trans_param[b_i]})
            cur_pts = transed['points'].cpu().numpy()
        else:
            cur_pts = cur_pts[:, 1:4].cpu().numpy()

        pts_rect = calib[b_i].lidar_to_rect(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img(pts_rect)
        pts_img = pts_img.astype(np.int32)
        pts_img = torch.from_numpy(pts_img).to(new_uv.device)
        new_uv[indices[:, 0] == b_i, 2:4] = pts_img

    new_uv[:, 1] = 1
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=1400) // stride
    new_uv[:, 3] = torch.clamp(new_uv[:, 3], min=0, max=600) // stride
    return new_uv


def index2uv(indices, batch_size, calib, stride, x_trans_train, trans_param):
    """
    convert the 3D voxel indices to image pixel indices.
    """

    new_uv = indices.new(size=(indices.shape[0], 3))
    depth = indices.new(size=(indices.shape[0], 1)).float()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                         'transform_param': trans_param[b_i]})
            cur_pts = transed['points']  # .cpu().numpy()
        else:
            cur_pts = cur_pts[:, 1:4]  # .cpu().numpy()

        pts_rect = calib[b_i].lidar_to_rect_cuda(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img_cuda(pts_rect)

        pts_img = pts_img.int()
        # pts_img = torch.from_numpy(pts_img).to(new_uv.device)
        new_uv[indices[:, 0] == b_i, 1:3] = pts_img
        # pts_rect_depth = torch.from_numpy(pts_rect_depth).to(new_uv.device).float()
        depth[indices[:, 0] == b_i, 0] = pts_rect_depth[:]
    new_uv[:, 0] = indices[:, 0]
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max=1400 - 1) // stride
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=600 - 1) // stride

    return new_uv, depth


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def layer_voxel_discard(sparse_t, rat=0.15):
    """
    discard the voxels based on the given rate.
    """

    if rat == 0:
        return

    len = sparse_t.features.shape[0]
    randoms = np.random.permutation(len)
    randoms = torch.from_numpy(randoms[0:int(len * (1 - rat))]).to(sparse_t.features.device)

    sparse_t = replace_feature(sparse_t, sparse_t.features[randoms])
    sparse_t.indices = sparse_t.indices[randoms]


class NRConvBlock(nn.Module):
    """
    convolve the voxel features in both 3D and 2D space.
    """

    def __init__(self, input_c=16, output_c=16, stride=1, padding=1, indice_key='vir1', conv_depth=False):
        super(NRConvBlock, self).__init__()
        self.stride = stride
        block = post_act_block
        block2d = post_act_block2d
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_depth = conv_depth

        if self.stride > 1:
            self.down_layer = block(input_c,
                                    output_c,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=stride,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        c1 = input_c

        if self.stride > 1:
            c1 = output_c
        if self.conv_depth:
            c1 += 4

        c2 = output_c

        self.d3_conv1 = block(c1,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.d2_conv1 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm3' + indice_key))

        self.d3_conv2 = block(c2 // 2,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))
        self.d2_conv2 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm4' + indice_key))

    def forward(self, sp_tensor, batch_size, calib, stride, x_trans_train, trans_param):

        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)

        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)

        uv_coords, depth = index2uv(d3_feat2.indices, batch_size, calib, stride, x_trans_train, trans_param)
        # N*3,N*1
        d2_sp_tensor1 = spconv.SparseConvTensor(
            features=d3_feat2.features,
            indices=uv_coords.int(),
            spatial_shape=[1600, 600],
            batch_size=batch_size
        )

        d2_feat1 = self.d2_conv1(d2_sp_tensor1)
        d2_feat2 = self.d2_conv2(d2_feat1)

        d3_feat3 = replace_feature(d3_feat2, torch.cat([d3_feat2.features, d2_feat2.features], -1))

        return d3_feat3


class VirConv8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,  **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.x_trans_train = X_TRANS()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES
        self.layer_discard_rate = model_cfg.LAYER_DISCARD_RATE

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.vir_conv1 = NRConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
            self.vir_conv2 = NRConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
            self.vir_conv3 = NRConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
            self.vir_conv4 = NRConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0, 1, 1),
                                          indice_key='vir4')

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        # for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def decompose_tensor(self, tensor, i, batch_size):
        """
        decompose a sparse tensor by the given transformation index.
        """

        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_size = batch_dict['batch_size']
        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        if self.training:
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                    'voxel_coords' + rot_num_id]

                batch_size = batch_dict['batch_size']
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv4)

                batch_dict.update({
                    'encoded_spconv_tensor' + rot_num_id: out,
                    'encoded_spconv_tensor_stride' + rot_num_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + rot_num_id: {
                        'x_conv1': x_conv1,
                        'x_conv2': x_conv2,
                        'x_conv3': x_conv3,
                        'x_conv4': x_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        else:
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                    'voxel_coords' + rot_num_id]

                # serial processing to parallel processing for speeding up inference
                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += i * self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)

            input_sp_tensor = spconv.SparseConvTensor(
                features=all_lidar_feat,
                indices=all_lidar_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
                this_out = self.decompose_tensor(out, i, batch_size)

                batch_dict.update({
                    'encoded_spconv_tensor' + rot_num_id: this_out,
                    'encoded_spconv_tensor_stride' + rot_num_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        for i in range(rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm' + rot_num_id], batch_dict[
                    'voxel_coords_mm' + rot_num_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                if self.training:
                    layer_voxel_discard(newinput_sp_tensor, self.layer_discard_rate)

                calib = batch_dict['calib']
                batch_size = batch_dict['batch_size']
                if 'aug_param' in batch_dict:
                    trans_param = batch_dict['aug_param']
                else:
                    trans_param = None
                if 'transform_param' in batch_dict:
                    trans_param = batch_dict['transform_param'][:, i, :]

                newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv1, self.layer_discard_rate)

                newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv2, self.layer_discard_rate)

                newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv3, self.layer_discard_rate)

                newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm' + rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm' + rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict


class VirConvL8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,  **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.x_trans_train = X_TRANS()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES
        self.layer_discard_rate = model_cfg.LAYER_DISCARD_RATE

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.vir_conv1 = NRConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
        self.vir_conv2 = NRConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
        self.vir_conv3 = NRConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
        self.vir_conv4 = NRConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0, 1, 1), indice_key='vir4')

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        # for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def decompose_tensor(self, tensor, i, batch_size):
        """
        decompose a sparse tensor by the given transformation index.
        """

        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_size = batch_dict['batch_size']

        for i in range(rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            newvoxel_features, newvoxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                'voxel_coords' + rot_num_id]

            newvoxel_features[:, 4:7] = 0 # remove the useless RGB features
            newvoxel_features[:, 7] * 100 # highlight the indicator value regarding LiDAR and RGB point

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=newvoxel_features,
                indices=newvoxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            if self.training:
                layer_voxel_discard(newinput_sp_tensor, 0)

            calib = batch_dict['calib']
            batch_size = batch_dict['batch_size']
            if 'aug_param' in batch_dict:
                trans_param = batch_dict['aug_param']
            else:
                trans_param = None
            if 'transform_param' in batch_dict:
                trans_param = batch_dict['transform_param'][:, i, :]

            newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

            if self.training:
                layer_voxel_discard(newx_conv1, self.layer_discard_rate)

            newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

            if self.training:
                layer_voxel_discard(newx_conv2, self.layer_discard_rate)

            newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

            if self.training:
                layer_voxel_discard(newx_conv3, self.layer_discard_rate)

            newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

            out = self.conv_out(newx_conv4)

            batch_dict.update({
                'encoded_spconv_tensor' + rot_num_id: out,
                'encoded_spconv_tensor_stride' + rot_num_id: 8,
            })

            batch_dict.update({
                'encoded_spconv_tensor_stride' + rot_num_id: 8
            })
            batch_dict.update({
                'multi_scale_3d_features' + rot_num_id: {
                    'x_conv1': newx_conv1,
                    'x_conv2': newx_conv2,
                    'x_conv3': newx_conv3,
                    'x_conv4': newx_conv4,
                },
                'multi_scale_3d_strides' + rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

        return batch_dict


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict








