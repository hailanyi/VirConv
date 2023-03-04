from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import torch.utils.data as torch_data
import os
from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .augmentor.X_transform import X_TRANS
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
import copy
import time

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, is_source=True, root_path=None, logger=None,
                 da_train=False):
        super().__init__()
        self.test_flip = False
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.is_source = is_source
        self.da_train = da_train
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        if self.dataset_cfg is None or class_names is None:
            return
        if self.training:
            self.rot_num = 1
        else:
            self.rot_num = self.dataset_cfg.get('ROT_NUM', 1)


        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range,
            rot_num=self.rot_num
        )

        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger,
        ) if self.training else None


        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training,
            rot_num=self.rot_num, num_point_features=self.point_feature_encoder.num_point_features
        )

        x_trans_cfg = self.dataset_cfg.get('X_TRANS', None)
        if x_trans_cfg is not None:
            self.x_trans = X_TRANS(x_trans_cfg, rot_num=self.rot_num)
        else:
            self.x_trans = None

        self.input_discard_rate = self.dataset_cfg.get('INPUT_DISCARD_RATE', 0.8)

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def partition(self, points, num=10, max_dis=60, rate=0.2):
        """
        partition the points into several bins.
        """

        points_list = []
        inter = max_dis / num

        all_points_num = points.shape[0]

        points_num_acc = 0

        position = num - 1

        distant_points_num_acc = 0

        for i in range(num):
            i = num - i - 1
            if i == num - 1:
                min_mask = points[:, 0] >= inter * i
                this_points = points[min_mask]

                points_num_acc += this_points.shape[0]

                sampled_sum = points_num_acc + i * this_points.shape[0]

                if sampled_sum / all_points_num < rate:
                    position = i
                    distant_points_num_acc = points_num_acc

                points_list.append(this_points)
            else:
                min_mask = points[:, 0] >= inter * i
                max_mask = points[:, 0] < inter * (i + 1)
                mask = min_mask * max_mask
                this_points = points[mask]

                points_num_acc += this_points.shape[0]

                sampled_sum = points_num_acc + i * this_points.shape[0]

                if sampled_sum / all_points_num < rate:
                    position = i
                    distant_points_num_acc = points_num_acc

                points_list.append(this_points)

        if position <= 0:
            position = 0

        return points_list, position, distant_points_num_acc

    def input_point_discard(self, points, bin_num=2, rate=0.8):
        """
        discard points by a bin-based sampling.
        """
        retain_rate = 1 - rate
        parts, pos, distant_points_num_acc = self.partition(points, num=bin_num, rate=retain_rate)

        output_pts_num = int(points.shape[0] * retain_rate)

        pts_per_bin_num = int((output_pts_num-distant_points_num_acc)/(pos+0.0001))

        for i in range(len(parts) - pos, len(parts)):

            if parts[i].shape[0] > pts_per_bin_num:
                rands = np.random.permutation(parts[i].shape[0])
                parts[i] = parts[i][rands[:pts_per_bin_num]]

        return np.concatenate(parts)

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')


        else:
            if self.x_trans is not None:
                data_dict = self.x_trans.input_transform(
                    data_dict={
                        **data_dict,
                    }
                )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            for i in range(self.rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)
                if 'gt_boxes'+rot_num_id in data_dict:
                    data_dict['gt_boxes'+rot_num_id] = data_dict['gt_boxes'+rot_num_id][selected]
                    gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                    gt_boxes = np.concatenate((data_dict['gt_boxes'+rot_num_id], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                    data_dict['gt_boxes'+rot_num_id] = gt_boxes

        for i in range(self.rot_num):
            if i ==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            # swap the scene for augmentation
            if self.training and np.random.choice([0,1]):
                randx = np.random.random()*70.4
                randx_1 = 70.4-randx
                points = data_dict['points'+rot_num_id]
                points[points[:,0]>70.4]=0
                points1 = points[points[:,0]<=randx]
                points1[:, 0]+=randx_1
                points2 = points[points[:,0]>randx]
                points2[:, 0]-= randx
                data_dict['points'+rot_num_id] = np.concatenate([points1,points2])
                boxes = data_dict['gt_boxes'+rot_num_id]
                boxes1 = boxes[boxes[:,0]<=randx]
                boxes1[:,0]+=randx_1
                boxes2 = boxes[boxes[:,0]>randx]
                boxes2[:,0]-=randx
                data_dict['gt_boxes'+rot_num_id] = np.concatenate([boxes1, boxes2 ])

            if 'mm' in data_dict:
                if self.dataset_cfg.get('LATER_FUSION', True):
                    points_mm = data_dict['points' + rot_num_id][data_dict['points' + rot_num_id][:, -1] == 1]
                    points = data_dict['points'+rot_num_id][data_dict['points'+rot_num_id][:, -1] == 2]

                    if self.training:
                        points_mm2 = self.input_point_discard(points_mm, rate=self.input_discard_rate)
                    else:
                        points_mm2 = self.input_point_discard(points_mm, bin_num=10, rate=self.input_discard_rate)

                    data_dict['points_mm'+rot_num_id] = points_mm2
                    data_dict['points'+rot_num_id] = points

                else:
                    points_mm = data_dict['points' + rot_num_id][data_dict['points' + rot_num_id][:, -1] == 1]
                    points = data_dict['points' + rot_num_id][data_dict['points' + rot_num_id][:, -1] == 2]

                    if self.training:
                        points_mm2 = self.input_point_discard(points_mm, rate=self.input_discard_rate)
                    else:
                        points_mm2 = self.input_point_discard(points_mm, bin_num=10, rate=self.input_discard_rate)

                    final_points = np.concatenate([points, points_mm2])
                    data_dict['points' + rot_num_id] = final_points
                    data_dict['points' + rot_num_id][:, 3]/=10


        if (not self.dataset_cfg.get('LATER_FUSION', True)) and 'mm' in data_dict:
            data_dict.pop('mm')


        data_dict = self.point_feature_encoder.forward(data_dict)
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        if 'valid_noise' in data_dict:
            data_dict.pop('valid_noise')
        return data_dict

    def collate_batch(self, batch_list, _unused=False):

        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        point_key_dict=['points', 'voxel_coords', 'points_mm', 'voxel_coords_mm']
        for i in range(1, 10):

            point_key_dict.append('points'+str(i))
            point_key_dict.append('voxel_coords'+str(i))
            point_key_dict.append('points_mm'+str(i))
            point_key_dict.append('voxel_coords_mm'+str(i))

        voxel_key_dict=['voxels', 'voxel_num_points', 'voxels_mm', 'voxel_num_points_mm']
        for i in range(1, 10):
            voxel_key_dict.append('voxels'+str(i))
            voxel_key_dict.append('voxel_num_points' + str(i))
            voxel_key_dict.append('voxels_mm'+str(i))
            voxel_key_dict.append('voxel_num_points_mm' + str(i))

        boxes_key = ['gt_boxes']
        for i in range(1, 10):
            boxes_key.append('gt_boxes'+str(i))


        for key, val in data_dict.items():
            try:
                if key in voxel_key_dict:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in point_key_dict:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in boxes_key:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret