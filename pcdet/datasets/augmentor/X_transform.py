from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils
import copy


class X_TRANS(object):
    def __init__(self, augmentor_configs=None, rot_num=1):
        self.rot_num = rot_num
        self.data_augmentor_queue = []
        self.test_back_queue = []
        if augmentor_configs is None:
            augmentor_configs=[{'NAME': 'world_rotation',
                                'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
                               {'NAME': 'world_flip',
                                'ALONG_AXIS_LIST': [0, 1]},
                               {'NAME': 'world_scaling',
                               'WORLD_SCALE_RANGE': [0.95, 1.05]}]
            self.augmentor_configs = augmentor_configs
        else:
            self.augmentor_configs = augmentor_configs
        self.aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for i, cur_cfg in enumerate(self.aug_config_list):
            cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
            back_config = self.aug_config_list[-(i+1)]
            cur_augmentor = getattr(self, back_config['NAME'])(config=back_config)
            self.test_back_queue.append(cur_augmentor)

        self.backward_flag = False

    def get_params(self):
        transform_param = np.zeros(shape=(self.rot_num, len(self.aug_config_list)))
        for s in range(self.rot_num):
            for i, config in enumerate(self.aug_config_list):
                if config.NAME == 'world_rotation':
                    transform_param[s][i] = config.WORLD_ROT_ANGLE[s]
                if config.NAME == 'world_flip':
                    transform_param[s][i] = config.ALONG_AXIS_LIST[s]
                if config.NAME == 'world_scaling':
                    transform_param[s][i] = config.WORLD_SCALE_RANGE[s]
        return transform_param

    def world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_rotation, config=config)

        rot_factor = data_dict['transform_param'][0]
        if isinstance(rot_factor, np.float64):
            rot_factor = np.array([rot_factor])
        else:
            rot_factor = rot_factor.unsqueeze(0)

        if 'points' in data_dict:
            points = data_dict['points']
            if self.backward_flag:
                points[:,0:3] = common_utils.rotate_points_along_z(points[np.newaxis, :, 0:3], -rot_factor)[0]
            else:
                points[:, 0:3] = common_utils.rotate_points_along_z(points[np.newaxis, :, 0:3], rot_factor)[0]
            data_dict['points'] = points

        if 'boxes' in data_dict:
            boxes_lidar = data_dict['boxes']
            if self.backward_flag:
                boxes_lidar[:, 0:3] = common_utils.rotate_points_along_z(boxes_lidar[np.newaxis, :, 0:3], -rot_factor)[0]
                boxes_lidar[:, 6] += -rot_factor
            else:
                boxes_lidar[:, 0:3] = common_utils.rotate_points_along_z(boxes_lidar[np.newaxis, :, 0:3], rot_factor)[0]
                boxes_lidar[:, 6] += rot_factor
            data_dict['boxes'] = boxes_lidar

        return data_dict

    def world_flip(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.world_flip, config=config)

        if 'points' in data_dict:
            points = getattr(augmentor_utils, 'random_flip_with_param')(
                data_dict['points'], data_dict['transform_param'][1], ax=1)
            data_dict['points'] = points

        if 'boxes' in data_dict:
            boxes = getattr(augmentor_utils, 'random_flip_with_param')(
                data_dict['boxes'], data_dict['transform_param'][1], ax=1)
            boxes = getattr(augmentor_utils, 'random_flip_with_param')(
                boxes, data_dict['transform_param'][1], ax=6)
            data_dict['boxes'] = boxes

        return data_dict

    def world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_scaling, config=config)
        scale_factor = data_dict['transform_param'][2]

        if 'points' in data_dict:
            points = data_dict['points']
            if self.backward_flag:
                points[:, 0:3] /= scale_factor
            else:
                points[:, 0:3] *= scale_factor

            data_dict['points'] = points

        if 'boxes' in data_dict:
            boxes_lidar = data_dict['boxes']
            if self.backward_flag:
                boxes_lidar[:, 0:6] /= scale_factor
            else:
                boxes_lidar[:, 0:6] *= scale_factor
            data_dict['boxes'] = boxes_lidar

        return data_dict

    def forward_with_param(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict

    def backward_with_param(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        self.backward_flag = True
        for cur_augmentor in self.test_back_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        self.backward_flag = False
        return data_dict

    def input_transform(self, data_dict, trans_boxes=False):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        params = self.get_params()

        src_points = copy.deepcopy(data_dict['points'])
        if trans_boxes:
            src_gt_boxes = copy.deepcopy(data_dict['gt_boxes'])

        for i in range(self.rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            ini_data_dict = {}
            ini_data_dict['points'] = copy.deepcopy(src_points)
            if trans_boxes:
                ini_data_dict['boxes'] = copy.deepcopy(src_gt_boxes)

            ini_data_dict['transform_param'] = copy.deepcopy(params[i])

            transformed_data = self.forward_with_param(ini_data_dict)

            data_dict['points'+rot_num_id] = transformed_data['points']
            if trans_boxes:
                data_dict['gt_boxes'+rot_num_id] = transformed_data['boxes']

        data_dict['transform_param'] = params

        return data_dict
