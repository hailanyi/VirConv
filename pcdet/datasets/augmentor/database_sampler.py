import pathlib
import pickle

import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
import time
import copy
import random

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        #self.gt_path = pathlib.Path(sampler_cfg.GT_PATH)
        self.use_van = self.sampler_cfg.get('USE_VAN', None)
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
        if self.use_van:
            self.db_infos['Van'] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:

            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                for cls in class_names:
                    if cls in infos.keys():
                        self.db_infos[cls].extend(infos[cls])
                if self.use_van:
                    if 'Van' in infos.keys():
                        self.db_infos['Van'].extend(infos['Van'])

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                if not (self.use_van and class_name == 'Van'):
                    continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            this_infos = []
            for info in dinfos:
                if 'difficulty' in info:
                    if info['difficulty'] not in removed_difficulty:
                        this_infos.append(info)
                else:
                    this_infos.append(info)
            new_db_infos[key] = this_infos
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height


        return pre_obj_points, pre_box3d_lidar

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

        gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        if 'gt_tracklets' in data_dict:
            data_dict['gt_tracklets']=data_dict['gt_tracklets'][gt_boxes_mask]
        points = data_dict['points']
        if 'road_plane' in data_dict:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )


        obj_points_list = []

        for idx, info in enumerate(total_valid_sampled_dict):

            file_path = self.root_path / info['path']

            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if 'road_plane' in data_dict:
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if self.use_van:
            sampled_gt_names = np.array(['Car' if sampled_gt_names[i]=='Van' else sampled_gt_names[i] for i in range(len(sampled_gt_names))])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, 0:points.shape[1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        valid_mask = np.ones((len(gt_names),), dtype=np.bool_)
        valid_mask[:len(gt_names) - len(sampled_gt_names)] = 0

        data_dict['valid_noise'] = valid_mask
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes1 = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes1 = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)
                sampled_boxes = copy.deepcopy(sampled_boxes1)
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])

                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:

            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)


        return data_dict


class DADataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        # self.gt_path = pathlib.Path(sampler_cfg.GT_PATH)
        self.use_van = self.sampler_cfg.get('USE_VAN', None)
        self.min_sampling_dis = sampler_cfg.MIN_SAMPLING_DIS
        self.max_sampling_dis = sampler_cfg.MIN_SAMPLING_DIS
        self.occlusion_noise = sampler_cfg.OCCLUSION_NOISE
        self.occlusion_offset = sampler_cfg.OCCLUSION_OFFSET
        self.sampling_method = sampler_cfg.SAMPLING_METHOD
        self.vert_res = sampler_cfg.VERT_RES
        self.hor_res = sampler_cfg.HOR_RES

        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
        if self.use_van:
            self.db_infos['Van'] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:

            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                for cls in class_names:
                    if cls in infos.keys():
                        self.db_infos[cls].extend(infos[cls])
                # [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]
                if self.use_van:
                    if 'Van' in infos.keys():
                        self.db_infos['Van'].extend(infos['Van'])

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                if not (self.use_van and class_name == 'Van'):
                    continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def to_sphere_coords(self, points):
        r = np.linalg.norm(points[:, 0:3], ord=2, axis=-1)
        theta = np.arccos(points[:, 2] / r)
        fan = np.arctan(points[:, 1] / points[:, 0])

        new_points = copy.deepcopy(points)
        new_points[:, 0] = r
        new_points[:, 1] = theta
        new_points[:, 2] = fan

        return new_points

    def la_sampling(self, points, vert_res=0.006, hor_res=0.003):
        new_points = copy.deepcopy(points)

        sp_coords = self.to_sphere_coords(new_points)

        voxel_dict = {}

        for i, point in enumerate(sp_coords):

            vert_coord = point[1] // vert_res
            hor_coord = point[2] // hor_res

            voxel_key = str(vert_coord) + '_' + str(hor_coord)

            if voxel_key in voxel_dict:

                voxel_dict[voxel_key]['sp'].append(point)
                voxel_dict[voxel_key]['pts'].append(new_points[i])
            else:
                voxel_dict[voxel_key] = {'sp': [point], 'pts': [new_points[i]]}

        sampled_list = []

        for voxel_key in voxel_dict:
            sp = voxel_dict[voxel_key]['sp']
            arg_min = np.argmin(np.array(sp)[:, 1])
            min_point = voxel_dict[voxel_key]['pts'][arg_min]
            sampled_list.append(min_point)
        new_points = np.array(sampled_list)
        if len(new_points) < 5:
            return points
        else:
            return new_points

    def random_sampling(self, points, box, dis):
        new_points = copy.deepcopy(points)
        new_box = copy.deepcopy(box)
        x_off = dis
        y_off = 0  # np.random.randn()*10

        new_points[:, 0] -= new_box[0]
        new_points[:, 1] -= new_box[1]

        new_box[0] = x_off
        new_box[1] = y_off

        new_points[:, 0] += new_box[0]
        new_points[:, 1] += new_box[1]
        nn = random.choices(new_points.tolist(), k=int((1 - dis / 100) ** 3 * 300))
        return np.array(nn), new_box

    def random_drop_out(self, points, rand_noise=0.2, offset=0.3):

        rand = np.random.choice([0, 1, 2, 3])
        new_points = []
        for i, p in enumerate(points):
            if rand == 0 and p[1] + np.random.randn() * rand_noise < offset:
                new_points.append(points[i])
            if rand == 1 and p[1] + np.random.randn() * rand_noise >= -offset:
                new_points.append(points[i])
            if rand == 2 and p[2] + np.random.randn() * rand_noise < offset:
                new_points.append(points[i])
            if rand == 3 and p[2] + np.random.randn() * rand_noise >= -offset:
                new_points.append(points[i])

        new_points = np.array(new_points)
        if len(new_points) < 5:
            return self.random_drop_out(points, rand_noise, offset)

        return new_points

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            this_infos = []
            for info in dinfos:
                if 'difficulty' in info:
                    if info['difficulty'] not in removed_difficulty:
                        this_infos.append(info)
                else:
                    this_infos.append(info)
            new_db_infos[key] = this_infos
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height


    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

        gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        if 'gt_tracklets' in data_dict:
            data_dict['gt_tracklets'] = data_dict['gt_tracklets'][gt_boxes_mask]
        points = data_dict['points']
        if 'road_plane' in data_dict:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )

        obj_points_list = []


        for idx, info in enumerate(total_valid_sampled_dict):

            file_path = self.root_path / info['path']

            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += sampled_gt_boxes[idx][:3]

            if self.sampling_method == 'LiDAR-aware':

                obj_points = self.la_sampling(obj_points,
                                          vert_res=self.vert_res,
                                          hor_res=self.hor_res)
                obj_points[:, 0:3] -= sampled_gt_boxes[idx][:3]

                obj_points = self.random_drop_out(obj_points, rand_noise=self.occlusion_noise, offset=self.occlusion_offset)

                obj_points[:, 0:3] += sampled_gt_boxes[idx][:3]


            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, 0:points.shape[1]], points], axis=0)

        if self.use_van:
            sampled_gt_names = np.array(
                ['Car' if sampled_gt_names[i] == 'Van' else sampled_gt_names[i] for i in range(len(sampled_gt_names))])

        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        valid_mask = np.ones((len(gt_names),), dtype=np.bool_)
        if 'valid_noise' in data_dict:
            valid_mask[:len(gt_names) - len(sampled_gt_names)] = data_dict['valid_noise'][:]
        else:
            valid_mask[:len(gt_names) - len(sampled_gt_names)] = 0
        data_dict['valid_noise'] = valid_mask

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """

        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes1 = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes1 = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes1)
                sampled_boxes = copy.deepcopy(sampled_boxes1)
                sampled_boxes[:, 0] += np.random.random()*(self.max_sampling_dis-self.min_sampling_dis) + self.min_sampling_dis

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])

                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]
                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        if total_valid_sampled_dict.__len__() > 0:

            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        return data_dict
