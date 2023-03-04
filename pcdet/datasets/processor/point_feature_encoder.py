import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None, rot_num=1):
        super().__init__()
        self.rot_num=rot_num
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """

        for i in range(self.rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            data_dict['points'+rot_num_id], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['points'+rot_num_id]
            )

            if 'mm' in data_dict:
                data_dict['points_mm'+rot_num_id], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                    data_dict['points_mm'+rot_num_id]
                )

        data_dict['use_lead_xyz'] = use_lead_xyz

        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True

    def absolute_coordinates_encoding_mm(self, points=None):
        if points is None:
            num_output_features = self.point_encoding_config.num_features
            return num_output_features

        point_features = points[:, 0:self.point_encoding_config.num_features]
        return point_features, True
