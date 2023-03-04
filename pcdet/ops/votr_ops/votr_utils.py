import torch
from torch.autograd import Function, Variable

from . import votr_ops_cuda as votr

class BuildTensorTable(Function):

    @staticmethod
    def forward(ctx, batch_size, spatial_shape, voxel_indices, v_bs_cnt):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        x_max, y_max, z_max = spatial_shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        dense_map = torch.zeros((batch_size, x_max, y_max, z_max)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)

        votr.build_mapping_with_tensor_wrapper(x_max, y_max, z_max, num_voxels, voxel_indices, v_bs_cnt, dense_map)
        return dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

build_tensor_table = BuildTensorTable.apply

class BuildHashTable(Function):

    @staticmethod
    def forward(ctx, batch_size, hash_size, spatial_shape, voxel_indices, v_bs_cnt):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        x_max, y_max, z_max = spatial_shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        dense_map = torch.zeros((batch_size, hash_size, 2)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)

        votr.build_mapping_with_hash_wrapper(x_max, y_max, z_max, num_voxels, hash_size, voxel_indices, v_bs_cnt, dense_map)
        return dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

build_hash_table = BuildHashTable.apply

class TensorDownSample(Function):
    @staticmethod
    def forward(ctx, strides, num_ds_voxels, batch_size, spatial_shape, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        x_max, y_max, z_max = spatial_shape
        dense_map = torch.zeros((batch_size, x_max, y_max, z_max)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        ds_voxel_indices = torch.zeros((batch_size, num_ds_voxels, 3)).int().fill_(-1).to(voxel_indices.device)
        vcount = torch.zeros(batch_size).int().to(voxel_indices.device)
        votr.downsample_with_tensor_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                num_voxels, num_ds_voxels,
                                                voxel_indices, ds_voxel_indices, dense_map, vcount)
        ds_voxel_list = []
        for i in range(batch_size):
            ds_voxel = ds_voxel_indices[i]
            ds_voxel = ds_voxel[ds_voxel[:, 0] >= 0] # not -1
            bs_idx = torch.zeros((ds_voxel.shape[0], 1)).int().fill_(i).to(voxel_indices.device)
            ds_voxel = torch.cat([bs_idx, ds_voxel], dim = 1)
            ds_voxel_list.append(ds_voxel)

        output_voxels = torch.cat(ds_voxel_list, dim = 0).contiguous()
        return output_voxels, dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

tensor_down_sample = TensorDownSample.apply

class HashTableDownSample(Function):
    @staticmethod
    def forward(ctx, strides, num_ds_voxels, batch_size, hash_size, spatial_shape, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        x_max, y_max, z_max = spatial_shape
        dense_map = torch.zeros((batch_size, hash_size, 2)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        ds_voxel_indices = torch.zeros((batch_size, num_ds_voxels, 3)).int().fill_(-1).to(voxel_indices.device)
        vcount = torch.zeros(batch_size).int().to(voxel_indices.device)
        votr.downsample_with_hash_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                num_voxels, num_ds_voxels, hash_size,
                                                voxel_indices, ds_voxel_indices, dense_map, vcount)
        ds_voxel_list = []
        for i in range(batch_size):
            ds_voxel = ds_voxel_indices[i]
            ds_voxel = ds_voxel[ds_voxel[:, 0] >= 0] # not -1
            bs_idx = torch.zeros((ds_voxel.shape[0], 1)).int().fill_(i).to(voxel_indices.device)
            ds_voxel = torch.cat([bs_idx, ds_voxel], dim = 1)
            ds_voxel_list.append(ds_voxel)

        output_voxels = torch.cat(ds_voxel_list, dim = 0).contiguous()
        return output_voxels, dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

hash_table_down_sample = HashTableDownSample.apply

class SparseLocalAttentionTensorIndices(Function):

    @staticmethod
    def forward(ctx, attend_size, attend_range, strides, dense_map, voxel_indices):
        """
        Args:
            ctx:
            dense_map: (bs_idx, x_max, y_max, z_max) -> old map table
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x) -> new downsampled indices
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        batch_size, x_max, y_max, z_max = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()

        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.sparse_local_attention_with_tensor_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                        num_voxels, attend_size, attend_range,
                                                        attend_indices, voxel_indices, dense_map)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

sparse_local_attention_tensor_indices = SparseLocalAttentionTensorIndices.apply

class SparseLocalAttentionHashIndices(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, attend_range, strides, dense_map, voxel_indices):
        """
        Args:
            ctx:
            dense_map: (bs_idx, hash_size, 2) -> old map table
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x) -> new downsampled indices
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        x_max, y_max, z_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()

        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.sparse_local_attention_with_hash_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                        num_voxels, attend_size, attend_range, hash_size,
                                                        attend_indices, voxel_indices, dense_map)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

sparse_local_attention_hash_indices = SparseLocalAttentionHashIndices.apply

class SparseStridedAttentionTensorIndices(Function):

    @staticmethod
    def forward(ctx, attend_size, range_spec, strides, dense_map, voxel_indices):
        """
        Args:
            ctx:
            dense_map: (bs_idx, x_max, y_max, z_max) -> old map table
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x) -> new downsampled indices
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        batch_size, x_max, y_max, z_max = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        range_spec = torch.tensor(range_spec).int().to(voxel_indices.device)
        num_range = range_spec.shape[0]
        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.sparse_strided_attention_with_tensor_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                            num_voxels, attend_size, num_range,
                                                            attend_indices, voxel_indices,
                                                            dense_map, range_spec)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

sparse_strided_attention_tensor_indices = SparseStridedAttentionTensorIndices.apply

class SparseStridedAttentionHashIndices(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, range_spec, strides, dense_map, voxel_indices):
        """
        Args:
            ctx:
            dense_map: (bs_idx, x_max, y_max, z_max) -> old map table
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x) -> new downsampled indices
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        x_max, y_max, z_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        range_spec = torch.tensor(range_spec).int().to(voxel_indices.device)
        num_range = range_spec.shape[0]
        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.sparse_strided_attention_with_hash_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                            num_voxels, attend_size, num_range, hash_size,
                                                            attend_indices, voxel_indices,
                                                            dense_map, range_spec)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

sparse_strided_attention_hash_indices = SparseStridedAttentionHashIndices.apply

class SubMLocalAttentionTensorIndices(Function):

    @staticmethod
    def forward(ctx, attend_size, attend_range, dense_map, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """

        batch_size, x_max, y_max, z_max = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()

        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.subm_local_attention_with_tensor_wrapper(x_max, y_max, z_max,
                                                        num_voxels, attend_size, attend_range,
                                                        attend_indices, voxel_indices, dense_map)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

subm_local_attention_tensor_indices = SubMLocalAttentionTensorIndices.apply

class SubMLocalAttentionHashIndices(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, attend_range, dense_map, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """

        x_max, y_max, z_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()

        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.subm_local_attention_with_hash_wrapper(x_max, y_max, z_max,
                                                        num_voxels, attend_size, attend_range, hash_size,
                                                        attend_indices, voxel_indices, dense_map)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

subm_local_attention_hash_indices = SubMLocalAttentionHashIndices.apply

class SubMStridedAttentionTensorIndices(Function):

    @staticmethod
    def forward(ctx, attend_size, range_spec, dense_map, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        batch_size, x_max, y_max, z_max = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        range_spec = torch.tensor(range_spec).int().to(voxel_indices.device)
        num_range = range_spec.shape[0]
        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.subm_strided_attention_with_tensor_wrapper(x_max, y_max, z_max,
                                                            num_voxels, attend_size, num_range,
                                                            attend_indices, voxel_indices,
                                                            dense_map, range_spec)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

subm_strided_attention_tensor_indices = SubMStridedAttentionTensorIndices.apply

class SubMStridedAttentionHashIndices(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, range_spec, dense_map, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        x_max, y_max, z_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        range_spec = torch.tensor(range_spec).int().to(voxel_indices.device)
        num_range = range_spec.shape[0]
        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.subm_strided_attention_with_hash_wrapper(x_max, y_max, z_max,
                                                            num_voxels, attend_size, num_range, hash_size,
                                                            attend_indices, voxel_indices,
                                                            dense_map, range_spec)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

subm_strided_attention_hash_indices = SubMStridedAttentionHashIndices.apply

class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample).zero_()

        votr.group_features_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        votr.group_features_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None

grouping_operation = GroupingOperation.apply