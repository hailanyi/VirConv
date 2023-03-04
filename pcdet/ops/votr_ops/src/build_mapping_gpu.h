/*
Building xyz -> idx sparse tensor mapping
Written by Jiageng Mao
*/

#ifndef BUILD_MAPPING_GPU_H
#define BUILD_MAPPING_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int build_mapping_with_tensor_wrapper(int x_max, int y_max, int z_max, int num_voxels,
                                        at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor);

void build_mapping_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx);

int build_mapping_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                        at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor);

void build_mapping_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                                const int *v_indices, const int *v_bs_cnt, int *xyz_to_vidx);

int downsample_with_tensor_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                            int num_voxels, int num_ds_voxels,
                                            at::Tensor v_indices_tensor, at::Tensor ds_v_indices_tensor,
                                            at::Tensor xyz_to_vidx_tensor, at::Tensor vcount_tensor);

void downsample_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int num_ds_voxels,
                                                const int *v_indices, int *ds_v_indices,
                                                int *xyz_to_vidx, int *vcount);

int downsample_with_hash_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                            int num_voxels, int num_ds_voxels, int hash_size,
                                            at::Tensor v_indices_tensor, at::Tensor ds_v_indices_tensor,
                                            at::Tensor xyz_to_vidx_tensor, at::Tensor vcount_tensor);

void downsample_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int num_ds_voxels, int hash_size,
                                                const int *v_indices, int *ds_v_indices,
                                                int *xyz_to_vidx, int *vcount);
#endif