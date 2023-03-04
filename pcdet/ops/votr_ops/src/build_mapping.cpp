/*
Building xyz -> idx sparse tensor mapping
Written by Jiageng Mao
*/

#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "build_mapping_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int build_mapping_with_tensor_wrapper(int x_max, int y_max, int z_max, int num_voxels,
                                        at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(v_bs_cnt_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    const int *v_bs_cnt = v_bs_cnt_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    build_mapping_with_tensor_kernel_launcher(x_max, y_max, z_max, num_voxels, v_indices, v_bs_cnt, xyz_to_vidx);
    return 1;
}

int downsample_with_tensor_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                    int num_voxels, int num_ds_voxels,
                                    at::Tensor v_indices_tensor, at::Tensor ds_v_indices_tensor, at::Tensor xyz_to_vidx_tensor, at::Tensor vcount_tensor) {
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(ds_v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(vcount_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    int *ds_v_indices = ds_v_indices_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    int *vcount = vcount_tensor.data<int>();

    downsample_with_tensor_kernel_launcher(x_max, y_max, z_max, x_stride, y_stride, z_stride, num_voxels, num_ds_voxels,
                                                v_indices, ds_v_indices, xyz_to_vidx, vcount);
    return 1;
}

int build_mapping_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int hash_size,
                                        at::Tensor v_indices_tensor, at::Tensor v_bs_cnt_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(v_bs_cnt_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    const int *v_bs_cnt = v_bs_cnt_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    build_mapping_with_hash_kernel_launcher(x_max, y_max, z_max, num_voxels, hash_size, v_indices, v_bs_cnt, xyz_to_vidx);
    return 1;
}

int downsample_with_hash_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                    int num_voxels, int num_ds_voxels, int hash_size,
                                    at::Tensor v_indices_tensor, at::Tensor ds_v_indices_tensor, at::Tensor xyz_to_vidx_tensor, at::Tensor vcount_tensor) {
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(ds_v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(vcount_tensor);

    const int *v_indices = v_indices_tensor.data<int>();
    int *ds_v_indices = ds_v_indices_tensor.data<int>();
    int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    int *vcount = vcount_tensor.data<int>();

    downsample_with_hash_kernel_launcher(x_max, y_max, z_max, x_stride, y_stride, z_stride, num_voxels, num_ds_voxels, hash_size,
                                                v_indices, ds_v_indices, xyz_to_vidx, vcount);
    return 1;
}