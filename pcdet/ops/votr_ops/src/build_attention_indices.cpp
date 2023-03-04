/*
Find indices for each attention pattern
Written by Jiageng Mao
*/

#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "build_attention_indices_gpu.h"

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

int sparse_local_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int attend_size, int attend_range,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    sparse_local_attention_with_tensor_kernel_launcher(x_max, y_max, z_max, x_stride, y_stride, z_stride, num_voxels, attend_size, attend_range,
                                                        attend_indices, v_indices, xyz_to_vidx);
    return 1;
}

int sparse_local_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int attend_size, int attend_range, int hash_size,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    sparse_local_attention_with_hash_kernel_launcher(x_max, y_max, z_max, x_stride, y_stride, z_stride, num_voxels, attend_size, attend_range, hash_size,
                                                        attend_indices, v_indices, xyz_to_vidx);
    return 1;
}

int subm_local_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    subm_local_attention_with_tensor_kernel_launcher(x_max, y_max, z_max, num_voxels, attend_size, attend_range,
                                                        attend_indices, v_indices, xyz_to_vidx);
    return 1;
}

int subm_local_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range, int hash_size,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();

    subm_local_attention_with_hash_kernel_launcher(x_max, y_max, z_max, num_voxels, attend_size, attend_range, hash_size,
                                                        attend_indices, v_indices, xyz_to_vidx);
    return 1;
}

int sparse_strided_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                    int num_voxels, int attend_size, int num_range,
                                                    at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                    at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(range_spec_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    const int *range_spec = range_spec_tensor.data<int>();

    sparse_strided_attention_with_tensor_kernel_launcher(x_max, y_max, z_max, x_stride, y_stride, z_stride, num_voxels, attend_size, num_range,
                                                       attend_indices, v_indices, xyz_to_vidx, range_spec);
    return 1;
}

int sparse_strided_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                    int num_voxels, int attend_size, int num_range, int hash_size,
                                                    at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                    at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(range_spec_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    const int *range_spec = range_spec_tensor.data<int>();

    sparse_strided_attention_with_hash_kernel_launcher(x_max, y_max, z_max, x_stride, y_stride, z_stride, num_voxels, attend_size, num_range, hash_size,
                                                       attend_indices, v_indices, xyz_to_vidx, range_spec);
    return 1;
}

int subm_strided_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(range_spec_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    const int *range_spec = range_spec_tensor.data<int>();

    subm_strided_attention_with_tensor_kernel_launcher(x_max, y_max, z_max, num_voxels, attend_size, num_range,
                                                       attend_indices, v_indices, xyz_to_vidx, range_spec);
    return 1;
}

int subm_strided_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor) {
    CHECK_INPUT(attend_indices_tensor);
    CHECK_INPUT(v_indices_tensor);
    CHECK_INPUT(xyz_to_vidx_tensor);
    CHECK_INPUT(range_spec_tensor);

    int *attend_indices = attend_indices_tensor.data<int>();
    const int *v_indices = v_indices_tensor.data<int>();
    const int *xyz_to_vidx = xyz_to_vidx_tensor.data<int>();
    const int *range_spec = range_spec_tensor.data<int>();

    subm_strided_attention_with_hash_kernel_launcher(x_max, y_max, z_max, num_voxels, attend_size, num_range, hash_size,
                                                       attend_indices, v_indices, xyz_to_vidx, range_spec);
    return 1;
}