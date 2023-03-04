/*
Find indices for each attention pattern
Written by Jiageng Mao
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "build_attention_indices_gpu.h"
#include "votr_cuda_utils.h"

__device__ int simple_hash(int k, int hash_size) {
    return k % hash_size;
}

__device__ int hash_table_find(int &key, int &hash_size, const int *xyz_to_vidx) {
    int hash_idx = simple_hash(key, hash_size);
    int v_idx = EMPTY_KEY;
    int prob_cnt = 0;
    while (true) {
        // found
        if (xyz_to_vidx[hash_idx * 2 + 0] == key) {
            v_idx = xyz_to_vidx[hash_idx * 2 + 1];
            break;
        }
        // empty, not found
        if (xyz_to_vidx[hash_idx * 2 + 0] == EMPTY_KEY) {
            break;
        }
        // linear probing
        hash_idx = (hash_idx + 1) % hash_size;
        // security in case of dead loop
        prob_cnt += 1;
        if (prob_cnt >= hash_size) break;
    }
    return v_idx;
}

__global__ void sparse_local_attention_with_tensor_kernel(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int attend_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx) {
    /*
        in sparse attention, voxels are not necessary at the non-empty location
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, x_max, y_max, z_max] voxel coordinates to voxel indices
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    xyz_to_vidx += bs_idx * x_max * y_max * z_max;

    int num_samples = 0;
    for (int sz_idx = z_idx * z_stride - attend_range; sz_idx <= z_idx * z_stride + (z_stride - 1) + attend_range; ++sz_idx){
        if (sz_idx >= z_max || sz_idx < 0) continue;
        for (int sy_idx = y_idx * y_stride - attend_range; sy_idx <= y_idx * y_stride + (y_stride - 1) + attend_range; ++sy_idx){
            if (sy_idx >= y_max || sy_idx < 0) continue;
            for (int sx_idx = x_idx * x_stride - attend_range; sx_idx <= x_idx * x_stride + (x_stride - 1) + attend_range; ++sx_idx){
                if (sx_idx >= x_max || sx_idx < 0) continue;
                int sv_idx = xyz_to_vidx[sx_idx * y_max * z_max + sy_idx * z_max + sz_idx];
                if (sv_idx != EMPTY_KEY) { // found non-empty index
                    if (num_samples >= attend_size) return; // full and return
                    attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                    num_samples++;
                }else { // not found
                    ;
                }
            }
        }
    }
    return;
}

void sparse_local_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                        int num_voxels, int attend_size, int attend_range,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    sparse_local_attention_with_tensor_kernel<<<blocks, threads>>>(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                                    num_voxels, attend_size, attend_range,
                                                                    attend_indices, v_indices, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void sparse_local_attention_with_hash_kernel(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int attend_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx) {
    /*
        in sparse attention, voxels are not necessary at the non-empty location
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, hash_size, 2] voxel coordinates to voxel indices
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];

    xyz_to_vidx += bs_idx * hash_size * 2;

    int num_samples = 0;
    for (int sz_idx = z_idx * z_stride - attend_range; sz_idx <= z_idx * z_stride + (z_stride - 1) + attend_range; ++sz_idx){
        if (sz_idx >= z_max || sz_idx < 0) continue;
        for (int sy_idx = y_idx * y_stride - attend_range; sy_idx <= y_idx * y_stride + (y_stride - 1) + attend_range; ++sy_idx){
            if (sy_idx >= y_max || sy_idx < 0) continue;
            for (int sx_idx = x_idx * x_stride - attend_range; sx_idx <= x_idx * x_stride + (x_stride - 1) + attend_range; ++sx_idx){
                if (sx_idx >= x_max || sx_idx < 0) continue;
                int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
                int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
                if (sv_idx != EMPTY_KEY) { // found non-empty index
                    if (num_samples >= attend_size) return; // full and return
                    attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                    num_samples++;
                }else { // not found
                    ;
                }
            }
        }
    }
    return;
}

void sparse_local_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                        int num_voxels, int attend_size, int attend_range, int hash_size,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx) {

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    sparse_local_attention_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                                    num_voxels, attend_size, attend_range, hash_size,
                                                                    attend_indices, v_indices, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void subm_local_attention_with_tensor_kernel(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx) {
    /*
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, x_max, y_max, z_max] voxel coordinates to voxel indices
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx += bs_idx * x_max * y_max * z_max;

    int num_samples = 0;
    for (int sz_idx = z_idx - attend_range; sz_idx <= z_idx + attend_range; ++sz_idx){
        if (sz_idx >= z_max || sz_idx < 0) continue;
        for (int sy_idx = y_idx - attend_range; sy_idx <= y_idx + attend_range; ++sy_idx){
            if (sy_idx >= y_max || sy_idx < 0) continue;
            for (int sx_idx = x_idx - attend_range; sx_idx <= x_idx + attend_range; ++sx_idx){
                if (sx_idx >= x_max || sx_idx < 0) continue;
                int sv_idx = xyz_to_vidx[sx_idx * y_max * z_max + sy_idx * z_max + sz_idx];
                if (sv_idx != EMPTY_KEY) { // found non-empty index
                    if (num_samples >= attend_size) return; // full and return
                    attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                    num_samples++;
                }else { // not found
                    ;
                }
            }
        }
    }
    return;
}

void subm_local_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    subm_local_attention_with_tensor_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels,
                                                                    attend_size, attend_range, attend_indices, v_indices, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void subm_local_attention_with_hash_kernel(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range, int hash_size,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx) {
    /*
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, x_max, y_max, z_max] voxel coordinates to voxel indices
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx += bs_idx * hash_size * 2;

    int num_samples = 0;
    for (int sz_idx = z_idx - attend_range; sz_idx <= z_idx + attend_range; ++sz_idx){
        if (sz_idx >= z_max || sz_idx < 0) continue;
        for (int sy_idx = y_idx - attend_range; sy_idx <= y_idx + attend_range; ++sy_idx){
            if (sy_idx >= y_max || sy_idx < 0) continue;
            for (int sx_idx = x_idx - attend_range; sx_idx <= x_idx + attend_range; ++sx_idx){
                if (sx_idx >= x_max || sx_idx < 0) continue;
                int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
                int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
                if (sv_idx != EMPTY_KEY) { // found non-empty index
                    if (num_samples >= attend_size) return; // full and return
                    attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                    num_samples++;
                }else { // not found
                    ;
                }
            }
        }
    }
    return;
}

void subm_local_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range, int hash_size,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    subm_local_attention_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, attend_size, attend_range, hash_size,
                                                                    attend_indices, v_indices, xyz_to_vidx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void sparse_strided_attention_with_tensor_kernel(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int num_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec) {
    /*
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, x_max, y_max, z_max] voxel coordinates to voxel indices
        range_spec: [num_range, 3] half start/end range & stride
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx += bs_idx * x_max * y_max * z_max;

    int num_samples = 0;
    for (int range_idx = 0; range_idx < num_range; ++range_idx) {
        int search_x_start_range = range_spec[range_idx * 9 + 0];
        int search_x_end_range = range_spec[range_idx * 9 + 1];
        int search_x_stride = range_spec[range_idx * 9 + 2];
        int search_y_start_range = range_spec[range_idx * 9 + 3];
        int search_y_end_range = range_spec[range_idx * 9 + 4];
        int search_y_stride = range_spec[range_idx * 9 + 5];
        int search_z_start_range = range_spec[range_idx * 9 + 6];
        int search_z_end_range = range_spec[range_idx * 9 + 7];
        int search_z_stride = range_spec[range_idx * 9 + 8];
        for (int z_offset = 0; z_offset < search_z_end_range; z_offset += search_z_stride) {
        for (int y_offset = 0; y_offset < search_y_end_range; y_offset += search_y_stride) {
        for (int x_offset = 0; x_offset < search_x_end_range; x_offset += search_x_stride) {
             if ((x_offset < search_x_start_range) && (y_offset < search_y_start_range)
             && (z_offset < search_z_start_range)) {
                continue;
             }
            // each loop process 8 points
            for (int sz_idx = z_idx * z_stride - z_offset; sz_idx <= z_idx * z_stride + (z_stride - 1) + z_offset; sz_idx += (2 * z_offset + z_stride - 1)){
                if (sz_idx >= z_max || sz_idx < 0) continue;
                for (int sy_idx = y_idx * y_stride - y_offset; sy_idx <= y_idx * y_stride + (y_stride - 1) + y_offset; sy_idx += (2 * y_offset + y_stride - 1)){
                    if (sy_idx >= y_max || sy_idx < 0) continue;
                    for (int sx_idx = x_idx * x_stride - x_offset; sx_idx <= x_idx * x_stride + (x_stride - 1) + x_offset; sx_idx += (2 * x_offset + x_stride - 1)){
                        if (sx_idx >= x_max || sx_idx < 0) continue;
                        int sv_idx = xyz_to_vidx[sx_idx * y_max * z_max + sy_idx * z_max + sz_idx];
                        if (sv_idx != EMPTY_KEY) { // found non-empty index
                            if (num_samples >= attend_size) return; // full and return
                            attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                            num_samples++;
                        }else { // not found
                            ;
                        }
                    }
                }
            }
        }
        }
        }

    }
    return;
}

void sparse_strided_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int num_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    sparse_strided_attention_with_tensor_kernel<<<blocks, threads>>>(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                                        num_voxels, attend_size, num_range,
                                                                        attend_indices, v_indices, xyz_to_vidx, range_spec);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void sparse_strided_attention_with_hash_kernel(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec) {
    /*
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, hash_size, 2] voxel coordinates to voxel indices
        range_spec: [num_range, 3] half start/end range & stride
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx += bs_idx * hash_size * 2;

    int num_samples = 0;
    for (int range_idx = 0; range_idx < num_range; ++range_idx) {
        int search_x_start_range = range_spec[range_idx * 9 + 0];
        int search_x_end_range = range_spec[range_idx * 9 + 1];
        int search_x_stride = range_spec[range_idx * 9 + 2];
        int search_y_start_range = range_spec[range_idx * 9 + 3];
        int search_y_end_range = range_spec[range_idx * 9 + 4];
        int search_y_stride = range_spec[range_idx * 9 + 5];
        int search_z_start_range = range_spec[range_idx * 9 + 6];
        int search_z_end_range = range_spec[range_idx * 9 + 7];
        int search_z_stride = range_spec[range_idx * 9 + 8];
        for (int z_offset = 0; z_offset < search_z_end_range; z_offset += search_z_stride) {
        for (int y_offset = 0; y_offset < search_y_end_range; y_offset += search_y_stride) {
        for (int x_offset = 0; x_offset < search_x_end_range; x_offset += search_x_stride) {
            if ((x_offset < search_x_start_range) && (y_offset < search_y_start_range)
             && (z_offset < search_z_start_range)) {
                continue;
             }
            // each loop process 8 points
            for (int sz_idx = z_idx * z_stride - z_offset; sz_idx <= z_idx * z_stride + (z_stride - 1) + z_offset; sz_idx += (2 * z_offset + z_stride - 1)){
                if (sz_idx >= z_max || sz_idx < 0) continue;
                for (int sy_idx = y_idx * y_stride - y_offset; sy_idx <= y_idx * y_stride + (y_stride - 1) + y_offset; sy_idx += (2 * y_offset + y_stride - 1)){
                    if (sy_idx >= y_max || sy_idx < 0) continue;
                    for (int sx_idx = x_idx * x_stride - x_offset; sx_idx <= x_idx * x_stride + (x_stride - 1) + x_offset; sx_idx += (2 * x_offset + x_stride - 1)){
                        if (sx_idx >= x_max || sx_idx < 0) continue;
                        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
                        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
                        if (sv_idx != EMPTY_KEY) { // found non-empty index
                            if (num_samples >= attend_size) return; // full and return
                            attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                            num_samples++;
                        }else { // not found
                            ;
                        }
                    }
                }
            }
        }
        }
        }

    }
    return;
}

void sparse_strided_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    sparse_strided_attention_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                                        num_voxels, attend_size, num_range, hash_size,
                                                                        attend_indices, v_indices, xyz_to_vidx, range_spec);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void subm_strided_attention_with_tensor_kernel(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec) {
    /*
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, x_max, y_max, z_max] voxel coordinates to voxel indices
        range_spec: [num_range, 3] half start/end range & stride
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx += bs_idx * x_max * y_max * z_max;

    int num_samples = 0;
    for (int range_idx = 0; range_idx < num_range; ++range_idx) {
        int search_x_start_range = range_spec[range_idx * 9 + 0];
        int search_x_end_range = range_spec[range_idx * 9 + 1];
        int search_x_stride = range_spec[range_idx * 9 + 2];
        int search_y_start_range = range_spec[range_idx * 9 + 3];
        int search_y_end_range = range_spec[range_idx * 9 + 4];
        int search_y_stride = range_spec[range_idx * 9 + 5];
        int search_z_start_range = range_spec[range_idx * 9 + 6];
        int search_z_end_range = range_spec[range_idx * 9 + 7];
        int search_z_stride = range_spec[range_idx * 9 + 8];
        int x_step = 0;
        int y_step = 0;
        int z_step = 0;
        for (int z_offset = 0; z_offset < search_z_end_range; z_offset += search_z_stride) {
        for (int y_offset = 0; y_offset < search_y_end_range; y_offset += search_y_stride) {
        for (int x_offset = 0; x_offset < search_x_end_range; x_offset += search_x_stride) {
            if ((x_offset < search_x_start_range) && (y_offset < search_y_start_range)
             && (z_offset < search_z_start_range)) {
                continue;
             }
            // each loop process 8 points
            if (z_offset == 0) {
                z_step = 1;
            } else {
                z_step = 2 * z_offset;
            }
            for (int sz_idx = z_idx - z_offset; sz_idx <= z_idx + z_offset; sz_idx += z_step){
                if (sz_idx >= z_max || sz_idx < 0) continue;
                if (sz_idx >= z_max || sz_idx < 0) continue;
                if (y_offset == 0) {
                    y_step = 1;
                } else {
                    y_step = 2 * y_offset;
                }
                for (int sy_idx = y_idx - y_offset; sy_idx <= y_idx + y_offset; sy_idx += y_step){
                    if (sy_idx >= y_max || sy_idx < 0) continue;
                    if (x_offset == 0) {
                        x_step = 1;
                    } else {
                        x_step = 2 * x_offset;
                    }
                    for (int sx_idx = x_idx - x_offset; sx_idx <= x_idx + x_offset; sx_idx += x_step){
                        if (sx_idx >= x_max || sx_idx < 0) continue;
                        int sv_idx = xyz_to_vidx[sx_idx * y_max * z_max + sy_idx * z_max + sz_idx];
                        if (sv_idx != EMPTY_KEY) { // found non-empty index
                            if (num_samples >= attend_size) return; // full and return
                            attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                            num_samples++;
                        }else { // not found
                            ;
                        }
                    }
                }
            }
        }
        }
        }

    }
    return;
}

void subm_strided_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    subm_strided_attention_with_tensor_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, attend_size, num_range,
                                                                    attend_indices, v_indices, xyz_to_vidx, range_spec);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void subm_strided_attention_with_hash_kernel(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec) {
    /*
        attend_indices: [num_voxels, attend_size] for gather attend indices
        v_indices: [num_voxels, 4] bs + zyx indices of voxels
        xyz_to_vidx: [bs, hash_size, 2] voxel coordinates to voxel indices
        range_spec: [num_range, 3] half start/end range & stride
    */

    int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx >= num_voxels) return;

    int bs_idx = v_indices[th_idx * 4 + 0];
    int z_idx = v_indices[th_idx * 4 + 1];
    int y_idx = v_indices[th_idx * 4 + 2];
    int x_idx = v_indices[th_idx * 4 + 3];
    if (x_idx >= x_max || x_idx < 0 || y_idx < 0 || y_idx >= y_max || z_idx < 0 || z_idx >= z_max) return;
    xyz_to_vidx += bs_idx * hash_size * 2;

    int num_samples = 0;
    for (int range_idx = 0; range_idx < num_range; ++range_idx) {
        int search_x_start_range = range_spec[range_idx * 9 + 0];
        int search_x_end_range = range_spec[range_idx * 9 + 1];
        int search_x_stride = range_spec[range_idx * 9 + 2];
        int search_y_start_range = range_spec[range_idx * 9 + 3];
        int search_y_end_range = range_spec[range_idx * 9 + 4];
        int search_y_stride = range_spec[range_idx * 9 + 5];
        int search_z_start_range = range_spec[range_idx * 9 + 6];
        int search_z_end_range = range_spec[range_idx * 9 + 7];
        int search_z_stride = range_spec[range_idx * 9 + 8];
        int x_step = 0;
        int y_step = 0;
        int z_step = 0;
        for (int z_offset = 0; z_offset < search_z_end_range; z_offset += search_z_stride) {
        for (int y_offset = 0; y_offset < search_y_end_range; y_offset += search_y_stride) {
        for (int x_offset = 0; x_offset < search_x_end_range; x_offset += search_x_stride) {
            if ((x_offset < search_x_start_range) && (y_offset < search_y_start_range)
             && (z_offset < search_z_start_range)) {
                continue;
             }
            // each loop process 8 points
            if (z_offset == 0) {
                z_step = 1;
            } else {
                z_step = 2 * z_offset;
            }
            for (int sz_idx = z_idx - z_offset; sz_idx <= z_idx + z_offset; sz_idx += z_step){
                if (sz_idx >= z_max || sz_idx < 0) continue;
                if (y_offset == 0) {
                    y_step = 1;
                } else {
                    y_step = 2 * y_offset;
                }
                for (int sy_idx = y_idx - y_offset; sy_idx <= y_idx + y_offset; sy_idx += y_step){
                    if (sy_idx >= y_max || sy_idx < 0) continue;
                    if (x_offset == 0) {
                        x_step = 1;
                    } else {
                        x_step = 2 * x_offset;
                    }
                    for (int sx_idx = x_idx - x_offset; sx_idx <= x_idx + x_offset; sx_idx += x_step){
                        if (sx_idx >= x_max || sx_idx < 0) continue;
                        int skey = sx_idx * y_max * z_max + sy_idx * z_max + sz_idx;
                        int sv_idx = hash_table_find(skey, hash_size, xyz_to_vidx);
                        if (sv_idx != EMPTY_KEY) { // found non-empty index
                            if (num_samples >= attend_size) return; // full and return
                            attend_indices[th_idx * attend_size + num_samples] = sv_idx;
                            num_samples++;
                        }else { // not found
                            ;
                        }
                    }
                }
            }
        }
        }
        }

    }
    return;
}

void subm_strided_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec){

    cudaError_t err;

    dim3 blocks(DIVUP(num_voxels, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    subm_strided_attention_with_hash_kernel<<<blocks, threads>>>(x_max, y_max, z_max, num_voxels, attend_size, num_range, hash_size,
                                                                    attend_indices, v_indices, xyz_to_vidx, range_spec);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}