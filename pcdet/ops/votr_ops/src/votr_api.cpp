#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "build_mapping_gpu.h"
#include "build_attention_indices_gpu.h"
#include "group_features_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_mapping_with_tensor_wrapper", &build_mapping_with_tensor_wrapper, "build_mapping_with_tensor_wrapper");
    m.def("build_mapping_with_hash_wrapper", &build_mapping_with_hash_wrapper, "build_mapping_with_hash_wrapper");
    m.def("downsample_with_tensor_wrapper", &downsample_with_tensor_wrapper, "downsample_with_tensor_wrapper");
    m.def("downsample_with_hash_wrapper", &downsample_with_hash_wrapper, "downsample_with_hash_wrapper");
    m.def("subm_local_attention_with_tensor_wrapper", &subm_local_attention_with_tensor_wrapper, "subm_local_attention_with_tensor_wrapper");
    m.def("subm_local_attention_with_hash_wrapper", &subm_local_attention_with_hash_wrapper, "subm_local_attention_with_hash_wrapper");
    m.def("sparse_local_attention_with_tensor_wrapper", &sparse_local_attention_with_tensor_wrapper, "sparse_local_attention_with_tensor_wrapper");
    m.def("sparse_local_attention_with_hash_wrapper", &sparse_local_attention_with_hash_wrapper, "sparse_local_attention_with_hash_wrapper");
    m.def("subm_strided_attention_with_tensor_wrapper", &subm_strided_attention_with_tensor_wrapper, "subm_strided_attention_with_tensor_wrapper");
    m.def("subm_strided_attention_with_hash_wrapper", &subm_strided_attention_with_hash_wrapper, "subm_strided_attention_with_hash_wrapper");
    m.def("sparse_strided_attention_with_tensor_wrapper", &sparse_strided_attention_with_tensor_wrapper, "sparse_strided_attention_with_tensor_wrapper");
    m.def("sparse_strided_attention_with_hash_wrapper", &sparse_strided_attention_with_hash_wrapper, "sparse_strided_attention_with_hash_wrapper");
    m.def("group_features_grad_wrapper", &group_features_grad_wrapper_stack, "group_features_grad_wrapper_stack");
    m.def("group_features_wrapper", &group_features_wrapper_stack, "group_features_wrapper_stack");
}
