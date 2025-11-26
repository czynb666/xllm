/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "loader/base_loader.h"

namespace xllm {
namespace layer {

BaseLoader::BaseLoader(uint64_t weight_count,
                       const xllm::ParallelArgs& parallel_args,
                       at::Device device)
    : weight_count_(weight_count),
      parallel_args_(parallel_args),
      device_(device) {
  at_weight_tensors_.resize(weight_count_);
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      correct_tensor_dtype(mutable_tensor, tensor_name);
      at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
    }
  }
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            int dim) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      if (parallel_args_.world_size() <= 1) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      } else {
        at_weight_tensors_[weight_position] =
            state_dict
                .get_sharded_tensor(tensor_name,
                                    /*dim=*/dim,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size())
                .to(device_);
        correct_tensor_dtype(at_weight_tensors_[weight_position], tensor_name);
      }
    }
  }
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            int dim,
                            int rank,
                            int world_size) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      if (world_size <= 1) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      } else {
        at_weight_tensors_[weight_position] =
            state_dict
                .get_sharded_tensor(tensor_name,
                                    /*dim=*/dim,
                                    /*rank=*/rank,
                                    /*world_size=*/world_size)
                .to(device_);
        correct_tensor_dtype(at_weight_tensors_[weight_position], tensor_name);
      }
    }
  }
}

void BaseLoader::correct_tensor_dtype(torch::Tensor& tensor,
                                      const std::string& tensorName) {
  if (absl::EndsWith(tensorName, "deq_scale") &&
      (torch_dtype_.compare("bfloat16") == 0)) {
    return;
  }

  if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
      tensor.dtype() != torch::kInt64) {
    torch::Dtype dtype = string2dtype(torch_dtype_);
    tensor = tensor.to(dtype);
  }
}

torch::Dtype BaseLoader::string2dtype(const std::string& dtype_str) {
  if (dtype_str.compare("float16") == 0) {
    return torch::kFloat16;
  } else if (dtype_str.compare("bfloat16") == 0) {
    return torch::kBFloat16;
  } else if (dtype_str.compare("float32") == 0) {
    return torch::kFloat32;
  } else if (dtype_str.compare("float64") == 0) {
    return torch::kFloat64;
  } else if (dtype_str.compare("int8") == 0) {
    return torch::kInt8;
  } else if (dtype_str.compare("int16") == 0) {
    return torch::kInt16;
  } else if (dtype_str.compare("int32") == 0) {
    return torch::kInt32;
  } else if (dtype_str.compare("int64") == 0) {
    return torch::kInt64;
  } else if (dtype_str.compare("uint8") == 0) {
    return torch::kUInt8;
  } else if (dtype_str.compare("bool") == 0) {
    return torch::kBool;
  }

  LOG(FATAL) << "Unsupported dtype string: " << dtype_str;
}

}  // namespace layer
}  // namespace xllm