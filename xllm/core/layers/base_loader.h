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

#pragma once

#include <torch/torch.h>

namespace xllm {
namespace layer {

class BaseLoader {
 public:
  explicit BaseLoader(uint64_t weight_count,
                      const xllm::ParallelArgs& parallel_args);
  virtual ~BaseLoader() = default;

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights() const = 0;
  virtual void merge_loaded_weights() = 0;

  torch::Dtype string2dtype(const std::string& dtype_str);

  void correct_tensor_dtype(torch::Tensor& tensor,
                            const std::string& tensorName);

 protected:
  uint64_t weight_count_;
  xllm::ParallelArgs parallel_args_;
  std::vector<at::Tensor> at_weight_tensors_;
  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim,
                  int rank,
                  int world_size);
};

}  // namespace layer
}  // namespace xllm