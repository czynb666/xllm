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

#include "core/layers/base_loader.h"
namespace xllm {
namespace layer {

class Qwen2ot5VisionEncoderLoader : public BaseLoader {
 public:
  explicit Qwen2ot5VisionEncoderLoader(uint64_t weight_count,
                                       const xllm::ParallelArgs& parallel_args);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;

 private:
  torch::Tensor pad_tensor(const torch::Tensor& tensor,
                           int64_t target_shape,
                           int64_t dim = 0) {
    int64_t pad_size = target_shape - tensor.size(dim);
    if (tensor.dim() == 1) {
      return torch::nn::functional::pad(
          tensor, torch::nn::functional::PadFuncOptions({0, pad_size}));
    } else if (tensor.dim() == 2) {
      if (1 == dim)
        return torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({0, pad_size, 0, 0}));
      else
        return torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    }
    return tensor;
  }
  void get_weights_col_packed_qkv();
  void pad_qkv_weights();
  void pad_mlp_weights();
  TransposeType check_transpose(at::Tensor& tensor);
};

}  // namespace layer
}  // namespace xllm