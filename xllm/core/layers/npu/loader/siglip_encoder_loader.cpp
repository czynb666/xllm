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

SiglipEncoderLoader::SiglipEncoderLoader(
    uint64_t weight_count,
    const xllm::ParallelArgs& parallel_args)
    : BaseLoader(weight_count, parallel_args) {}

void SiglipEncoderLoader::load_state_dict(const StateDict& state_dict) {
  const std::set<std::string> key_names = {"self_attn.out_proj.weight",
                                           "self_attn.out_proj.bias",
                                           "layer_norm2.weight",
                                           "layer_norm2.bias",
                                           "mlp.fc1.weight",
                                           "mlp.fc1.bias",
                                           "mlp.fc2.weight",
                                           "mlp.fc2.bias"};

  atb_torch::TorchTensorMap weights_map;
  for (const auto& [name, tensor] : state_dict) {
    if (key_names.find(name) == key_names.end()) continue;

    auto weight_npu = tensor.to(options_);

    weights_.push_back(weight_npu);
    weights_map[name] = weight_npu;
  }
  graph_.SetWeights(weights_map);
}

}  // namespace layer
}  // namespace xllm