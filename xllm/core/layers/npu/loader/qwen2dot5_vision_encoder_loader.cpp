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

#include "qwen2dot5_vision_encoder_loader.h"
namespace xllm {
namespace layer {
static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "qkv.weight"},
    {IN_QKV_BIAS, "qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_MLP_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_GATE_BIAS, "mlp.gate_proj.bias"},
    {IN_MLP_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_UP_BIAS, "mlp.up_proj.bias"},
    {IN_MLP_DOWN_WEIGHT, "mlp.down_proj.weight"},
    {IN_MLP_DOWN_BIAS, "mlp.down_proj.bias"},
};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATE_WEIGHT, 0},
    {IN_MLP_GATE_BIAS, 0},
    {IN_MLP_UP_WEIGHT, 0},
    {IN_MLP_UP_BIAS, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

Qwen2ot5VisionEncoderLoader::Qwen2ot5VisionEncoderLoader() {
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen2ot5VisionEncoderLoader::
    merge_loaded_weights() {  // spilt pack qkv weight when enable tp
  get_weights_col_packed_qkv();
  if (encode_param_.worldSize > 1) {
    // merge qkv weight
    auto new_qkv_weight = torch::cat({at_weight_tensors_[IN_VISION_Q_WEIGHT],
                                      at_weight_tensors_[IN_VISION_K_WEIGHT],
                                      at_weight_tensors_[IN_VISION_V_WEIGHT]},
                                     0);
    at_weight_tensors_[IN_QKV_WEIGHT] = new_qkv_weight;
    at_weight_tensors_[IN_VISION_Q_WEIGHT] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_K_WEIGHT] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_V_WEIGHT] = torch::zeros({1}).to(device_);

    // merge qkv bias
    auto new_qkv_bias = torch::cat({at_weight_tensors_[IN_VISION_Q_BIAS],
                                    at_weight_tensors_[IN_VISION_K_BIAS],
                                    at_weight_tensors_[IN_VISION_V_BIAS]},
                                   0);
    at_weight_tensors_[IN_QKV_BIAS] = new_qkv_bias;
    at_weight_tensors_[IN_VISION_Q_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_K_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_V_BIAS] = torch::zeros({1}).to(device_);
  }
  // pad qkv weights
  pad_qkv_weights();
  // merge gate up
  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_GATE_WEIGHT],
                                    at_weight_tensors_[IN_MLP_UP_WEIGHT]},
                                   0);
  at_weight_tensors_[IN_MLP_GATE_WEIGHT] = new_mlp_weight;
  auto new_mlp_bias = torch::cat({at_weight_tensors_[IN_MLP_GATE_BIAS],
                                  at_weight_tensors_[IN_MLP_UP_BIAS]},
                                 0);
  at_weight_tensors_[IN_MLP_GATE_BIAS] = new_mlp_bias;
  at_weight_tensors_[IN_MLP_UP_BIAS] = torch::zeros({1}).to(device_);
  // pad mlp weights
  pad_mlp_weights();
}

void Qwen2ot5VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen2ot5VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
  get_weights_col_packed_qkv();
}
}  // namespace layer
}  // namespace xllm