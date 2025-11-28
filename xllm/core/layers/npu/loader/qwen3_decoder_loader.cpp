/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include "core/layers/npu/npu_qwen3_decoder_owner_impl.h"
#include "loader/qwen3_loader.h"

namespace xllm {
namespace layer {

namespace {

const std::vector<std::pair<int, std::string>> kWeightMapping = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

const std::vector<std::pair<int, std::string>> kWeightMappingW8A8 = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.quant_bias"},
    {IN_Q_DEQSCALE, "self_attn.q_proj.deq_scale"},
    {IN_Q_OFFSET, "self_attn.q_proj.input_offset"},
    {IN_Q_SCALE, "self_attn.q_proj.input_scale"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.quant_bias"},
    {IN_K_DEQSCALE, "self_attn.k_proj.deq_scale"},
    {IN_K_OFFSET, "self_attn.k_proj.input_offset"},
    {IN_K_SCALE, "self_attn.k_proj.input_scale"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.quant_bias"},
    {IN_V_DEQSCALE, "self_attn.v_proj.deq_scale"},
    {IN_V_OFFSET, "self_attn.v_proj.input_offset"},
    {IN_V_SCALE, "self_attn.v_proj.input_scale"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_ATTENTION_OUT_BIAS, "self_attn.o_proj.quant_bias"},
    {IN_ATTENTION_OUT_DEQSCALE, "self_attn.o_proj.deq_scale"},
    {IN_ATTENTION_OUT_OFFSET, "self_attn.o_proj.input_offset"},
    {IN_ATTENTION_OUT_SCALE, "self_attn.o_proj.input_scale"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W2_BIAS, "mlp.gate_proj.quant_bias"},
    {IN_MLP_W2_DEQSCALE, "mlp.gate_proj.deq_scale"},
    {IN_MLP_W2_OFFSET, "mlp.gate_proj.input_offset"},
    {IN_MLP_W2_SCALE, "mlp.gate_proj.input_scale"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_W1_BIAS, "mlp.up_proj.quant_bias"},
    {IN_MLP_W1_DEQSCALE, "mlp.up_proj.deq_scale"},
    {IN_MLP_W1_OFFSET, "mlp.up_proj.input_offset"},
    {IN_MLP_W1_SCALE, "mlp.up_proj.input_scale"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

const std::map<int, int> kWeightShard = {{IN_Q_WEIGHT, 0},
                                         {IN_K_WEIGHT, 0},
                                         {IN_V_WEIGHT, 0},
                                         {IN_ATTENTION_OUT_WEIGHT, 1},
                                         {IN_MLP_W2_WEIGHT, 0},
                                         {IN_MLP_W1_WEIGHT, 0},
                                         {IN_MLP_CPROJ_WEIGHT, 1}};

const std::map<int, int> kWeightShardW8A8 = {{IN_Q_WEIGHT, 0},
                                             {IN_Q_BIAS, 0},
                                             {IN_Q_DEQSCALE, 0},
                                             {IN_K_WEIGHT, 0},
                                             {IN_K_BIAS, 0},
                                             {IN_K_DEQSCALE, 0},
                                             {IN_V_WEIGHT, 0},
                                             {IN_V_BIAS, 0},
                                             {IN_V_DEQSCALE, 0},
                                             {IN_ATTENTION_OUT_WEIGHT, 1},
                                             {IN_MLP_W2_WEIGHT, 0},
                                             {IN_MLP_W2_BIAS, 0},
                                             {IN_MLP_W2_DEQSCALE, 0},
                                             {IN_MLP_W1_WEIGHT, 0},
                                             {IN_MLP_W1_BIAS, 0},
                                             {IN_MLP_W1_DEQSCALE, 0},
                                             {IN_MLP_CPROJ_WEIGHT, 1}};

}  // namespace

Qwen3DecoderLoader::Qwen3DecoderLoader()

    void Qwen3DecoderLoader::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_.compare("w8a8") == 0) {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict, name, index, WEIGHT_SHARD_W8A8[index]);
      } else {
        set_weight(state_dict, name, index);
      }
    }
    at_weight_tensors_[IN_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_NORM_WEIGHT].options())
            .to(device_);

    at_weight_tensors_[IN_SELFOUT_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].options())
            .to(device_);
    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Qwen3DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : kWeightMapping) {
    CHECK(owner_.at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3DecoderLoader::merge_loaded_weights() {}

}  // namespace layer
}  // namespace xllm
