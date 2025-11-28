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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include "core/layers/npu/npu_llama_decoder_owner_impl.h"
#include "loader/llama_loader.h"

namespace xllm {
namespace layer {

namespace {

using TensorId = NpuLlamaDecoderLayerImpl::TensorId;

const std::unordered_map<std::string, TensorId> kWeightMapping = {
    {"input_layernorm.weight", TensorId::IN_NORM_WEIGHT},
    {"self_attn.q_proj.weight", TensorId::IN_Q_WEIGHT},
    {"self_attn.k_proj.weight", TensorId::IN_K_WEIGHT},
    {"self_attn.v_proj.weight", TensorId::IN_V_WEIGHT},
    {"self_attn.o_proj.weight", TensorId::IN_ATTENTION_OUT_WEIGHT},
    {"post_attention_layernorm.weight", TensorId::IN_SELFOUT_NORM_WEIGHT},
    {"mlp.gate_proj.weight", TensorId::IN_MLP_W2_WEIGHT},
    {"mlp.up_proj.weight", TensorId::IN_MLP_W1_WEIGHT},
    {"mlp.down_proj.weight", TensorId::IN_MLP_CPROJ_WEIGHT},
};

const std::map<TensorId, int> kWeightShard = {
    {TensorId::IN_Q_WEIGHT, 0},
    {TensorId::IN_K_WEIGHT, 0},
    {TensorId::IN_V_WEIGHT, 0},
    {TensorId::IN_ATTENTION_OUT_WEIGHT, 1},
    {TensorId::IN_MLP_W2_WEIGHT, 0},
    {TensorId::IN_MLP_W1_WEIGHT, 0},
    {TensorId::IN_MLP_CPROJ_WEIGHT, 1}};

inline int Index(TensorId id) { return static_cast<int>(id); }

}  // namespace

LlamaLoader::LlamaLoader(NpuLlamaDecoderLayerImpl& owner)
    : BaseLoader(layer), owner_(layer) {}

void LlamaLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, id] : kWeightMapping) {
    const auto shard = kWeightShard.find(id);
    if (shard != kWeightShard.end()) {
      set_weight(state_dict, name, Index(id), shard->second);
    } else {
      set_weight(state_dict, name, Index(id));
    }
  }
}

void LlamaLoader::verify_loaded_weights() const {
  for (const auto& [name, id] : kWeightMapping) {
    CHECK(owner_.at_weight_tensors_[Index(id)].sizes() !=
          std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void LlamaLoader::merge_loaded_weights() {
  auto new_q_weight =
      torch::cat({owner_.at_weight_tensors_[Index(TensorId::IN_Q_WEIGHT)],
                  owner_.at_weight_tensors_[Index(TensorId::IN_K_WEIGHT)],
                  owner_.at_weight_tensors_[Index(TensorId::IN_V_WEIGHT)]},
                 0);
  owner_.at_weight_tensors_[Index(TensorId::IN_Q_WEIGHT)] = new_q_weight;

  owner_.at_weight_tensors_[Index(TensorId::IN_K_WEIGHT)] =
      torch::zeros({1}).to(owner_.device_);
  owner_.at_weight_tensors_[Index(TensorId::IN_V_WEIGHT)] =
      torch::zeros({1}).to(owner_.device_);

  auto new_mlp_weight =
      torch::cat({owner_.at_weight_tensors_[Index(TensorId::IN_MLP_W2_WEIGHT)],
                  owner_.at_weight_tensors_[Index(TensorId::IN_MLP_W1_WEIGHT)]},
                 0);
  owner_.at_weight_tensors_[Index(TensorId::IN_MLP_W2_WEIGHT)] = new_mlp_weight;

  owner_.at_weight_tensors_[Index(TensorId::IN_MLP_W1_WEIGHT)] =
      torch::zeros({1}).to(owner_.device_);

  c10_npu::NPUCachingAllocator::emptyCache();
  for (size_t i = 0; i < owner_.at_weight_tensors_.size(); ++i) {
    owner_.atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(owner_.at_weight_tensors_[i]);
  }

  owner_.init_layer();
}

}  // namespace layer
}  // namespace xllm
