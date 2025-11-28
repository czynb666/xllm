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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model/npu_dp_ep_padding.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "npu_base_layer.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/models/qwen3/layer/moe_decoder_layer.h"

namespace xllm {
namespace layer {

class NpuQwen3MoeDecoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuQwen3MoeDecoderLayerImpl(const ModelContext& context,
                                       const int32_t layer_id);

  ~NpuQwen3MoeDecoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict);

  virtual void verify_loaded_weights(const std::string& prefix) const;

  virtual void merge_loaded_weights();

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor& expert_array,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

 private:
  struct ShardingConfig {
    bool is_sharded;
    int index;
    bool use_dp_sharding = false;
  };

  void param_from_args(atb_speed::qwen::MoeDecoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool is_prefill);

  void initialize_basic_parameters(atb_speed::qwen::MoeDecoderLayerParam& param,
                                   const ModelArgs& args,
                                   const ParallelArgs& parallel_args,
                                   bool is_prefill);

  void initialize_attention_parameters(
      atb_speed::qwen::MoeDecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_mlp_parameters(atb_speed::qwen::MoeDecoderLayerParam& param,
                                 const ModelArgs& args,
                                 const ParallelArgs& parallel_args);

  void initialize_parallel_parameters(
      atb_speed::qwen::MoeDecoderLayerParam& param,
      const ParallelArgs& parallel_args);

  void initialize_quantization_parameters(
      atb_speed::qwen::MoeDecoderLayerParam& param);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::MoeDecoderLayerParam& param);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               const ModelInputParams& input_params,
                               torch::Tensor& expert_array,
                               bool is_prefill);

  torch::Tensor block_tables_placeholder_;
  std::string model_name_;

  int32_t device_id_;
  int32_t layer_id_;

  int32_t num_speculative_tokens_ = 0;
  atb_speed::qwen::MoeDecoderLayerParam prefill_param_;
  atb_speed::qwen::MoeDecoderLayerParam decode_param_;

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
};
}  // namespace layer
}  // namespace xllm
