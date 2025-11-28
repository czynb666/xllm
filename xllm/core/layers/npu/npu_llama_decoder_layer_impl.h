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
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>
#include <memory>

#include "atb/atb_infer.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_kernels/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_kernels/core/include/atb_speed/base/model.h"
#include "xllm_kernels/core/include/atb_speed/log.h"
#include "xllm_kernels/core/include/atb_speed/utils/model_factory.h"
#include "xllm_kernels/models/llama/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class LlamaLoader;

class NpuLlamaDecoderLayerImpl : public NpuBaseLayer {
 public:
  explicit NpuLlamaDecoderLayerImpl(const ModelContext& context);

  ~NpuLlamaDecoderLayerImpl() {};

  virtual void load_state_dict(const StateDict& state_dict) override;

  virtual void verify_loaded_weights() const override;

  virtual void merge_loaded_weights() override;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        int node_id = 0);

 private:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::llama::LlamaLayerParam& param);

  int64_t init_attn_mask();

  void param_from_args(atb_speed::llama::LlamaLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  std::string model_name_;
  atb_speed::llama::LlamaLayerParam prefill_param_;
  atb_speed::llama::LlamaLayerParam decode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;

  // at::Tensor encode_attn_mask_;
  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;

  int device_id_;

  enum TensorId : int {
    IN_NORM_WEIGHT = 0,
    IN_NORM_BIAS,
    IN_NORM_NEW_WEIGHT,
    IN_NORM_NEW_BIAS,
    IN_Q_WEIGHT,
    IN_Q_BIAS,
    IN_Q_DEQSCALE,
    IN_Q_OFFSET,
    IN_Q_SCALE,
    IN_Q_COMPRESS_IDX,
    IN_K_WEIGHT,
    IN_K_BIAS,
    IN_K_DEQSCALE,
    IN_K_OFFSET,
    IN_K_SCALE,
    IN_K_COMPRESS_IDX,
    IN_V_WEIGHT,
    IN_V_BIAS,
    IN_V_DEQSCALE,
    IN_V_OFFSET,
    IN_V_SCALE,
    IN_V_COMPRESS_IDX,
    IN_ATTENTION_OUT_WEIGHT,
    IN_ATTENTION_OUT_BIAS,
    IN_ATTENTION_OUT_DEQSCALE,
    IN_ATTENTION_OUT_OFFSET,
    IN_ATTENTION_OUT_SCALE,
    IN_ATTENTION_OUT_COMPRESS_IDX,
    IN_SELFOUT_NORM_WEIGHT,
    IN_SELFOUT_NORM_BIAS,
    IN_SELFOUT_NORM_NEW_WEIGHT,
    IN_SELFOUT_NORM_NEW_BIAS,
    IN_MLP_W2_WEIGHT,
    IN_MLP_W2_BIAS,
    IN_MLP_W2_DEQSCALE,
    IN_MLP_W2_OFFSET,
    IN_MLP_W2_SCALE,
    IN_MLP_W2_COMPRESS_IDX,
    IN_MLP_W1_WEIGHT,
    IN_MLP_W1_BIAS,
    IN_MLP_W1_DEQSCALE,
    IN_MLP_W1_OFFSET,
    IN_MLP_W1_SCALE,
    IN_MLP_W1_COMPRESS_IDX,
    IN_MLP_CPROJ_WEIGHT,
    IN_MLP_CPROJ_BIAS,
    IN_MLP_CPROJ_DEQSCALE,
    IN_MLP_CPROJ_OFFSET,
    IN_MLP_CPROJ_SCALE,
    IN_MLP_CPROJ_COMPRESS_IDX
  };

  friend class LlamaLoader;
  std::unique_ptr<LlamaLoader> loader_;
};

}  // namespace layer
}  // namespace xllm
