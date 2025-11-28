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

#include <map>
#include <unordered_map>
#include <vector>

#include "core/layers/base_loader.h"

namespace xllm {
namespace layer {

class NpuQwen3MoeDecoderLayerImp;

class Qwen3MoeDecoderLoader : public BaseLoader {
 public:
  explicit Qwen3MoeDecoderLoader(uint64_t weight_count,
                                 const xllm::ParallelArgs& parallel_args);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void merge_loaded_weights() override;
};

private:
int get_mapped_index(const std::string& name,
                     const std::unordered_map<std::string, int>& mapping);

void process_expert_weights(const StateDict& state_dict,
                            const std::string& name,
                            const torch::Tensor& tensor);

void process_mlp_common_weights(const StateDict& state_dict,
                                const std::string& name,
                                const torch::Tensor& tensor);

void process_general_weights(const StateDict& state_dict,
                             const std::string& name,
                             const torch::Tensor& tensor);

torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                 const std::string& name,
                                 int dim,
                                 int loacal_tp_rank,
                                 int local_tp_size);

torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                 const std::string& name,
                                 int dim);

torch::Tensor convert_fp16_to_int64(const torch::Tensor& fp16_tensor);

int extract_expert_index(const std::string& name);

std::string extract_endswith(const std::string& input);

torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                    bool transpose = false);

torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_up,
                                    std::vector<torch::Tensor>& experts_gate,
                                    bool transpose = false);

void resize_experts_weights(int num_of_device_experts);

void initialize_tensors(const torch::TensorOptions& options);

void initialize_weight_tensors(const torch::TensorOptions& options);

int32_t ep_size_;
int32_t num_experts_;
int32_t num_experts_per_partition_;
int32_t ep_local_tp_size_;
int32_t ep_local_tp_rank_;
int32_t start_expert_id_;
int32_t end_expert_id_;
int32_t ep_rank_;

int32_t dp_size_;
int32_t dp_local_tp_size_;
int32_t dp_rank_;
int32_t dp_local_tp_rank_;

atb::Tensor internal_tensor_;

torch::Tensor tensor_placeholder_;
torch::Tensor slot_tensor_placeholder_;
torch::Tensor int_tensor_placeholder_;
torch::Tensor expert_group_;
torch::Tensor one_hot_;
torch::Tensor zero_hot_;
torch::Tensor at_start_expert_id_;
torch::Tensor at_in_device_expert_count_;

std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;

friend class NpuQwen3MoeDecoderLayerImp;
}  // namespace layer
}  // namespace xllm