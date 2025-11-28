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
#include "qwen3_moe_decoder_loader.h"

namespace xllm {
namespace layer {
Qwen3MoeDecoderLoader::Qwen3MoeDecoderLoader(
    uint64_t weight_count,
    const xllm::ParallelArgs& parallel_args, )
    : BaseLoader(weight_count, parallel_args) {
  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;
  initialize_tensors(options);
}

void Qwen3MoeDecoderLoader void Qwen3MoeDecoderLoader::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    bool is_sharded = false;
    int index = 0;

    if (absl::StartsWith(name, "mlp.experts")) {
      process_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp") && !absl::StrContains(name, "gate.")) {
      process_mlp_common_weights(state_dict, name, tensor);
      continue;
    }

    process_general_weights(state_dict, name, tensor);
  }
}

void Qwen3MoeDecoderLoader::verify_loaded_weights() {
  for (const auto& [name, index] : QWEN3_MOE_WEIGHT_MAPPING) {
    if (name == "down_proj.weight" || name == "gate_proj.weight" ||
        name == "up_proj.weight") {
      continue;
    }
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3MoeDecoderLoader::merge_loaded_weights() {
  merge_experts_weights();
  at_weight_tensors_[IN_QKV_WEIGHT_0] =
      torch::cat({at_weight_tensors_[IN_QKV_WEIGHT_0],
                  at_weight_tensors_[IN_QKV_WEIGHT_1],
                  at_weight_tensors_[IN_QKV_WEIGHT_2]},
                 0)
          .contiguous();
  at_weight_tensors_[IN_QKV_WEIGHT_1] =
      torch::zeros({1}, torch::kFloat16).to(device_);
  at_weight_tensors_[IN_QKV_WEIGHT_2] =
      torch::zeros({1}, torch::kFloat16).to(device_);

  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    at_weight_tensors_[IN_QKV_BIAS_0] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_BIAS_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_BIAS_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_BIAS] =
        torch::zeros({1}, torch::kFloat16).to(device_);

    at_weight_tensors_[IN_QKV_DESCALE_0] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DESCALE_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_DESCALE_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_DESCALE] =
        torch::zeros({1}, torch::kFloat16).to(device_);

    at_weight_tensors_[IN_QKV_OFFSET_0] =
        torch::cat({at_weight_tensors_[IN_QKV_OFFSET_0],
                    at_weight_tensors_[IN_QKV_OFFSET_1],
                    at_weight_tensors_[IN_QKV_OFFSET_2]},
                   0)
            .contiguous()
            .view(-1);
    at_weight_tensors_[IN_QKV_OFFSET_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_OFFSET_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_weight_tensors_[IN_ATTENTION_OUT_OFFSET].contiguous().view(-1);

    at_weight_tensors_[IN_QKV_SCALE_0] =
        torch::cat({at_weight_tensors_[IN_QKV_SCALE_0],
                    at_weight_tensors_[IN_QKV_SCALE_1],
                    at_weight_tensors_[IN_QKV_SCALE_2]},
                   0)
            .contiguous()
            .view(-1);
    at_weight_tensors_[IN_QKV_SCALE_1] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_QKV_SCALE_2] =
        torch::zeros({1}, torch::kFloat16).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_SCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_SCALE].contiguous().view(-1);
  }
}

int Qwen3MoeDecoderLoader::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(ERROR) << "Missing mapping for: " << name;
    return -1;
  }

  return it->second;
}

void Qwen3MoeDecoderLoader::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  if (expert_index < start_expert_id_ || expert_index > end_expert_id_) {
    return;
  }

  const std::string suffix = extract_endswith(name);
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(suffix, weight_mapping);
  const int local_index = expert_index % num_experts_per_partition_;
  const bool is_sharded = shard_map.count(index);

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      ep_local_tp_rank_,
                                                      ep_local_tp_size_)
                                 : tensor;

  experts_weights_[suffix][local_index] = tmp_tensor.clone();
}

void Qwen3MoeDecoderLoader::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(name, weight_mapping);
  const bool is_sharded = shard_map.count(index);

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      dp_local_tp_rank_,
                                                      dp_local_tp_size_)
                                       .to(device_)
                                 : tensor.to(device_);
  if (absl::StrContains(name, "down_proj")) {
    at_weight_tensors_[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void Qwen3MoeDecoderLoader::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;

  if (weight_mapping.find(name) == weight_mapping.end()) {
    return;
  }

  const int index = get_mapped_index(name, weight_mapping);
  const bool is_sharded = shard_map.count(index);
  torch::Tensor tmp_tensor;

  if (is_sharded) {
    tmp_tensor = get_sharded_tensor(state_dict,
                                    name,
                                    shard_map.at(index),
                                    dp_local_tp_rank_,
                                    dp_local_tp_size_)
                     .to(device_);
  } else {
    tmp_tensor = tensor.to(device_);
  }

  correct_tensor_dtype(tmp_tensor, name);
  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    auto it = SPECIAL_MULTI_ASSIGN_W8A8.find(name);
    if (it != SPECIAL_MULTI_ASSIGN_W8A8.end()) {
      for (int idx : it->second) {
        at_weight_tensors_[idx] = tmp_tensor;
      }
      return;
    }
  }
  at_weight_tensors_[index] = tmp_tensor;
}

torch::Tensor Qwen3MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim) {
  if (parallel_args_.world_size() > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, parallel_args_.rank(), parallel_args_.world_size());
  } else {
    return state_dict.get_tensor(name);
  }
}

torch::Tensor Qwen3MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim,
    int loacal_tp_rank,
    int local_tp_size) {
  if (local_tp_size > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, loacal_tp_rank, local_tp_size);
  } else {
    return state_dict.get_tensor(name);
  }
}

std::string Qwen3MoeDecoderLoader::extract_endswith(const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.push_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  std::string result = parts[parts.size() - 2] + "." + parts[parts.size() - 1];

  return result;
}

int Qwen3MoeDecoderLoader::extract_expert_index(const std::string& name) {
  std::string prefix = "experts.";
  size_t pos = name.find(prefix);
  if (pos != std::string::npos) {
    pos += prefix.length();
    size_t end_pos = pos;
    while (end_pos < name.length() && std::isdigit(name[end_pos])) {
      ++end_pos;
    }
    if (end_pos > pos) {
      return std::stoi(name.substr(pos, end_pos - pos));
    }
  }

  return -1;
}

void Qwen3MoeDecoderLoader::merge_experts_weights() {
  if (experts_weights_.count("gate_proj.weight") > 0) {
    auto& gate_weight = experts_weights_["gate_proj.weight"];
  }

  if (experts_weights_.count("up_proj.weight") > 0) {
    auto& up_weight = experts_weights_["up_proj.weight"];
  }

  try {
    torch::Tensor mlp_gateup_weight;
    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/true);
      at_weight_tensors_[IN_MLP_GATEUP_OFFSET_EXPERT] =
          merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                                experts_weights_["up_proj.weight_offset"]);
      at_weight_tensors_[IN_MLP_GATEUP_SCALE_EXPERT] =
          merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                                experts_weights_["up_proj.weight_scale"]);
    } else {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/false);
    }
    at_weight_tensors_[IN_MLP_GATEUP_WEIGHT_EXPERT] =
        at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in gateup weight processing: " << e.what();
    throw;
  }

  if (experts_weights_.count("down_proj.weight") > 0) {
    auto& down_weight = experts_weights_["down_proj.weight"];
  }

  try {
    torch::Tensor mlp_down_weight =
        merge_experts_weights(experts_weights_["down_proj.weight"],
                              /*transpose=*/false);

    at_weight_tensors_[IN_MLP_DOWN_WEIGHT_EXPERT] =
        at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();

    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      at_weight_tensors_[IN_MLP_DOWN_OFFSET_EXPERT] =
          merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
      at_weight_tensors_[IN_MLP_DOWN_SCALE_EXPERT] =
          merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in down weight processing: " << e.what();
    throw;
  }
}

torch::Tensor Qwen3MoeDecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(device_);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts.clear();

  return merged_tensor;
}

torch::Tensor Qwen3MoeDecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }
  torch::Tensor merged_tensor = torch::stack(experts_gate, 0).to(device_);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  experts_gate.clear();
  experts_up.clear();

  return merged_tensor;
}

void Qwen3MoeDecoderLoader::resize_experts_weights(int num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    experts_weights_["gate_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["gate_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

void Qwen3MoeDecoderLoader::initialize_tensors(
    const torch::TensorOptions& options) {
  // initializ placeholder
  at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  resize_experts_weights(num_experts_per_partition_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
  expert_group_ = torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
  initialize_weight_tensors(options);
}

void Qwen3MoeDecoderLoader::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

}  // namespace layer
}  // namespace xllm