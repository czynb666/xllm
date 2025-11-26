#include "word_embedding_loader.h"

namespace xllm {
namespace layer {

WordEmbeddingLoader::WordEmbeddingLoader(
    uint64_t weight_count,
    const xllm::ParallelArgs& parallel_args)
    : BaseLoader(weight_count, parallel_args) {
  at_weight_tensors_[0] = torch::zeros({1}).to(options);
}

void WordEmbeddingLoader::verify_loaded_weights(
    const std::string weight_str) const {
  CHECK(at_weight_tensors_[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void WordEmbeddingLoader::merge_loaded_weights() {
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[0]);
  init_layer();
}

void WordEmbeddingLoader::load_state_dict(const StateDict& state_dict) {
  if (dp_size_ > 1) {
    set_weight(
        state_dict, "weight", 0, 1, dp_local_tp_rank_, dp_local_tp_size_);
  } else {
    set_weight(state_dict, "weight", 0, 1);
  }
}

}  // namespace layer
}  // namespace xllm