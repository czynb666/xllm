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

#include "block_copy_loader.h"

namespace xllm {
namespace layer {

BlockCopyLoader::BlockCopyLoader(uint64_t weight_count,
                                 const xllm::ParallelArgs& parallel_args,
                                 at::Device device)
    : BaseLoader(weight_count, parallel_args, device) {}

void BlockCopyLoader::load_state_dict(const StateDict&) {}

void BlockCopyLoader::verify_loaded_weights() const {}

void BlockCopyLoader::merge_loaded_weights() {}

}  // namespace layer
}  // namespace xllm
