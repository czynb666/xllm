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

#include "base_manual_loader.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {
namespace layer {

namespace {
static inline size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}
}  // namespace

BaseManualLoader::~BaseManualLoader() {
  release_host_storage();
  release_device_storage();
}

void BaseManualLoader::init_weight_slices() {
  weight_slices_.resize(weight_count_);
  size_t offset = 0;
  for (size_t i = 0; i < weight_count_; ++i) {
    weight_slices_[i] = {};
    const auto& tensor = at_host_weight_tensors_[i];
    if (!tensor.defined() || tensor.numel() < 1) {
      continue;
    }
    offset = AlignUp(offset, kHostAlignment);
    weight_slices_[i].offset = offset;
    weight_slices_[i].bytes = tensor.nbytes();
    weight_slices_[i].sizes = tensor.sizes().vec();
    weight_slices_[i].dtype = tensor.scalar_type();
    offset += weight_slices_[i].bytes;
  }
  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(offset, max_alignment);
  storage_size_ = offset;
}

void BaseManualLoader::copy_weights_to_pinned_host() {
  CHECK_GT(storage_size_, 0) << "model size must be greater than 0.";
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";

  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(storage_size_, max_alignment);

  auto ret = aclrtMallocHost(&host_pinned_storage_, storage_size_);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate pinned host storage size=" << storage_size_;

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    auto host_tensor = at_host_weight_tensors_[i].to(torch::kCPU).contiguous();
    void* dst = static_cast<char*>(host_pinned_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    std::memcpy(dst, host_tensor.data_ptr(), slice.bytes);
    at_host_weight_tensors_[i] = at::Tensor();
  }

  ret = aclrtMallocAlign32(
      &device_storage_, storage_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate contiguous device storage size=" << storage_size_;
}

void BaseManualLoader::copy_weights_to_device_async() {
  CHECK_EQ(weight_slices_.size(), at_weight_tensors_.size())
      << "weight_slices_ size and at_weight_tensors_ size mismatch.";
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  void* dst = static_cast<char*>(device_storage_);
  void* src = static_cast<char*>(host_pinned_storage_);

  auto err = aclrtMemcpyAsync(dst,
                              storage_size_,
                              src,
                              storage_size_,
                              ACL_MEMCPY_HOST_TO_DEVICE,
                              stream);
  CHECK_EQ(err, ACL_SUCCESS) << "aclrtMemcpyAsync failed";
}

void BaseManualLoader::copy_weights_to_device() {
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";
  LOG(INFO) << "storage_size_: " << storage_size_;
  auto ret = aclrtMallocAlign32(
      &device_storage_, storage_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate contiguous device storage size=" << storage_size_;

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* dst = static_cast<char*>(device_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    auto host_tensor = at_host_weight_tensors_[i].contiguous();
    auto err = aclrtMemcpy(dst,
                           slice.bytes,
                           host_tensor.data_ptr(),
                           slice.bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_EQ(err, ACL_SUCCESS) << "aclrtMemcpy failed for tensor index " << i;
    at_host_weight_tensors_[i] = at::Tensor();
  }
}

void BaseManualLoader::init_device_at_weights() {
  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* base = static_cast<char*>(device_storage_) +
                 static_cast<ptrdiff_t>(slice.offset);
    at_weight_tensors_[i] = convert_to_torch_tensor(
        slice.sizes, slice.dtype, reinterpret_cast<uintptr_t>(base));
  }
}

void BaseManualLoader::release_device_storage() {
  if (device_storage_ == nullptr) {
    return;
  }
  auto ret = aclrtFree(device_storage_);
  LOG(INFO) << "wfawfaw";
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to free contiguous layer storage, ret=" << ret;
  }
  device_storage_ = nullptr;
}

void BaseManualLoader::release_host_storage() {
  if (host_pinned_storage_ == nullptr) {
    return;
  }
  auto ret = aclrtFreeHost(host_pinned_storage_);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to free pinned host storage, ret=" << ret;
  }
  host_pinned_storage_ = nullptr;
}

BaseManualLoader::BaseManualLoader(uint64_t weight_count,
                                   const ModelContext& context)
    : BaseLoader(weight_count, context) {
  at_host_weight_tensors_.resize(weight_count);
}

torch::Tensor BaseManualLoader::convert_to_torch_tensor(
    const std::vector<int64_t>& dims,
    const torch::ScalarType dtype,
    const uintptr_t& dev_addr,
    int acl_format) {
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  auto tensor = at_npu::native::empty_with_format({0}, option, 2);
  auto address = reinterpret_cast<void*>(dev_addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);
  storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
  storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
  storage.set_data_ptr(std::move(c10_data_ptr));

  tensor.set_(storage, 0, dims);
  // convert to NZ format forcefully, with the underlying data format guaranteed
  // by the developer.
  if (acl_format == 29) {
    // auto* tensor_storage_impl = static_cast<torch_npu::NPUStorageImpl*>(
    //     tensor.storage().unsafeGetStorageImpl());
    // tensor_storage_impl->npu_desc_.npu_format = ACL_FORMAT_FRACTAL_NZ;
    auto tmp_tensor = at_npu::native::npu_format_cast(tensor, acl_format);
    // LOG(INFO) << "tmp_tensor_storage size: " <<
    // tmp_tensor.storage().nbytes();
    void* dst_ptr = tensor.data_ptr();
    const void* src_ptr = tmp_tensor.data_ptr();
    // LOG(INFO) << "tensor_storage size: " << tensor.storage().nbytes();
    aclrtMemcpy(dst_ptr,
                tmp_tensor.storage().nbytes(),
                src_ptr,
                tmp_tensor.storage().nbytes(),
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    auto* tensor_storage_impl = static_cast<torch_npu::NPUStorageImpl*>(
        tensor.storage().unsafeGetStorageImpl());
    auto* tmp_tensor_storage_impl = static_cast<torch_npu::NPUStorageImpl*>(
        tmp_tensor.storage().unsafeGetStorageImpl());
    tensor_storage_impl->npu_desc_ = tmp_tensor_storage_impl->npu_desc_;
    bool sim = torch::allclose(tmp_tensor, tensor, 1e-05, 1e-08, false);
    LOG(INFO) << "is sim: " << sim << "tensor norm: " << torch::norm(tensor)
              << "tmp_tensor norm:" << torch::norm(tmp_tensor) << "sim cos: "
              << torch::cosine_similarity(tensor, tmp_tensor, -1, 1e-8);

    tmp_tensor = torch::Tensor();
  }

  return tensor;
}

}  // namespace layer
}  // namespace xllm
