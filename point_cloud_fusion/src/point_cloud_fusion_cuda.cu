// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "point_cloud_fusion/point_cloud_fusion_cuda.hpp"

namespace point_cloud_fusion {
namespace cuda {

// Fused kernel: Transform, Check, and Append using Atomic Counter with Strided
// Sampling
__global__ void fusedTransformKernel(const uint8_t* input_points,
                                     uint8_t* output_buffer,
                                     unsigned int* global_count,
                                     const CloudMetadata* metadata,
                                     const CudaFieldCopy* copy_plan,
                                     int num_copy_ops,
                                     int num_slots,
                                     int slot_size_points,
                                     int input_point_step,
                                     int output_point_step,
                                     int src_x_offset,
                                     int src_y_offset,
                                     int src_z_offset,
                                     int dst_x_offset,
                                     int dst_y_offset,
                                     int dst_z_offset,
                                     float x_min,
                                     float x_max,
                                     float y_min,
                                     float y_max,
                                     float z_min,
                                     float z_max,
                                     int range_enable) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int slot_idx = idx / slot_size_points;

  if (slot_idx >= num_slots) return;

  int sample_idx = idx % slot_size_points;  // Which sample within this slot (0 to num_samples-1)

  const CloudMetadata& meta = metadata[slot_idx];

  // Check if this thread should process a sample
  if (sample_idx >= meta.num_samples) return;

  // Compute source point index using strided sampling for uniform spatial
  // distribution stride >= 1.0, and we clamp to avoid out-of-bounds
  int source_idx = min(static_cast<int>(sample_idx * meta.stride), meta.num_points - 1);

  // Compute actual position in input buffer (slot_idx * slot_size + source_idx)
  int input_idx = slot_idx * slot_size_points + source_idx;

  const uint8_t* point_ptr = input_points + input_idx * input_point_step;

  float x, y, z;
  // Load XYZ (vectorized if possible)
  if (src_x_offset == 0 && src_y_offset == 4 && src_z_offset == 8 && input_point_step >= 16 &&
      (reinterpret_cast<uintptr_t>(point_ptr) % 16 == 0)) {
    float4 p = *reinterpret_cast<const float4*>(point_ptr);
    x = p.x;
    y = p.y;
    z = p.z;
  } else {
    // Safe unaligned load
    uint8_t* x_dst = reinterpret_cast<uint8_t*>(&x);
    uint8_t* y_dst = reinterpret_cast<uint8_t*>(&y);
    uint8_t* z_dst = reinterpret_cast<uint8_t*>(&z);
    const uint8_t* x_src = point_ptr + src_x_offset;
    const uint8_t* y_src = point_ptr + src_y_offset;
    const uint8_t* z_src = point_ptr + src_z_offset;

    for (int k = 0; k < 4; ++k) x_dst[k] = x_src[k];
    for (int k = 0; k < 4; ++k) y_dst[k] = y_src[k];
    for (int k = 0; k < 4; ++k) z_dst[k] = z_src[k];
  }

  // Skip invalid points
  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;

  float transformed_x, transformed_y, transformed_z;

  if (meta.apply_transform) {
    transformed_x = meta.rotation[0] * x + meta.rotation[1] * y + meta.rotation[2] * z + meta.translation[0];
    transformed_y = meta.rotation[3] * x + meta.rotation[4] * y + meta.rotation[5] * z + meta.translation[1];
    transformed_z = meta.rotation[6] * x + meta.rotation[7] * y + meta.rotation[8] * z + meta.translation[2];
  } else {
    transformed_x = x;
    transformed_y = y;
    transformed_z = z;
  }

  if (range_enable) {
    if (transformed_x < x_min || transformed_x > x_max || transformed_y < y_min || transformed_y > y_max ||
        transformed_z < z_min || transformed_z > z_max)
      return;
  }

  // Atomic append
  unsigned int out_idx = atomicAdd(global_count, 1);

  uint8_t* dest_ptr = output_buffer + out_idx * output_point_step;

  // Copy fields based on plan (safe unaligned copy)
  for (int i = 0; i < num_copy_ops; ++i) {
    const CudaFieldCopy& op = copy_plan[i];
    const uint8_t* src = point_ptr + op.src_offset;
    uint8_t* dst = dest_ptr + op.dst_offset;

    for (int k = 0; k < op.size; ++k) {
      dst[k] = src[k];
    }
  }

  // Safe unaligned store for XYZ
  const uint8_t* x_src = reinterpret_cast<const uint8_t*>(&transformed_x);
  const uint8_t* y_src = reinterpret_cast<const uint8_t*>(&transformed_y);
  const uint8_t* z_src = reinterpret_cast<const uint8_t*>(&transformed_z);
  uint8_t* x_dst = dest_ptr + dst_x_offset;
  uint8_t* y_dst = dest_ptr + dst_y_offset;
  uint8_t* z_dst = dest_ptr + dst_z_offset;

  for (int k = 0; k < 4; ++k) x_dst[k] = x_src[k];
  for (int k = 0; k < 4; ++k) y_dst[k] = y_src[k];
  for (int k = 0; k < 4; ++k) z_dst[k] = z_src[k];
}

// Helper function to check CUDA errors (for functions that can return bool)
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t error = call;                                                                      \
    if (error != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      return false;                                                                                \
    }                                                                                              \
  } while (0)

// Helper macro for CUDA errors in constructors (throws exception)
#define CUDA_CHECK_THROW(call)                                                                                          \
  do {                                                                                                                  \
    cudaError_t error = call;                                                                                           \
    if (error != cudaSuccess) {                                                                                         \
      char error_msg[256];                                                                                              \
      snprintf(error_msg, sizeof(error_msg), "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(error)); \
      throw std::runtime_error(error_msg);                                                                              \
    }                                                                                                                   \
  } while (0)

CudaTransformContext::CudaTransformContext()
    : d_input_points_(nullptr),
      d_accumulated_buffer_(nullptr),
      d_global_count_(nullptr),
      d_cloud_metadata_(nullptr),
      d_copy_plan_(nullptr),
      h_global_count_pinned_(nullptr),
      h_cloud_metadata_pinned_(nullptr),
      stream_(nullptr),
      input_capacity_(0),
      output_capacity_(0),
      current_input_point_step_(0),
      current_output_point_step_(0),
      slot_size_points_(0),
      num_slots_(0),
      metadata_capacity_(0),
      copy_plan_capacity_(0),
      current_src_x_offset_(0),
      current_src_y_offset_(0),
      current_src_z_offset_(0),
      current_dst_x_offset_(0),
      current_dst_y_offset_(0),
      current_dst_z_offset_(0),
      current_x_min_(-INFINITY),
      current_x_max_(INFINITY),
      current_y_min_(-INFINITY),
      current_y_max_(INFINITY),
      current_z_min_(-INFINITY),
      current_z_max_(INFINITY),
      current_range_enable_(false),
      num_copy_ops_(0) {
  CUDA_CHECK_THROW(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)));
  CUDA_CHECK_THROW(cudaMallocHost(&h_global_count_pinned_, sizeof(unsigned int)));
  CUDA_CHECK_THROW(cudaMalloc(&d_global_count_, sizeof(unsigned int)));
}

CudaTransformContext::~CudaTransformContext() {
  cleanup();
  if (h_global_count_pinned_) cudaFreeHost(h_global_count_pinned_);
  if (h_cloud_metadata_pinned_) cudaFreeHost(h_cloud_metadata_pinned_);
  if (d_global_count_) cudaFree(d_global_count_);
  if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void CudaTransformContext::cleanup() {
  if (d_input_points_) {
    cudaFree(d_input_points_);
    d_input_points_ = nullptr;
  }
  if (d_accumulated_buffer_) {
    cudaFree(d_accumulated_buffer_);
    d_accumulated_buffer_ = nullptr;
  }
  if (d_cloud_metadata_) {
    cudaFree(d_cloud_metadata_);
    d_cloud_metadata_ = nullptr;
  }
  if (d_copy_plan_) {
    cudaFree(d_copy_plan_);
    d_copy_plan_ = nullptr;
  }
  input_capacity_ = 0;
  output_capacity_ = 0;
  metadata_capacity_ = 0;
  copy_plan_capacity_ = 0;
}

bool CudaTransformContext::ensureInputCapacity(size_t num_points) {
  if (num_points == 0) return true;  // Nothing to allocate
  if (num_points <= input_capacity_ && d_input_points_) return true;

  if (d_input_points_) cudaFree(d_input_points_);

  size_t new_cap = num_points + num_points / 2;
  size_t bytes = new_cap * current_input_point_step_;

  if (bytes == 0) {
    fprintf(stderr,
            "ERROR: Trying to allocate 0 bytes for input buffer (new_cap=%zu, "
            "point_step=%zu)\n",
            new_cap, current_input_point_step_);
    return false;
  }

  CUDA_CHECK(cudaMalloc(&d_input_points_, bytes));

  input_capacity_ = new_cap;
  return true;
}

bool CudaTransformContext::ensureOutputCapacity(size_t total_max_points, size_t point_step) {
  size_t required_bytes = total_max_points * point_step;
  if (required_bytes == 0) return true;  // Nothing to allocate
  if (required_bytes <= output_capacity_ && d_accumulated_buffer_) return true;

  if (d_accumulated_buffer_) cudaFree(d_accumulated_buffer_);

  size_t new_cap = required_bytes + required_bytes / 2;
  CUDA_CHECK(cudaMalloc(&d_accumulated_buffer_, new_cap));

  output_capacity_ = new_cap;
  return true;
}

bool CudaTransformContext::resetBatch(size_t total_max_points,
                                      size_t max_single_cloud_points,
                                      size_t input_point_step,
                                      size_t output_point_step,
                                      int src_x_offset,
                                      int src_y_offset,
                                      int src_z_offset,
                                      int dst_x_offset,
                                      int dst_y_offset,
                                      int dst_z_offset,
                                      const std::vector<CudaFieldCopy>& copy_plan,
                                      float x_min,
                                      float x_max,
                                      float y_min,
                                      float y_max,
                                      float z_min,
                                      float z_max,
                                      bool range_enable) {
  current_input_point_step_ = input_point_step;
  current_output_point_step_ = output_point_step;
  current_src_x_offset_ = src_x_offset;
  current_src_y_offset_ = src_y_offset;
  current_src_z_offset_ = src_z_offset;
  current_dst_x_offset_ = dst_x_offset;
  current_dst_y_offset_ = dst_y_offset;
  current_dst_z_offset_ = dst_z_offset;
  current_x_min_ = x_min;
  current_x_max_ = x_max;
  current_y_min_ = y_min;
  current_y_max_ = y_max;
  current_z_min_ = z_min;
  current_z_max_ = z_max;
  current_range_enable_ = range_enable;

  // Treat max_single_cloud_points as the per-cloud slot size (padding)
  slot_size_points_ = max_single_cloud_points;
  if (slot_size_points_ == 0) slot_size_points_ = 1;
  num_slots_ = (slot_size_points_ > 0) ? (total_max_points / slot_size_points_) : 0;

  pending_inputs_.clear();
  host_metadata_.clear();

  // Reset global counter
  CUDA_CHECK(cudaMemsetAsync(d_global_count_, 0, sizeof(unsigned int), static_cast<cudaStream_t>(stream_)));

  // Ensure capacity for output buffer
  if (!ensureOutputCapacity(total_max_points, output_point_step)) return false;

  // Ensure capacity for input buffer
  if (!ensureInputCapacity(total_max_points)) return false;

  // Ensure metadata capacity
  if (num_slots_ > metadata_capacity_) {
    if (d_cloud_metadata_) cudaFree(d_cloud_metadata_);
    if (h_cloud_metadata_pinned_) cudaFreeHost(h_cloud_metadata_pinned_);

    size_t new_cap = num_slots_ + 8;  // small buffer
    CUDA_CHECK(cudaMalloc(&d_cloud_metadata_, new_cap * sizeof(CloudMetadata)));
    CUDA_CHECK(cudaMallocHost(&h_cloud_metadata_pinned_, new_cap * sizeof(CloudMetadata)));
    metadata_capacity_ = new_cap;
  }

  // Upload copy plan
  num_copy_ops_ = static_cast<int>(copy_plan.size());
  if (num_copy_ops_ > 0) {
    size_t plan_bytes = num_copy_ops_ * sizeof(CudaFieldCopy);
    if (plan_bytes > copy_plan_capacity_) {
      if (d_copy_plan_) cudaFree(d_copy_plan_);
      CUDA_CHECK(cudaMalloc(&d_copy_plan_, plan_bytes));
      copy_plan_capacity_ = plan_bytes;
    }
    CUDA_CHECK(
        cudaMemcpyAsync(d_copy_plan_, copy_plan.data(), plan_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)));
  }

  return true;
}

bool CudaTransformContext::addCloud(const uint8_t* input_data,
                                    size_t num_points,
                                    const float* rotation_matrix_host,
                                    const float* translation_host,
                                    bool apply_transform,
                                    size_t slot_index,
                                    int desired_points) {
  if (num_points == 0) return true;

  // Compute strided sampling parameters for uniform spatial distribution
  const int total_points = static_cast<int>(num_points);
  int num_samples;
  float stride;

  if (desired_points > 0 && desired_points < total_points) {
    // Downsampling: use strided sampling
    num_samples = desired_points;
    stride = static_cast<float>(total_points) / static_cast<float>(desired_points);
  } else {
    // No downsampling: take all points
    num_samples = total_points;
    stride = 1.0f;
  }

  // Store metadata
  CloudMetadata meta;
  meta.num_points = total_points;
  meta.num_samples = num_samples;
  meta.stride = stride;
  meta.apply_transform = apply_transform ? 1 : 0;

  if (apply_transform) {
    memcpy(meta.rotation, rotation_matrix_host, 9 * sizeof(float));
    memcpy(meta.translation, translation_host, 3 * sizeof(float));
  } else {
    memset(meta.rotation, 0, 9 * sizeof(float));
    memset(meta.translation, 0, 3 * sizeof(float));
  }

  if (host_metadata_.size() <= slot_index) {
    host_metadata_.resize(slot_index + 1);
  }
  host_metadata_[slot_index] = meta;

  // Queue input for upload
  InputCloudInfo info;
  info.data = input_data;
  info.size_bytes = num_points * current_input_point_step_;
  info.slot_index = slot_index;
  pending_inputs_.push_back(info);

  return true;
}

bool CudaTransformContext::getBatchOutput(std::vector<uint8_t>& output_data, size_t& valid_count) {
  if (host_metadata_.empty()) {
    valid_count = 0;
    return true;
  }

  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  // 1. Upload all pending inputs to their slots
  for (const auto& info : pending_inputs_) {
    size_t dest_offset_bytes = info.slot_index * slot_size_points_ * current_input_point_step_;

    CUDA_CHECK(cudaMemcpyAsync(d_input_points_ + dest_offset_bytes, info.data, info.size_bytes, cudaMemcpyHostToDevice, stream));
  }

  // 2. Upload metadata
  size_t meta_count = host_metadata_.size();
  CloudMetadata* pinned_meta = static_cast<CloudMetadata*>(h_cloud_metadata_pinned_);
  memcpy(pinned_meta, host_metadata_.data(), meta_count * sizeof(CloudMetadata));

  CUDA_CHECK(cudaMemcpyAsync(d_cloud_metadata_, pinned_meta, meta_count * sizeof(CloudMetadata), cudaMemcpyHostToDevice, stream));

  // 3. Launch Fused Kernel
  int threads_per_block = 256;
  int total_slot_points = static_cast<int>(num_slots_ * slot_size_points_);
  int num_blocks = (total_slot_points + threads_per_block - 1) / threads_per_block;

  fusedTransformKernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_input_points_, d_accumulated_buffer_, d_global_count_, static_cast<CloudMetadata*>(d_cloud_metadata_),
      static_cast<CudaFieldCopy*>(d_copy_plan_), num_copy_ops_, static_cast<int>(num_slots_), static_cast<int>(slot_size_points_),
      static_cast<int>(current_input_point_step_), static_cast<int>(current_output_point_step_), current_src_x_offset_,
      current_src_y_offset_, current_src_z_offset_, current_dst_x_offset_, current_dst_y_offset_, current_dst_z_offset_,
      current_x_min_, current_x_max_, current_y_min_, current_y_max_, current_z_min_, current_z_max_,
      current_range_enable_ ? 1 : 0);

  CUDA_CHECK(cudaGetLastError());

  // 4. Read back count (Async)
  CUDA_CHECK(cudaMemcpyAsync(h_global_count_pinned_, d_global_count_, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

  // 5. Synchronize to get the count
  CUDA_CHECK(cudaStreamSynchronize(stream));

  unsigned int total_valid = *h_global_count_pinned_;
  valid_count = static_cast<size_t>(total_valid);

  if (valid_count == 0) {
    output_data.clear();
    return true;
  }

  // 6. Resize output vector and read back data
  output_data.resize(valid_count * current_output_point_step_);

  // Direct copy to std::vector (driver will handle staging)
  CUDA_CHECK(
      cudaMemcpy(output_data.data(), d_accumulated_buffer_, valid_count * current_output_point_step_, cudaMemcpyDeviceToHost));

  return true;
}

}  // namespace cuda
}  // namespace point_cloud_fusion
