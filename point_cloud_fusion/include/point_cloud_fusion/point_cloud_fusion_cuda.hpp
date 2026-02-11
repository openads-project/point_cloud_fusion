#pragma once

#ifdef ENABLE_CUDA

#include <cstddef>
#include <cstdint>
#include <vector>

namespace point_cloud_fusion {
namespace cuda {

struct BatchCloudInfo {
  const uint8_t* input_data;
  size_t num_points;
  size_t point_step;
  int x_offset;
  int y_offset;
  int z_offset;
  const float* rotation_matrix;  // 9 floats
  const float* translation;      // 3 floats
  bool apply_transform;
};

struct CudaFieldCopy {
  int src_offset;
  int dst_offset;
  int size;
};

// Metadata struct for device consumption
struct CloudMetadata {
  int num_points;           // Actual number of points in this cloud
  int num_samples;          // Number of samples to take (for strided sampling)
  float stride;             // Stride for uniform sampling (>= 1.0)
  int apply_transform;      // bool as int for alignment
  float rotation[9];
  float translation[3];
  int pad;                  // alignment padding
};

/**
 * @brief CUDA context for point cloud transformation operations
 * 
 * Manages GPU memory and provides methods for transforming point clouds on GPU.
 * Memory is reused across multiple calls for efficiency.
 */
class CudaTransformContext {
 public:
  CudaTransformContext();
  ~CudaTransformContext();

  // Disable copy/move
  CudaTransformContext(const CudaTransformContext&) = delete;
  CudaTransformContext& operator=(const CudaTransformContext&) = delete;

  /**
   * @brief Prepare for a new batch of point clouds
   * 
   * @param total_max_points Maximum possible number of points in the batch (sum of all clouds)
   * @param max_single_cloud_points Maximum number of points in any single cloud
   * @param input_point_step Size of each input point in bytes
   * @param output_point_step Size of each output point in bytes
   * @param src_x_offset Byte offset to x coordinate in input
   * @param src_y_offset Byte offset to y coordinate in input
   * @param src_z_offset Byte offset to z coordinate in input
   * @param dst_x_offset Byte offset to x coordinate in output
   * @param dst_y_offset Byte offset to y coordinate in output
   * @param dst_z_offset Byte offset to z coordinate in output
   * @param copy_plan Vector of field copy operations
   * @return true on success
   */
  bool resetBatch(size_t total_max_points, size_t max_single_cloud_points, size_t input_point_step,
                  size_t output_point_step, int src_x_offset, int src_y_offset, int src_z_offset, int dst_x_offset,
                  int dst_y_offset, int dst_z_offset, const std::vector<CudaFieldCopy>& copy_plan);

  /**
   * @brief Add a point cloud to the current batch
   * 
   * @param input_data Raw point cloud data
   * @param num_points Number of points in input
   * @param rotation_matrix_host 3x3 rotation matrix (row-major, 9 floats)
   * @param translation_host Translation vector (3 floats)
   * @param apply_transform Whether to apply transformation
   * @param slot_index which fixed-size slot to write into (0..num_slots-1)
   * @param desired_points Desired number of points to sample (0 = all points, uses strided sampling)
   * @return true on success, false on error
   */
  bool addCloud(const uint8_t* input_data, size_t num_points, const float* rotation_matrix_host,
                const float* translation_host, bool apply_transform, size_t slot_index, int desired_points = 0);

  /**
   * @brief Get the accumulated results from the batch
   * 
   * @param output_data Output point cloud data (full points with all fields)
   * @param valid_count Number of valid (finite) points output
   * @return true on success, false on error
   */
  bool getBatchOutput(std::vector<uint8_t>& output_data, size_t& valid_count);

 private:
  bool ensureInputCapacity(size_t num_points);
  bool ensureOutputCapacity(size_t total_max_points, size_t point_step);
  void cleanup();

  // Device memory
  uint8_t* d_input_points_;
  uint8_t* d_accumulated_buffer_;
  unsigned int* d_global_count_;      // Atomic counter for total output
  unsigned int* d_per_cloud_counts_;  // Atomic counter per cloud for limiting
  void* d_cloud_metadata_;            // Device buffer for CloudMetadata
  void* d_copy_plan_;                 // Device buffer for CudaFieldCopy

  // Host pinned memory
  unsigned int* h_global_count_pinned_;
  void* h_cloud_metadata_pinned_;  // Host pinned buffer for CloudMetadata

  // CUDA resources
  void* stream_;  // cudaStream_t

  size_t input_capacity_;
  size_t output_capacity_;
  size_t current_input_point_step_;
  size_t current_output_point_step_;
  size_t current_output_offset_;
  size_t slot_size_points_;   // fixed slot size in points (padding per cloud)
  size_t num_slots_;          // number of slots (usually number of input clouds)
  size_t metadata_capacity_;  // number of slots we can store metadata for
  size_t copy_plan_capacity_;

  int current_src_x_offset_;
  int current_src_y_offset_;
  int current_src_z_offset_;
  int current_dst_x_offset_;
  int current_dst_y_offset_;
  int current_dst_z_offset_;
  int num_copy_ops_;

  // We keep a host vector of metadata to fill before upload
  std::vector<CloudMetadata> host_metadata_;

  // Stored input pointers for batch upload
  struct InputCloudInfo {
    const uint8_t* data;
    size_t size_bytes;
    size_t slot_index;
  };
  std::vector<InputCloudInfo> pending_inputs_;
};

}  // namespace cuda
}  // namespace point_cloud_fusion

#endif  // ENABLE_CUDA
