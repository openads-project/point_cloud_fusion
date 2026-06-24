// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <point_cloud_transport/point_cloud_transport.hpp>
#include <point_cloud_transport/subscriber_filter.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#ifdef ENABLE_CUDA
#include <point_cloud_fusion/point_cloud_fusion_cuda.hpp>
#endif

namespace point_cloud_fusion {

template <typename C>
struct is_vector : std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};
template <typename C>
inline constexpr bool is_vector_v = is_vector<C>::value;

/**
 * @brief PointCloudFusion class
 */
class PointCloudFusion : public rclcpp::Node {
 public:
  /**
   * @brief Constructor
   *
   * @param options node options
   */
  explicit PointCloudFusion(const rclcpp::NodeOptions& options);

 private:
  /**
   * @brief Declares and loads a ROS parameter
   *
   * @param name name
   * @param param parameter variable to load into
   * @param description description
   * @param add_to_auto_reconfigurable_params enable reconfiguration of
   * parameter
   * @param is_required whether failure to load parameter will stop node
   * @param read_only set parameter to read-only
   * @param from_value parameter range minimum
   * @param to_value parameter range maximum
   * @param step_value parameter range step
   * @param additional_constraints additional constraints description
   */
  template <typename T>
  void declareAndLoadParameter(const std::string& name,
                               T& param,
                               const std::string& description,
                               const bool add_to_auto_reconfigurable_params = true,
                               const bool is_required = false,
                               const bool read_only = false,
                               const std::optional<double>& from_value = std::nullopt,
                               const std::optional<double>& to_value = std::nullopt,
                               const std::optional<double>& step_value = std::nullopt,
                               const std::string& additional_constraints = "");

  /**
   * @brief Validate and apply dynamic parameter updates.
   *
   * @param parameters ROS parameters requested for update.
   * @return Result indicating whether the update was accepted.
   */
  rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter>& parameters);

  /**
   * @brief Sets up subscribers, publishers, etc. to configure the node
   */
  void setup();

  /**
   * @brief Process synchronized point clouds
   *
   * @param msgs batch of synchronized point clouds
   */
  void handleSynchronizedPointClouds(const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>& msgs);

  /**
   * @brief Create the approximate-time synchronizer for a fixed number of input
   * topics.
   *
   * @tparam N Number of synchronized point-cloud inputs.
   */
  template <std::size_t N>
  void setupSynchronizer();

  using PointCloudMsg = sensor_msgs::msg::PointCloud2;

  /**
   * @brief Timing metadata for one synchronized fusion batch.
   */
  struct FusionTiming {
    /**
     * @brief Construct timing metadata with zero-initialized stamps and deltas.
     */
    FusionTiming()
        : earliest_stamp(rclcpp::Time()),
          latest_stamp(rclcpp::Time()),
          input0_stamp(rclcpp::Time()),
          max_dt_from_input0_sec(0.0) {}
    rclcpp::Time earliest_stamp;
    rclcpp::Time latest_stamp;
    rclcpp::Time input0_stamp;
    double max_dt_from_input0_sec;
  };

  /**
   * @brief Collect timestamp statistics for synchronized input clouds.
   *
   * @param msgs Synchronized input point clouds.
   * @param timing Output timing metadata populated from the batch.
   * @return True if timing metadata could be collected for the batch.
   */
  bool collectTimingInfo(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, FusionTiming& timing) const;

  /**
   * @brief Fuse a synchronized point-cloud batch using the CPU path.
   *
   * @param msgs Synchronized input point clouds.
   * @param timing Timing metadata for the batch.
   * @param valid_point_count Number of valid points written to the output
   * cloud.
   * @return Fused point cloud.
   */
  PointCloudMsg::UniquePtr fusePointCloudBatch(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs,
                                               const FusionTiming& timing,
                                               std::size_t& valid_point_count) const;

#ifdef ENABLE_CUDA
  PointCloudMsg::UniquePtr fusePointCloudBatchCUDA(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs,
                                                   const FusionTiming& timing,
                                                   std::size_t& valid_point_count) const;
#endif

  /**
   * @brief Publish a fused point cloud and emit tracing/timing diagnostics.
   *
   * @param cloud Fused point cloud to publish.
   * @param timing Timing metadata for the synchronized input batch.
   * @param input_count Number of input clouds in the batch.
   * @param total_points Number of points before filtering.
   * @param callback_start Time when the synchronized callback started.
   * @param processing_start Time when fusion processing started.
   * @param processing_end Time when fusion processing ended.
   * @param event_name Trace event name for the selected backend.
   */
  void publishFusedCloud(PointCloudMsg::UniquePtr cloud,
                         const FusionTiming& timing,
                         std::size_t input_count,
                         std::size_t total_points,
                         std::chrono::steady_clock::time_point callback_start,
                         std::chrono::steady_clock::time_point processing_start,
                         std::chrono::steady_clock::time_point processing_end,
                         const char* event_name);

 private:
  enum class OutputStampMode { Latest, Earliest, Mean, Input0 };

  /**
   * @brief Parse and store the configured output timestamp mode.
   *
   * @param mode Timestamp mode parameter value.
   */
  void configureOutputStampMode(const std::string& mode);
  /**
   * @brief Validate that the configured input topic list is usable.
   */
  void validateInputTopicsParameter() const;

  /**
   * @brief Validate configured XYZ range limits and disable filtering when
   * invalid.
   */
  void validateRangeLimits();

  static constexpr int32_t kMinSyncQueueSize = 1;
  static constexpr int32_t kMaxSyncQueueSize = 1000;
  static constexpr int32_t kStepSizeSyncQueueSize = 1;
  static constexpr int32_t kMinOutputQueueSize = 1;
  static constexpr int32_t kMaxOutputQueueSize = 1000;
  static constexpr int32_t kStepSizeOutputQueueSize = 1;
  static constexpr int64_t kMinFixedPointsPerInputCloud = 0;
  static constexpr int64_t kMaxFixedPointsPerInputCloud = 10000000;
  static constexpr int64_t kStepSizeFixedPointsPerInputCloud = 100;
  static constexpr std::size_t kMaxInputTopics = 9;
  static constexpr const char* kDefaultTransportHint = "raw";
  static constexpr const char* kAllowedOutputStampModes = "latest, earliest, mean, input0";
  static constexpr double kMinRangeXY = -1000.0;
  static constexpr double kMaxRangeXY = 1000.0;
  static constexpr double kMinRangeZ = -20.0;
  static constexpr double kMaxRangeZ = 20.0;

  /**
   * @brief ROS parameters
   */
  double max_time_diff_sec_ = 0.05;  // 50 ms default window
  double age_penalty_ = 0.1;         // Matches message_filters::ApproximateTime default
  int64_t sync_queue_size_ = 3;      // queue size for synchronizer
  int64_t output_queue_size_ = 10;   // queue size for output publisher
  // Optional: limit each input cloud to this many points before processing.
  // 0 = disabled (use actual point count per cloud)
  int64_t fixed_points_per_input_cloud_ = 0;
  bool range_limits_enable_ = false;
  double range_limits_x_min_ = -1000.0;
  double range_limits_x_max_ = 1000.0;
  double range_limits_y_min_ = -1000.0;
  double range_limits_y_max_ = 1000.0;
  double range_limits_z_min_ = -20.0;
  double range_limits_z_max_ = 20.0;
  bool use_cuda_ = true;
  OutputStampMode output_stamp_mode_ = OutputStampMode::Earliest;
  std::string output_stamp_mode_param_ = "earliest";
  std::string target_frame_ = "base_link";
  std::vector<std::string> output_fields_ = {};
  std::vector<std::string> input_topics_;
  std::vector<std::string> input_transport_hints_ = {};

  /**
   * @brief Auto-reconfigurable parameters for dynamic reconfiguration
   */
  std::vector<std::tuple<std::string, std::function<void(const rclcpp::Parameter&)>>> auto_reconfigurable_params_;
  mutable std::shared_mutex config_mutex_;

  std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>> cloud_subscribers_;
  std::vector<rclcpp::CallbackGroup::SharedPtr> cloud_subscriber_callback_groups_;
  std::shared_ptr<void> synchronizer_;

  /**
   * @brief Callback handle for dynamic parameter reconfiguration
   */
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_;

  /**
   * @brief Publisher
   */
  std::shared_ptr<point_cloud_transport::Publisher> cloud_publisher_;

  /**
   * @brief TF2 buffer and transform listener
   */
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /**
   * @brief Timer to delay setup
   */
  rclcpp::TimerBase::SharedPtr setup_timer_;

#ifdef ENABLE_CUDA
  /**
   * @brief CUDA context for GPU acceleration
   */
  std::unique_ptr<cuda::CudaTransformContext> cuda_context_;
  std::mutex cuda_context_mutex_;
#endif
};

}  // namespace point_cloud_fusion
