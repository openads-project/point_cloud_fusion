#pragma once

#include <chrono>
#include <memory>
#include <optional>
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
   * @param add_to_auto_reconfigurable_params enable reconfiguration of parameter
   * @param is_required whether failure to load parameter will stop node
   * @param read_only set parameter to read-only
   * @param from_value parameter range minimum
   * @param to_value parameter range maximum
   * @param step_value parameter range step
   * @param additional_constraints additional constraints description
   */
  template <typename T>
  void declareAndLoadParameter(const std::string& name, T& param, const std::string& description,
                               const bool add_to_auto_reconfigurable_params = true, const bool is_required = false,
                               const bool read_only = false, const std::optional<double>& from_value = std::nullopt,
                               const std::optional<double>& to_value = std::nullopt,
                               const std::optional<double>& step_value = std::nullopt,
                               const std::string& additional_constraints = "");
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

  template <std::size_t N>
  void setupSynchronizer();

  using PointCloudMsg = sensor_msgs::msg::PointCloud2;

  struct FusionTiming {
    FusionTiming()
        : earliest_stamp(rclcpp::Time()),
          latest_stamp(rclcpp::Time()),
          reference_stamp(rclcpp::Time()),
          max_dt_sec(0.0) {}
    rclcpp::Time earliest_stamp;
    rclcpp::Time latest_stamp;
    rclcpp::Time reference_stamp;
    double max_dt_sec;
  };

  bool collectTimingInfo(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, FusionTiming& timing) const;

  PointCloudMsg::UniquePtr fusePointCloudBatch(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs,
                                               const FusionTiming& timing, std::size_t& valid_point_count) const;

  void publishFusedCloud(PointCloudMsg::UniquePtr cloud, const FusionTiming& timing, std::size_t input_count,
                         std::size_t total_points, std::chrono::steady_clock::time_point callback_start,
                         std::chrono::steady_clock::time_point transform_start,
                         std::chrono::steady_clock::time_point processing_end);

 private:
  enum class OutputStampMode { Latest, Earliest, Mean, Reference };

  void configureOutputStampMode(const std::string& mode);
  void validateInputTopicsParameter() const;

  static constexpr int64_t kMinSyncQueueSize = 1;
  static constexpr std::size_t kMaxInputTopics = 9;
  static constexpr const char* kDefaultTransportHint = "raw";
  static constexpr const char* kAllowedOutputStampModes = "latest, earliest, mean, reference";

  /**
   * @brief ROS parameters
   */
  double max_time_diff_sec_ = 0.05;  // 50 ms default window
  int64_t sync_queue_size_ = 3;      // queue size for synchronizer
  OutputStampMode output_stamp_mode_ = OutputStampMode::Earliest;
  std::string output_stamp_mode_param_ = "earliest";
  std::string target_frame_ = "base_link";
  std::vector<std::string> output_fields_;
  std::vector<std::string> input_topics_;
  std::vector<std::string> input_transport_hints_;

  /**
   * @brief Auto-reconfigurable parameters for dynamic reconfiguration
   */
  std::vector<std::tuple<std::string, std::function<void(const rclcpp::Parameter&)>>> auto_reconfigurable_params_;

  std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>> cloud_subscribers_;
  std::shared_ptr<void> synchronizer_;

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
};

}  // namespace point_cloud_fusion
