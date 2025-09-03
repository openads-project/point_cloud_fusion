#pragma once

#include <memory>
#include <string>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <point_cloud_transport/point_cloud_transport.hpp>
#include <point_cloud_transport/subscriber_filter.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace point_cloud_fusion {

template <typename C> struct is_vector : std::false_type {};    
template <typename T,typename A> struct is_vector< std::vector<T,A> > : std::true_type {};    
template <typename C> inline constexpr bool is_vector_v = is_vector<C>::value;


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
  void declareAndLoadParameter(const std::string &name,
                               T &param,
                               const std::string &description,
                               const bool add_to_auto_reconfigurable_params = true,
                               const bool is_required = false,
                               const bool read_only = false,
                               const std::optional<double> &from_value = std::nullopt,
                               const std::optional<double> &to_value = std::nullopt,
                               const std::optional<double> &step_value = std::nullopt,
                               const std::string &additional_constraints = "");
  /**
   * @brief Sets up subscribers, publishers, etc. to configure the node
   */
  void setup();

  /**
   * @brief Process two input point clouds to be fused
   *
   * @param msg message
   */
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg1, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg2);

 private:

  /**
   * @brief Auto-reconfigurable parameters for dynamic reconfiguration
   */
  std::vector<std::tuple<std::string, std::function<void(const rclcpp::Parameter &)>>> auto_reconfigurable_params_;

  /**
   * @brief Subscribers
   */
  std::shared_ptr<point_cloud_transport::SubscriberFilter> cloud_subscriber1_;
  std::shared_ptr<point_cloud_transport::SubscriberFilter> cloud_subscriber2_;

  /**
   * @brief Publisher
   */
  std::shared_ptr<point_cloud_transport::Publisher> cloud_publisher_;

  /**
   * @brief Synchronizer for the two point cloud subscribers
   */
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> cloud_synchronizer_;

  /**
   * @brief TF2 buffer and transform listener
   */
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /**
   * @brief Target frame to which all input point clouds are transformed before fusion
   */
  std::string target_frame_ = "base_link";

  /**
   * @brief Timer to delay setup
   */
  rclcpp::TimerBase::SharedPtr setup_timer_;
};


}
