#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
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
   * @brief Processes messages received by a subscriber
   *
   * @param msg message
   */
  void topicCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

 private:

  /**
   * @brief Auto-reconfigurable parameters for dynamic reconfiguration
   */
  std::vector<std::tuple<std::string, std::function<void(const rclcpp::Parameter &)>>> auto_reconfigurable_params_;

  /**
   * @brief Subscriber
   */
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;

  /**
   * @brief Publisher
   */
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

  /**
   * @brief Dummy parameter (parameter) 
   */
  std::string target_frame_ = "base_link";
};


}
