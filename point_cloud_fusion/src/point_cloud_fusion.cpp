#include <functional>
#include <chrono>

#include <point_cloud_fusion/point_cloud_fusion.hpp>

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_fusion::PointCloudFusion)


namespace point_cloud_fusion {


PointCloudFusion::PointCloudFusion(const rclcpp::NodeOptions& options) : Node("point_cloud_fusion", options) {

  this->declareAndLoadParameter("target_frame", target_frame_, "Target frame of fused point cloud", false, true);

  // run setup after constructor has finished to enable shared_from_this()
  setup_timer_ = this->create_wall_timer(std::chrono::milliseconds(1), [this]() {
    setup();
    setup_timer_->cancel();
  });

}


template <typename T>
void PointCloudFusion::declareAndLoadParameter(const std::string& name,
                                               T& param,
                                               const std::string& description,
                                               const bool add_to_auto_reconfigurable_params,
                                               const bool is_required,
                                               const bool read_only,
                                               const std::optional<double>& from_value,
                                               const std::optional<double>& to_value,
                                               const std::optional<double>& step_value,
                                               const std::string& additional_constraints) {

  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.description = description;
  param_desc.additional_constraints = additional_constraints;
  param_desc.read_only = read_only;

  auto type = rclcpp::ParameterValue(param).get_type();

  if (from_value.has_value() && to_value.has_value()) {
    if constexpr(std::is_integral_v<T>) {
      rcl_interfaces::msg::IntegerRange range;
      T step = static_cast<T>(step_value.has_value() ? step_value.value() : 1);
      range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value())).set__step(step);
      param_desc.integer_range = {range};
    } else if constexpr(std::is_floating_point_v<T>) {
      rcl_interfaces::msg::FloatingPointRange range;
      T step = static_cast<T>(step_value.has_value() ? step_value.value() : 1.0);
      range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value())).set__step(step);
      param_desc.floating_point_range = {range};
    } else {
      RCLCPP_WARN(this->get_logger(), "Parameter type of parameter '%s' does not support specifying a range", name.c_str());
    }
  }

  this->declare_parameter(name, type, param_desc);

  try {
    param = this->get_parameter(name).get_value<T>();
    std::stringstream ss;
    ss << "Loaded parameter '" << name << "': ";
    if constexpr(is_vector_v<T>) {
      ss << "[";
      for (const auto& element : param) ss << element << (&element != &param.back() ? ", " : "");
      ss << "]";
    } else {
      ss << param;
    }
    RCLCPP_INFO_STREAM(this->get_logger(), ss.str());
  } catch (rclcpp::exceptions::ParameterUninitializedException&) {
    if (is_required) {
      RCLCPP_FATAL_STREAM(this->get_logger(), "Missing required parameter '" << name << "', exiting");
      exit(EXIT_FAILURE);
    } else {
      std::stringstream ss;
      ss << "Missing parameter '" << name << "', using default value: ";
      if constexpr(is_vector_v<T>) {
        ss << "[";
        for (const auto& element : param) ss << element << (&element != &param.back() ? ", " : "");
        ss << "]";
      } else {
        ss << param;
      }
      RCLCPP_WARN_STREAM(this->get_logger(), ss.str());
      this->set_parameters({rclcpp::Parameter(name, rclcpp::ParameterValue(param))});
    }
  }

  if (add_to_auto_reconfigurable_params) {
    std::function<void(const rclcpp::Parameter&)> setter = [&param](const rclcpp::Parameter& p) {
      param = p.get_value<T>();
    };
    auto_reconfigurable_params_.push_back(std::make_tuple(name, setter));
  }
}


void PointCloudFusion::setup() {

  // create transform buffer and listener
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // create subscribers
  std::string input1_topic = this->get_node_topics_interface()->resolve_topic_name("~/input1");
  std::string input2_topic = this->get_node_topics_interface()->resolve_topic_name("~/input2");
  point_cloud_transport::TransportHints transport_hint1(this->shared_from_this(), "raw", "point_cloud_transport1");
  point_cloud_transport::TransportHints transport_hint2(this->shared_from_this(), "raw", "point_cloud_transport2");
  cloud_subscriber1_ = std::make_shared<point_cloud_transport::SubscriberFilter>();
  cloud_subscriber2_ = std::make_shared<point_cloud_transport::SubscriberFilter>();
  cloud_subscriber1_->subscribe(this->shared_from_this(), input1_topic, transport_hint1.getTransport());
  cloud_subscriber2_->subscribe(this->shared_from_this(), input2_topic, transport_hint2.getTransport());

  // create a synchronizer with approximate time policy
  cloud_synchronizer_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(20), *cloud_subscriber1_, *cloud_subscriber2_); // queue size
  cloud_synchronizer_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.05));                                                           // max interval duration: only pair point clouds if their timestamps differ by ≤ 50 ms.
  cloud_synchronizer_->registerCallback(std::bind(&PointCloudFusion::pointCloudCallback, this, std::placeholders::_1, std::placeholders::_2));

  // create publisher
  point_cloud_transport::PointCloudTransport pct(this->shared_from_this());
  std::string output_topic_name = this->get_node_topics_interface()->resolve_topic_name("~/output");
  cloud_publisher_ = std::make_shared<point_cloud_transport::Publisher>(pct.advertise(output_topic_name, 10));
  RCLCPP_INFO(this->get_logger(), "Subscribed to '%s' and '%s' (synchronized)", cloud_subscriber1_->getTopic().c_str(), cloud_subscriber2_->getTopic().c_str());
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", cloud_publisher_->getTopic().c_str());
}


void PointCloudFusion::pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg1,
                                          const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg2) {
  RCLCPP_INFO(this->get_logger(), "Received synchronized point clouds - timestamp diff: %.3f ms",
    std::abs((rclcpp::Time(msg1->header.stamp) - rclcpp::Time(msg2->header.stamp)).seconds() * 1000.0));

  // transform sensor_msgs::msg::PointCloud2 msg if required
  sensor_msgs::msg::PointCloud2::UniquePtr transformed_point_cloud_1 = std::make_unique<sensor_msgs::msg::PointCloud2>();
  sensor_msgs::msg::PointCloud2::UniquePtr transformed_point_cloud_2 = std::make_unique<sensor_msgs::msg::PointCloud2>();
  sensor_msgs::msg::PointCloud2::UniquePtr fused_point_cloud = std::make_unique<sensor_msgs::msg::PointCloud2>();

  std::vector<std::pair<const sensor_msgs::msg::PointCloud2::ConstSharedPtr&, sensor_msgs::msg::PointCloud2::UniquePtr&>> point_clouds = {
    {msg1, transformed_point_cloud_1},
    {msg2, transformed_point_cloud_2}
  };

  for (auto& [msg, transformed_point_cloud] : point_clouds) {
    if (msg->header.frame_id != target_frame_) {
      try {
        tf_buffer_->transform(*msg, *transformed_point_cloud, target_frame_, tf2::durationFromSec(0.1));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(),
          "Cannot transform Pointcloud: Transformation from its frame (%s) to inference_frame (%s) not found: %s",
          msg->header.frame_id.c_str(), target_frame_.c_str(), ex.what());
        return;
      }
    } else {
      *transformed_point_cloud = *msg;
    }
  }

  // concatenate the two transformed point clouds
  pcl::concatenatePointCloud(*transformed_point_cloud_1, *transformed_point_cloud_2, *fused_point_cloud);
  fused_point_cloud->header.stamp =    // apply time stamp of the most recent point cloud
    (rclcpp::Time(msg1->header.stamp) > rclcpp::Time(msg2->header.stamp))
      ? msg1->header.stamp : msg2->header.stamp;
  cloud_publisher_->publish(std::move(fused_point_cloud));
  RCLCPP_INFO(this->get_logger(), "Published fused point cloud (synchronized)");
}


}
