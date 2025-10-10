#include <functional>
#include <chrono>

#include <point_cloud_fusion/point_cloud_fusion.hpp>

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_fusion::PointCloudFusion)


namespace point_cloud_fusion {


PointCloudFusion::PointCloudFusion(const rclcpp::NodeOptions& options) : Node("point_cloud_fusion", options) {

  this->declareAndLoadParameter("target_frame", target_frame_, "Target frame of fused point cloud", false, true);
  this->declareAndLoadParameter("input_topics", input_topics_, "List of input point cloud topics", false, true);
  this->declareAndLoadParameter("input_transport_hints", input_transport_hints_, "List of transport hints (one per input)", false);
  this->declareAndLoadParameter("max_time_diff_sec", max_time_diff_sec_, "Max time diff for synchronization (seconds)");

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

  // validate inputs
  if (input_topics_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "No input topics configured (parameter 'input_topics'). Exiting");
    exit(EXIT_FAILURE);
  }
  if (!input_transport_hints_.empty() && input_transport_hints_.size() != input_topics_.size()) {
    RCLCPP_WARN(this->get_logger(), "'input_transport_hints' length (%zu) does not match 'input_topics' (%zu). Missing hints default to 'raw'",
                input_transport_hints_.size(), input_topics_.size());
  }

  // create subscribers
  cloud_subscribers_.resize(input_topics_.size());
  cloud_queues_.assign(input_topics_.size(), {});

  for (size_t i = 0; i < input_topics_.size(); ++i) {
    const std::string configured = input_topics_[i];
    const std::string resolved = this->get_node_topics_interface()->resolve_topic_name(configured);
    const std::string hint = (i < input_transport_hints_.size() && !input_transport_hints_[i].empty()) ? input_transport_hints_[i] : std::string("raw");

    cloud_subscribers_[i] = std::make_shared<point_cloud_transport::SubscriberFilter>();
    cloud_subscribers_[i]->subscribe(this->shared_from_this(), resolved, hint);
    RCLCPP_INFO(this->get_logger(), "Subscribed to '%s' (hint=%s)",
                cloud_subscribers_[i]->getTopic().c_str(), hint.c_str());

    // capture index for callback
    cloud_subscribers_[i]->registerCallback([this, i](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      this->pointCloudCallback(i, msg);
    });
  }

  // create publisher
  point_cloud_transport::PointCloudTransport pct(this->shared_from_this());
  std::string output_topic_name = this->get_node_topics_interface()->resolve_topic_name("~/output");
  cloud_publisher_ = std::make_shared<point_cloud_transport::Publisher>(pct.advertise(output_topic_name, 10));
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", cloud_publisher_->getTopic().c_str());
}

void PointCloudFusion::pointCloudCallback(size_t idx, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
  const auto callback_start = std::chrono::steady_clock::now();
  if (idx >= cloud_queues_.size()) return;

  // push to queue for this input
  auto &queue = cloud_queues_[idx];
  queue.push_back(msg);
  while (queue.size() > max_queue_size_) queue.pop_front();

  // try to find a synchronized set centered on this message's timestamp
  const rclcpp::Time t_ref(msg->header.stamp);
  std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> selected(cloud_queues_.size());
  selected[idx] = msg;

  for (size_t i = 0; i < cloud_queues_.size(); ++i) {
    if (i == idx) continue;
    // find closest msg within window
    const auto &q = cloud_queues_[i];
    sensor_msgs::msg::PointCloud2::ConstSharedPtr best;
    rclcpp::Duration best_dt = rclcpp::Duration::from_seconds(max_time_diff_sec_);
    for (const auto &m : q) {
      const rclcpp::Time t_m(m->header.stamp);
      rclcpp::Duration dt = (t_m > t_ref) ? (t_m - t_ref) : (t_ref - t_m);
      if (dt <= rclcpp::Duration::from_seconds(max_time_diff_sec_) && dt <= best_dt) {
        best_dt = dt;
        best = m;
      }
    }
    if (!best) {
      // not all inputs have a close enough message yet
      RCLCPP_WARN(this->get_logger(),
        "Input %zu: No point cloud found within %.3f s of reference time %.3f (queue size %zu)",
        i, max_time_diff_sec_, t_ref.seconds(), q.size());
      return;
    }
    selected[i] = best;
  }

  // Transform and fuse all selected clouds into target frame
  std::unique_ptr<sensor_msgs::msg::PointCloud2> fused_point_cloud = std::make_unique<sensor_msgs::msg::PointCloud2>();
  bool first = true;
  rclcpp::Time earliest_stamp(0, 0, get_clock()->get_clock_type());
  rclcpp::Time latest_stamp(0, 0, get_clock()->get_clock_type());

  for (const auto &pc_msg : selected) {
    sensor_msgs::msg::PointCloud2 transformed;
    if (pc_msg->header.frame_id != target_frame_) {
      try {
        tf_buffer_->transform(*pc_msg, transformed, target_frame_, tf2::durationFromSec(0.1));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(),
          "Cannot transform Pointcloud: Transformation from its frame (%s) to target_frame (%s) not found: %s",
          pc_msg->header.frame_id.c_str(), target_frame_.c_str(), ex.what());
        return;
      }
    } else {
      transformed = *pc_msg;
    }

    const rclcpp::Time current_stamp(pc_msg->header.stamp);
    if (first) {
      *fused_point_cloud = transformed;
      earliest_stamp = current_stamp;
      latest_stamp = current_stamp;
      first = false;
    } else {
      sensor_msgs::msg::PointCloud2 tmp;
      // concatenate into fused_point_cloud
      pcl::concatenatePointCloud(*fused_point_cloud, transformed, tmp);
      *fused_point_cloud = std::move(tmp);
      if (current_stamp < earliest_stamp) earliest_stamp = current_stamp;
      if (current_stamp > latest_stamp) latest_stamp = current_stamp;
    }

  }

  fused_point_cloud->header.stamp = latest_stamp;
  fused_point_cloud->header.frame_id = target_frame_;
  cloud_publisher_->publish(std::move(fused_point_cloud));
  const double batch_dt_sec = (latest_stamp - earliest_stamp).seconds();
  const double fusion_duration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callback_start).count();
  RCLCPP_INFO(this->get_logger(),
              "Published fused point cloud (%zu inputs), batch_dt=%.6f s, fusion_duration=%.3f ms",
              selected.size(), batch_dt_sec, fusion_duration_ms);
}


}
