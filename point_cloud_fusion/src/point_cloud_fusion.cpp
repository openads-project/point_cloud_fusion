#include <chrono>
#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <cstring>

#include <point_cloud_fusion/point_cloud_fusion.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/time.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_fusion::PointCloudFusion)


namespace point_cloud_fusion {


PointCloudFusion::PointCloudFusion(const rclcpp::NodeOptions& options) : Node("point_cloud_fusion", options) {

  this->declareAndLoadParameter("target_frame", target_frame_, "Target frame of fused point cloud", false, true);
  this->declareAndLoadParameter("input_topics", input_topics_, "List of input point cloud topics", false, true);
  this->declareAndLoadParameter("input_transport_hints", input_transport_hints_, "List of transport hints (one per input)", false);
  this->declareAndLoadParameter("max_time_diff_sec", max_time_diff_sec_, "Max time diff for synchronization (seconds)");
  this->declareAndLoadParameter("filter_invalid_points", filter_invalid_points_, "Filter out NaN/Inf points (slower but safer)");

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

namespace detail {

template <typename MsgT, std::size_t Remaining, typename... Accumulated>
struct ApproximatePolicyBuilder;

template <typename MsgT, typename... Accumulated>
struct ApproximatePolicyBuilder<MsgT, 0, Accumulated...> {
  using type = message_filters::sync_policies::ApproximateTime<Accumulated...>;
};

template <typename MsgT, std::size_t Remaining, typename... Accumulated>
struct ApproximatePolicyBuilder {
  static_assert(Remaining > 0, "Remaining must be positive");
  using type = typename ApproximatePolicyBuilder<MsgT, Remaining - 1, Accumulated..., MsgT>::type;
};

template <std::size_t N>
using SyncPolicy = typename ApproximatePolicyBuilder<sensor_msgs::msg::PointCloud2, N>::type;

template <std::size_t N>
using SyncType = message_filters::Synchronizer<SyncPolicy<N>>;

template <std::size_t N, std::size_t... Is>
void connectInputsImpl(SyncType<N> &sync,
                       const std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>> &subs,
                       std::index_sequence<Is...>) {
  sync.connectInput(*subs[Is]...);
}

template <std::size_t N>
void connectInputs(SyncType<N> &sync,
                   const std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>> &subs) {
  connectInputsImpl<N>(sync, subs, std::make_index_sequence<N>{});
}

}  // namespace detail


void PointCloudFusion::setup() {

  // create transform buffer and listener
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // validate inputs
  if (input_topics_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "No input topics configured (parameter 'input_topics'). Exiting");
    exit(EXIT_FAILURE);
  }
  if (input_topics_.size() > 9) {
    RCLCPP_FATAL(this->get_logger(),
                 "Configured with %zu input topics, but only up to 9 inputs are supported",
                 input_topics_.size());
    exit(EXIT_FAILURE);
  }
  if (!input_transport_hints_.empty() && input_transport_hints_.size() != input_topics_.size()) {
    RCLCPP_WARN(this->get_logger(), "'input_transport_hints' length (%zu) does not match 'input_topics' (%zu). Missing hints default to 'raw'",
                input_transport_hints_.size(), input_topics_.size());
  }

  // create subscribers
  cloud_subscribers_.clear();
  cloud_subscribers_.reserve(input_topics_.size());
  for (size_t i = 0; i < input_topics_.size(); ++i) {
    const std::string configured = input_topics_[i];
    const std::string resolved = this->get_node_topics_interface()->resolve_topic_name(configured);
    const std::string hint = (i < input_transport_hints_.size() && !input_transport_hints_[i].empty()) ? input_transport_hints_[i] : std::string("raw");

    auto subscriber = std::make_shared<point_cloud_transport::SubscriberFilter>();
    subscriber->subscribe(this->shared_from_this(), resolved, hint);
    RCLCPP_INFO(this->get_logger(), "Subscribed to '%s' (hint=%s)",
                subscriber->getTopic().c_str(), hint.c_str());
    cloud_subscribers_.push_back(std::move(subscriber));
  }

  synchronizer_.reset();

  // configure synchronization or direct passthrough
  if (cloud_subscribers_.size() == 1) {
    cloud_subscribers_.front()->registerCallback(
      [this](const PointCloudMsg::ConstSharedPtr msg) {
        std::vector<PointCloudMsg::ConstSharedPtr> batch;
        batch.reserve(1);
        batch.emplace_back(msg);
        this->handleSynchronizedPointClouds(batch);
      });
    RCLCPP_INFO(this->get_logger(),
                "Configured single-input mode for topic '%s'",
                cloud_subscribers_.front()->getTopic().c_str());
  } else if (cloud_subscribers_.size() <= 9) {
    switch (cloud_subscribers_.size()) {
      case 2: setupSynchronizer<2>(); break;
      case 3: setupSynchronizer<3>(); break;
      case 4: setupSynchronizer<4>(); break;
      case 5: setupSynchronizer<5>(); break;
      case 6: setupSynchronizer<6>(); break;
      case 7: setupSynchronizer<7>(); break;
      case 8: setupSynchronizer<8>(); break;
      case 9: setupSynchronizer<9>(); break;
      default:
        RCLCPP_FATAL(this->get_logger(),
                     "Unsupported number of input topics: %zu", cloud_subscribers_.size());
        exit(EXIT_FAILURE);
    }
  }

  // create publisher
  point_cloud_transport::PointCloudTransport pct(this->shared_from_this());
  std::string output_topic_name = this->get_node_topics_interface()->resolve_topic_name("~/output");
  cloud_publisher_ = std::make_shared<point_cloud_transport::Publisher>(pct.advertise(output_topic_name, 10));
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", cloud_publisher_->getTopic().c_str());
}

template <std::size_t N>
void PointCloudFusion::setupSynchronizer() {
  static_assert(N >= 2 && N <= 9, "Supported synchronizer size is between 2 and 9");

  using Policy = detail::SyncPolicy<N>;
  using Sync = detail::SyncType<N>;

  auto sync = std::make_shared<Sync>(Policy(sync_queue_size_));

  detail::connectInputs<N>(*sync, cloud_subscribers_);

  sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_time_diff_sec_));
  sync->registerCallback([this](auto &&... msgs) {
    std::vector<PointCloudMsg::ConstSharedPtr> batch;
    batch.reserve(sizeof...(msgs));
    auto append = [&batch](auto &&m) {
      using ArgT = std::remove_cv_t<std::remove_reference_t<decltype(m)>>;
      if constexpr (std::is_convertible_v<ArgT, PointCloudMsg::ConstSharedPtr>) {
        batch.emplace_back(m);
      }
    };
    (append(std::forward<decltype(msgs)>(msgs)), ...);
    if (!batch.empty()) {
      this->handleSynchronizedPointClouds(batch);
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "ApproximateTime synchronizer yielded no valid point clouds; skipping fusion.");
    }
  });

  synchronizer_ = sync;

  RCLCPP_INFO(this->get_logger(),
              "Configured approximate time synchronizer for %zu inputs (queue=%zu, max_dt=%.3f s)",
              static_cast<size_t>(N), sync_queue_size_, max_time_diff_sec_);
}

void PointCloudFusion::handleSynchronizedPointClouds(const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> &msgs) {
  if (msgs.empty()) {
    return;
  }

  const auto callback_start = std::chrono::steady_clock::now();
  const auto clock_type = this->get_clock()->get_clock_type();

  const auto reference_stamp = rclcpp::Time(msgs.front()->header.stamp);
  rclcpp::Time earliest_stamp(0, 0, clock_type);
  rclcpp::Time latest_stamp(0, 0, clock_type);
  bool first_stamp = true;
  double max_dt_sec = 0.0;

  for (const auto &pc_msg : msgs) {
    if (!pc_msg) {
      RCLCPP_WARN(this->get_logger(), "Received null point cloud pointer in synchronized batch, skipping fusion");
      return;
    }

    const rclcpp::Time current_stamp(pc_msg->header.stamp);
    if (first_stamp) {
      earliest_stamp = current_stamp;
      latest_stamp = current_stamp;
      first_stamp = false;
    } else {
      if (current_stamp < earliest_stamp) earliest_stamp = current_stamp;
      if (current_stamp > latest_stamp) latest_stamp = current_stamp;
    }

    const double dt_sec = std::fabs((current_stamp - reference_stamp).seconds());
    if (dt_sec > max_dt_sec) {
      max_dt_sec = dt_sec;
    }
  }

  const auto transform_start = std::chrono::steady_clock::now();

  // Transform all clouds to target frame using optimized bulk transformation
  std::vector<sensor_msgs::msg::PointCloud2> transformed_clouds;
  transformed_clouds.reserve(msgs.size());

  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto &msg = msgs[i];
    
    if (msg->header.frame_id != target_frame_) {
      try {
        sensor_msgs::msg::PointCloud2 transformed_msg;
        tf_buffer_->transform(*msg, transformed_msg, target_frame_, tf2::durationFromSec(0.1));
        transformed_clouds.push_back(std::move(transformed_msg));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(),
                     "Cannot transform Pointcloud: Transformation from its frame (%s) to target_frame (%s) not found: %s",
                     msg->header.frame_id.c_str(), target_frame_.c_str(), ex.what());
        return;
      }
    } else {
      transformed_clouds.push_back(*msg);
    }
  }

  // Concatenate all transformed clouds using optimized PCL function
  sensor_msgs::msg::PointCloud2 concatenated_msg;
  
  if (!transformed_clouds.empty()) {
    concatenated_msg = transformed_clouds[0];
    for (size_t i = 1; i < transformed_clouds.size(); ++i) {
      pcl::concatenatePointCloud(concatenated_msg, transformed_clouds[i], concatenated_msg);
    }
  }

  concatenated_msg.header.frame_id = target_frame_;
  concatenated_msg.header.stamp = latest_stamp;

  // Fast path: publish concatenated cloud directly without filtering
  if (!filter_invalid_points_) {
    auto fused_point_cloud = std::make_unique<PointCloudMsg>(std::move(concatenated_msg));
    const size_t total_points = fused_point_cloud->width * fused_point_cloud->height;
    
    const auto processing_end = std::chrono::steady_clock::now();
    cloud_publisher_->publish(std::move(fused_point_cloud));
    const auto publish_end = std::chrono::steady_clock::now();

    const double batch_dt_sec = (latest_stamp - earliest_stamp).seconds();
    const double fusion_duration_ms = std::chrono::duration<double, std::milli>(publish_end - callback_start).count();
    const double transform_duration_ms = std::chrono::duration<double, std::milli>(processing_end - transform_start).count();
    const double publish_duration_ms = std::chrono::duration<double, std::milli>(publish_end - processing_end).count();

    RCLCPP_INFO(this->get_logger(),
                "Published fused point cloud (%zu inputs, %zu pts), batch_dt=%.6f s, max_dt=%.6f s, fusion_duration=%.3f ms "
                "(transform=%.3f ms, publish=%.3f ms)",
                msgs.size(), total_points, batch_dt_sec, max_dt_sec, fusion_duration_ms,
                transform_duration_ms, publish_duration_ms);
    return;
  }

  // Filter invalid points by checking x, y, z coordinates
  int x_offset = -1, y_offset = -1, z_offset = -1;
  for (const auto& field : concatenated_msg.fields) {
    if (field.name == "x") x_offset = field.offset;
    else if (field.name == "y") y_offset = field.offset;
    else if (field.name == "z") z_offset = field.offset;
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    // No xyz fields, just publish as-is
    auto fused_point_cloud = std::make_unique<PointCloudMsg>(std::move(concatenated_msg));
    const size_t total_points = fused_point_cloud->width * fused_point_cloud->height;
    
    const auto processing_end = std::chrono::steady_clock::now();
    cloud_publisher_->publish(std::move(fused_point_cloud));
    const auto publish_end = std::chrono::steady_clock::now();

    const double batch_dt_sec = (latest_stamp - earliest_stamp).seconds();
    const double fusion_duration_ms = std::chrono::duration<double, std::milli>(publish_end - callback_start).count();
    const double transform_duration_ms = std::chrono::duration<double, std::milli>(processing_end - transform_start).count();
    const double publish_duration_ms = std::chrono::duration<double, std::milli>(publish_end - processing_end).count();

    RCLCPP_INFO(this->get_logger(),
                "Published fused point cloud (%zu inputs, %zu pts), batch_dt=%.6f s, max_dt=%.6f s, fusion_duration=%.3f ms "
                "(transform=%.3f ms, publish=%.3f ms)",
                msgs.size(), total_points, batch_dt_sec, max_dt_sec, fusion_duration_ms,
                transform_duration_ms, publish_duration_ms);
    return;
  }

  // First pass: count valid points
  const size_t num_points = concatenated_msg.data.size() / concatenated_msg.point_step;
  const uint8_t *src_data = concatenated_msg.data.data();
  
  size_t valid_count = 0;
  for (size_t idx = 0; idx < num_points; ++idx) {
    const uint8_t *point_ptr = src_data + idx * concatenated_msg.point_step;
    const float x = *reinterpret_cast<const float*>(point_ptr + x_offset);
    const float y = *reinterpret_cast<const float*>(point_ptr + y_offset);
    const float z = *reinterpret_cast<const float*>(point_ptr + z_offset);
    
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      ++valid_count;
    }
  }

  if (valid_count == 0) {
    RCLCPP_WARN(this->get_logger(), "All points are invalid, skipping fusion");
    return;
  }

  // Create output message and pre-allocate exact size
  sensor_msgs::msg::PointCloud2::UniquePtr fused_point_cloud = std::make_unique<sensor_msgs::msg::PointCloud2>();
  fused_point_cloud->header = concatenated_msg.header;
  fused_point_cloud->height = 1;
  fused_point_cloud->width = valid_count;
  fused_point_cloud->fields = concatenated_msg.fields;
  fused_point_cloud->is_bigendian = concatenated_msg.is_bigendian;
  fused_point_cloud->point_step = concatenated_msg.point_step;
  fused_point_cloud->row_step = fused_point_cloud->point_step * fused_point_cloud->width;
  fused_point_cloud->is_dense = true;
  
  // Pre-allocate output buffer to avoid reallocations during copying
  fused_point_cloud->data.resize(valid_count * concatenated_msg.point_step);
  
  // Second pass: copy valid points using direct memory operations
  uint8_t *dest_ptr = fused_point_cloud->data.data();
  for (size_t idx = 0; idx < num_points; ++idx) {
    const uint8_t *point_ptr = src_data + idx * concatenated_msg.point_step;
    const float x = *reinterpret_cast<const float*>(point_ptr + x_offset);
    const float y = *reinterpret_cast<const float*>(point_ptr + y_offset);
    const float z = *reinterpret_cast<const float*>(point_ptr + z_offset);
    
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      memcpy(dest_ptr, point_ptr, concatenated_msg.point_step);
      dest_ptr += concatenated_msg.point_step;
    }
  }

  const size_t total_points = fused_point_cloud->width * fused_point_cloud->height;
  
  const auto processing_end = std::chrono::steady_clock::now();
  cloud_publisher_->publish(std::move(fused_point_cloud));
  const auto publish_end = std::chrono::steady_clock::now();

  const double batch_dt_sec = (latest_stamp - earliest_stamp).seconds();
  const double fusion_duration_ms = std::chrono::duration<double, std::milli>(publish_end - callback_start).count();
  const double transform_duration_ms = std::chrono::duration<double, std::milli>(processing_end - transform_start).count();
  const double publish_duration_ms = std::chrono::duration<double, std::milli>(publish_end - processing_end).count();

  RCLCPP_INFO(this->get_logger(),
              "Published fused point cloud (%zu inputs, %zu pts), batch_dt=%.6f s, max_dt=%.6f s, fusion_duration=%.3f ms "
              "(transform=%.3f ms, publish=%.3f ms)",
              msgs.size(), total_points, batch_dt_sec, max_dt_sec, fusion_duration_ms,
              transform_duration_ms, publish_duration_ms);
}


}
