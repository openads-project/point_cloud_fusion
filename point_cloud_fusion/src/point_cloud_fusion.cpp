#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include <point_cloud_fusion/point_cloud_fusion.hpp>

#include <tf2/LinearMath/Transform.h>
#include <tf2/time.h>
#include <tracetools/tracetools.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <rclcpp_components/register_node_macro.hpp>
#include <rmw/qos_profiles.h>
RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_fusion::PointCloudFusion)

namespace {

inline std::size_t pointFieldDatatypeSize(uint8_t datatype) {
  using sensor_msgs::msg::PointField;
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      return 1;
    case PointField::INT16:
    case PointField::UINT16:
      return 2;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      return 4;
    case PointField::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

}  // namespace

namespace point_cloud_fusion {

// clang-format off
PointCloudFusion::PointCloudFusion(const rclcpp::NodeOptions& options) : Node("point_cloud_fusion", options) {
  this->declareAndLoadParameter("target_frame", target_frame_,                   // name
                                "Target frame of fused point cloud",             // description
                                false,                                           // add_to_auto_reconfigurable_params
                                true,                                            // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Must be set.");                                 // additional_constraints
  this->declareAndLoadParameter("input_topics", input_topics_,                   // name
                                "List of input point cloud topics",              // description
                                false,                                           // add_to_auto_reconfigurable_params
                                true,                                            // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Must configure between 1 and " + std::to_string(kMaxInputTopics) + " topics");
  validateInputTopicsParameter();
  this->declareAndLoadParameter("input_transport_hints", input_transport_hints_, // name
                                "List of transport hints (one per input)",       // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Length must be zero or match input_topics; unspecified entries default to '" +
                                    std::string(kDefaultTransportHint) + "'.");  // additional_constraints
  this->declareAndLoadParameter("sync_queue_size", sync_queue_size_,             // name
                                "Queue size for message_filters synchronizer",   // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                kMinSyncQueueSize,                               // from_value
                                kMaxSyncQueueSize,                               // to_value
                                kStepSizeSyncQueueSize,                          // step_value
                                std::string("Must be >= ") + std::to_string(kMinSyncQueueSize));  // additional_constraints
  this->declareAndLoadParameter("output_queue_size", output_queue_size_,         // name
                                "Queue size for fused output publisher",         // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                kMinOutputQueueSize,                             // from_value
                                kMaxOutputQueueSize,                             // to_value
                                kStepSizeOutputQueueSize,                        // step_value
                                std::string("Must be >= ") + std::to_string(kMinOutputQueueSize)); // additional_constraints
  this->declareAndLoadParameter(
      "output_fields", output_fields_,                                           // name
      "Fields to include in the fused output (empty publishes all fields)",      // description
      true,                                                                      // add_to_auto_reconfigurable_params
      false,                                                                     // is_required
      false,                                                                     // read_only
      std::nullopt, std::nullopt, std::nullopt,                                  // from_value, to_value, step_value
      "Typical fields include: x, y, z, intensity, t, reflectivity, ring, ambient, range.");  // additional_constraints
  this->declareAndLoadParameter("output_stamp_mode", output_stamp_mode_param_,   // name
                                std::string("Timestamp to assign to fused cloud (") + kAllowedOutputStampModes + ")",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                std::string("Allowed values: ") + kAllowedOutputStampModes); // additional_constraints
  // Allow user to optionally limit the per-cloud point count to a maximum
  this->declareAndLoadParameter("fixed_points_per_input_cloud", fixed_points_per_input_cloud_,
                                "If >0, limit each input cloud to this many valid points before fusing",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                0.0,                                             // from_value
                                10000000.0,                                      // to_value
                                1.0,                                             // step_value
                                "0 = disabled; reasonable range is 0 to 10,000,000 points per input cloud");
  this->declareAndLoadParameter("use_cuda", use_cuda_,
                                "Enable CUDA acceleration if available",
                                false, false, true, std::nullopt, std::nullopt, std::nullopt,
                                "Default: true");
  this->declareAndLoadParameter("max_time_diff_sec", max_time_diff_sec_,         // name
                                "Max time diff for synchronization (seconds)",   // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                0.0, std::nullopt, std::nullopt,                 // from_value, to_value, step_value
                                "Must be non-negative");                         // additional_constraints
  this->declareAndLoadParameter("age_penalty", age_penalty_,                     // name
                                "ApproximateTime age penalty",                   // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                0.0,                                             // from_value
                                100.0,                                           // to_value
                                std::nullopt,                                    // step_value
                                "Valid range is [0, 100].");                     // additional_constraints
  configureOutputStampMode(output_stamp_mode_param_);

#ifdef ENABLE_CUDA
  if (use_cuda_) {
    // Initialize CUDA context
    try {
      cuda_context_ = std::make_unique<cuda::CudaTransformContext>();
      RCLCPP_INFO(this->get_logger(), "CUDA acceleration enabled");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA context: %s", e.what());
      RCLCPP_WARN(this->get_logger(), "Falling back to CPU-only implementation");
      cuda_context_.reset();
    }
  } else {
    RCLCPP_INFO(this->get_logger(), "CUDA disabled by parameter; using CPU-only implementation");
  }
#else
  RCLCPP_INFO(this->get_logger(), "CUDA support not compiled, using CPU-only implementation");
#endif

  // run setup after constructor has finished to enable shared_from_this()
  setup_timer_ = this->create_wall_timer(std::chrono::milliseconds(1), [this]() {
    setup();
    setup_timer_->cancel();
  });
}
// clang-format on

template <typename T>
void PointCloudFusion::declareAndLoadParameter(const std::string& name, T& param, const std::string& description,
                                               const bool add_to_auto_reconfigurable_params, const bool is_required,
                                               const bool read_only, const std::optional<double>& from_value,
                                               const std::optional<double>& to_value,
                                               const std::optional<double>& step_value,
                                               const std::string& additional_constraints) {
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.description = description;
  param_desc.additional_constraints = additional_constraints;
  param_desc.read_only = read_only;

  auto type = rclcpp::ParameterValue(param).get_type();

  if (from_value.has_value() && to_value.has_value()) {
    if constexpr (std::is_integral_v<T>) {
      rcl_interfaces::msg::IntegerRange range;
      range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value()));
      if (step_value.has_value()) range.set__step(static_cast<T>(step_value.value()));
      param_desc.integer_range = {range};
    } else if constexpr (std::is_floating_point_v<T>) {
      rcl_interfaces::msg::FloatingPointRange range;
      range.set__from_value(static_cast<T>(from_value.value())).set__to_value(static_cast<T>(to_value.value()));
      if (step_value.has_value()) range.set__step(static_cast<T>(step_value.value()));
      param_desc.floating_point_range = {range};
    } else {
      RCLCPP_WARN(this->get_logger(), "Parameter type of parameter '%s' does not support specifying a range",
                  name.c_str());
    }
  }

  this->declare_parameter(name, type, param_desc);

  try {
    param = this->get_parameter(name).get_value<T>();
    std::stringstream ss;
    ss << "Loaded parameter '" << name << "': ";
    if constexpr (is_vector_v<T>) {
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
      if constexpr (is_vector_v<T>) {
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

rcl_interfaces::msg::SetParametersResult PointCloudFusion::parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters) {
  for (const auto& param : parameters) {
    for (auto& auto_reconfigurable_param : auto_reconfigurable_params_) {
      if (param.get_name() == std::get<0>(auto_reconfigurable_param)) {
        std::get<1>(auto_reconfigurable_param)(param);
        RCLCPP_INFO(this->get_logger(), "Reconfigured parameter '%s' to: %s", param.get_name().c_str(),
                    param.value_to_string().c_str());
        break;
      }
    }
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  return result;
}

namespace detail {

template <std::size_t N>
struct SyncPolicyTraits;

template <>
struct SyncPolicyTraits<2> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<3> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<4> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<5> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<6> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<7> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<8> {
  using Policy =
      message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
                                                      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<9> {
  using Policy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
      sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <std::size_t N>
using SyncPolicy = typename SyncPolicyTraits<N>::Policy;

template <std::size_t N>
using SyncType = message_filters::Synchronizer<SyncPolicy<N>>;

template <std::size_t N, std::size_t... Is>
void connectInputsImpl(SyncType<N>& sync,
                       const std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>>& subs,
                       std::index_sequence<Is...>) {
  // Fan the subscriber filters into the synchronizer inputs.
  sync.connectInput(*subs[Is]...);
}

template <std::size_t N>
void connectInputs(SyncType<N>& sync,
                   const std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>>& subs) {
  connectInputsImpl<N>(sync, subs, std::make_index_sequence<N>{});
}

}  // namespace detail

void PointCloudFusion::setup() {
  // callback for dynamic parameter configuration
  parameters_callback_ = this->add_on_set_parameters_callback(
      std::bind(&PointCloudFusion::parametersCallback, this, std::placeholders::_1));

  // create transform buffer and listener
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // validate inputs
  if (!input_transport_hints_.empty() && input_transport_hints_.size() != input_topics_.size()) {
    RCLCPP_WARN(this->get_logger(),
                "'input_transport_hints' length (%zu) does not match 'input_topics' (%zu). Missing hints default to "
                "'%s'",
                input_transport_hints_.size(), input_topics_.size(), kDefaultTransportHint);
  }

  // create subscribers
  cloud_subscribers_.clear();
  cloud_subscriber_callback_groups_.clear();
  cloud_subscribers_.reserve(input_topics_.size());
  cloud_subscriber_callback_groups_.reserve(input_topics_.size());
  for (size_t i = 0; i < input_topics_.size(); ++i) {
    const std::string configured = input_topics_[i];
    const std::string resolved = this->get_node_topics_interface()->resolve_topic_name(configured);
    const std::string hint = (i < input_transport_hints_.size() && !input_transport_hints_[i].empty())
                                 ? input_transport_hints_[i]
                                 : std::string(kDefaultTransportHint);

    auto callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.callback_group = callback_group;

    auto subscriber = std::make_shared<point_cloud_transport::SubscriberFilter>();
    subscriber->subscribe(this->shared_from_this(), resolved, hint, rmw_qos_profile_default, subscription_options);
    RCLCPP_INFO(this->get_logger(), "Subscribed to '%s' (hint=%s)", subscriber->getTopic().c_str(), hint.c_str());
    cloud_subscriber_callback_groups_.push_back(std::move(callback_group));
    cloud_subscribers_.push_back(std::move(subscriber));
  }

  RCLCPP_INFO(this->get_logger(),
              "Configured %zu input subscriber callback groups for %zu input topics",
              cloud_subscriber_callback_groups_.size());

  synchronizer_.reset();

  // configure synchronization or direct passthrough
  if (cloud_subscribers_.size() == 1) {
    cloud_subscribers_.front()->registerCallback([this](const PointCloudMsg::ConstSharedPtr msg) {
      std::vector<PointCloudMsg::ConstSharedPtr> batch;
      batch.reserve(1);
      batch.emplace_back(msg);
      this->handleSynchronizedPointClouds(batch);
    });
    RCLCPP_INFO(this->get_logger(), "Configured single-input mode for topic '%s'",
                cloud_subscribers_.front()->getTopic().c_str());
  } else if (cloud_subscribers_.size() <= 9) {
    switch (cloud_subscribers_.size()) {
      case 2:
        setupSynchronizer<2>();
        break;
      case 3:
        setupSynchronizer<3>();
        break;
      case 4:
        setupSynchronizer<4>();
        break;
      case 5:
        setupSynchronizer<5>();
        break;
      case 6:
        setupSynchronizer<6>();
        break;
      case 7:
        setupSynchronizer<7>();
        break;
      case 8:
        setupSynchronizer<8>();
        break;
      case 9:
        setupSynchronizer<9>();
        break;
      default:
        RCLCPP_FATAL(this->get_logger(), "Unsupported number of input topics: %zu", cloud_subscribers_.size());
        exit(EXIT_FAILURE);
    }
  }

  // create publisher
  point_cloud_transport::PointCloudTransport pct(this->shared_from_this());
  std::string output_topic_name = this->get_node_topics_interface()->resolve_topic_name("~/output");
  cloud_publisher_ = std::make_shared<point_cloud_transport::Publisher>(
      pct.advertise(output_topic_name, static_cast<uint32_t>(output_queue_size_)));
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", cloud_publisher_->getTopic().c_str());

  // Annotate message links for tracing: Each publisher (for raw and compressed point clouds) depends an all input point clouds.
  std::vector<const void*> link_subs;
  std::vector<const void*> link_pubs;
  for (const auto& sub_filter : cloud_subscribers_) {
    auto sub_base = sub_filter->getSubscriber().getSubscription();
    if (sub_base) {
      link_subs.push_back(static_cast<const void*>(sub_base->get_subscription_handle().get()));
    }
  }
  if (cloud_publisher_) {
    std::map<std::string, rclcpp::PublisherBase::SharedPtr> pubs_base = cloud_publisher_->getPublishers();
    for (const auto& [transport, pub_base] : pubs_base) {
      link_pubs.push_back(static_cast<const void*>(pub_base->get_publisher_handle().get()));
    }
  }
  TRACETOOLS_TRACEPOINT(message_link_partial_sync, link_subs.data(), link_subs.size(), link_pubs.data(),
                        link_pubs.size());
}

template <std::size_t N>
void PointCloudFusion::setupSynchronizer() {
  static_assert(N >= 2 && N <= 9, "Supported synchronizer size is between 2 and 9");

  using Policy = detail::SyncPolicy<N>;
  using Sync = detail::SyncType<N>;

  // Instantiate ApproximateTime policy tuned to the active input count.
  auto sync = std::make_shared<Sync>(Policy(static_cast<size_t>(sync_queue_size_)));

  // Wire the configured subscribers into the synchronizer slots.
  detail::connectInputs<N>(*sync, cloud_subscribers_);

  sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_time_diff_sec_));
  sync->setAgePenalty(age_penalty_);
  sync->registerCallback([this](auto&&... msgs) {
    std::vector<PointCloudMsg::ConstSharedPtr> batch;
    batch.reserve(sizeof...(msgs));
    // Extract only valid PointCloud2 pointers from the variadic callback.
    auto append = [&batch](auto&& msg) {
      using ArgT = std::decay_t<decltype(msg)>;
      if constexpr (std::is_same_v<ArgT, PointCloudMsg::ConstSharedPtr>) {
        if (msg) {
          batch.emplace_back(std::forward<decltype(msg)>(msg));
        }
      }
    };
    (append(std::forward<decltype(msgs)>(msgs)), ...);
    if (!batch.empty()) {
      this->handleSynchronizedPointClouds(batch);
    } else {
      RCLCPP_WARN(this->get_logger(), "ApproximateTime synchronizer yielded no valid point clouds; skipping fusion.");
    }
  });

  synchronizer_ = sync;

  RCLCPP_INFO(this->get_logger(),
              "Configured approximate time synchronizer for %zu inputs (queue=%zu, max_dt=%.3f s, age_penalty=%.6f)",
              static_cast<size_t>(N), static_cast<size_t>(sync_queue_size_), max_time_diff_sec_, age_penalty_);
}

void PointCloudFusion::handleSynchronizedPointClouds(
    const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>& msgs) {
  if (msgs.empty()) {
    return;
  }

  const auto callback_start = std::chrono::steady_clock::now();

  FusionTiming timing;
  if (!collectTimingInfo(msgs, timing)) {
    return;
  }

#ifdef ENABLE_CUDA
  std::size_t cuda_valid_count = 0;
  auto processing_start = std::chrono::steady_clock::time_point{};
  auto processing_end = std::chrono::steady_clock::time_point{};
  PointCloudMsg::UniquePtr cuda_result;
  bool used_cuda = false;
  {
    // Guard shared CUDA pipeline state against concurrent synchronized callbacks.
    std::lock_guard<std::mutex> cuda_lock(cuda_context_mutex_);
    // Run either CPU or CUDA implementation based on parameter.
    if (cuda_context_ && use_cuda_) {
      used_cuda = true;
      processing_start = std::chrono::steady_clock::now();
      cuda_result = fusePointCloudBatchCUDA(msgs, timing, cuda_valid_count);
      processing_end = std::chrono::steady_clock::now();
    }
  }

  if (used_cuda) {
    if (cuda_result) {
      publishFusedCloud(std::move(cuda_result), timing, msgs.size(), cuda_valid_count, callback_start, processing_start,
                        processing_end, "cuda_fusion_complete");
    } else {
      RCLCPP_WARN(this->get_logger(), "CUDA processing failed");
    }
    return;
  }
#endif

  // CPU-only path
  std::size_t valid_count = 0;
  const auto processing_start = std::chrono::steady_clock::now();
  auto fused_point_cloud = fusePointCloudBatch(msgs, timing, valid_count);

  if (!fused_point_cloud) {
    RCLCPP_WARN(this->get_logger(), "All points are invalid, skipping fusion");
    return;
  }

  const auto processing_end = std::chrono::steady_clock::now();
  publishFusedCloud(std::move(fused_point_cloud), timing, msgs.size(), valid_count, callback_start, processing_start,
                    processing_end, "cpu_fusion_complete");
}

bool PointCloudFusion::collectTimingInfo(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs,
                                         FusionTiming& timing) const {
  if (msgs.empty()) {
    return false;
  }

  const auto input0_stamp = rclcpp::Time(msgs.front()->header.stamp);
  bool first_stamp = true;
  double max_dt_from_input0_sec = 0.0;
  rclcpp::Time earliest_stamp;
  rclcpp::Time latest_stamp;

  // Walk every cloud once to gather min/max stamps and the largest skew from input0.
  for (const auto& pc_msg : msgs) {
    if (!pc_msg) {
      RCLCPP_WARN(this->get_logger(), "Received null point cloud pointer in synchronized batch, skipping fusion");
      return false;
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

    const double dt_sec = std::fabs((current_stamp - input0_stamp).seconds());
    if (dt_sec > max_dt_from_input0_sec) {
      max_dt_from_input0_sec = dt_sec;
    }
  }

  timing.earliest_stamp = earliest_stamp;
  timing.latest_stamp = latest_stamp;
  timing.input0_stamp = input0_stamp;
  timing.max_dt_from_input0_sec = max_dt_from_input0_sec;
  return true;
}

PointCloudFusion::PointCloudMsg::UniquePtr PointCloudFusion::fusePointCloudBatch(
    const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, const FusionTiming& timing,
    std::size_t& valid_point_count) const {
  if (msgs.empty()) {
    return nullptr;
  }

  const auto& input0_msg = msgs.front();
  if (!input0_msg) {
    return nullptr;
  }

  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  for (const auto& field : input0_msg->fields) {
    if (field.name == "x")
      x_offset = field.offset;
    else if (field.name == "y")
      y_offset = field.offset;
    else if (field.name == "z")
      z_offset = field.offset;
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    RCLCPP_WARN(this->get_logger(), "Point cloud lacks x/y/z fields; skipping fusion for this batch.");
    valid_point_count = 0;
    return nullptr;
  }

  const size_t point_step = input0_msg->point_step;
  const auto& input0_fields = input0_msg->fields;
  const bool is_bigendian = input0_msg->is_bigendian;

  struct FieldCopyPlan {
    const sensor_msgs::msg::PointField* source;
    sensor_msgs::msg::PointField destination;
    std::size_t byte_length;
  };

  bool use_all_fields = output_fields_.empty();
  std::vector<FieldCopyPlan> copy_plan;
  std::vector<sensor_msgs::msg::PointField> fused_fields;
  fused_fields.reserve(input0_fields.size());

  std::size_t fused_point_step = point_step;
  int fused_x_offset = x_offset;
  int fused_y_offset = y_offset;
  int fused_z_offset = z_offset;

  if (!use_all_fields) {
    bool selection_valid = true;
    fused_point_step = 0;
    fused_fields.clear();
    copy_plan.reserve(output_fields_.size());

    for (const auto& requested_name : output_fields_) {
      auto iter = std::find_if(
          input0_fields.begin(), input0_fields.end(),
          [&requested_name](const sensor_msgs::msg::PointField& field) { return field.name == requested_name; });
      if (iter == input0_fields.end()) {
        RCLCPP_WARN(
            this->get_logger(),
            "Requested output field '%s' not present in incoming point cloud; publishing full field set instead.",
            requested_name.c_str());
        selection_valid = false;
        break;
      }

      const std::size_t datatype_size = pointFieldDatatypeSize(iter->datatype);
      if (datatype_size == 0) {
        RCLCPP_WARN(this->get_logger(),
                    "Point field '%s' uses unsupported datatype %u; publishing full field set instead.",
                    requested_name.c_str(), static_cast<unsigned int>(iter->datatype));
        selection_valid = false;
        break;
      }

      FieldCopyPlan plan;
      plan.source = &(*iter);
      plan.destination = *iter;
      plan.destination.offset = static_cast<uint32_t>(fused_point_step);
      plan.byte_length = datatype_size * static_cast<std::size_t>(iter->count);
      fused_point_step += plan.byte_length;

      if (requested_name == "x")
        fused_x_offset = plan.destination.offset;
      else if (requested_name == "y")
        fused_y_offset = plan.destination.offset;
      else if (requested_name == "z")
        fused_z_offset = plan.destination.offset;

      copy_plan.push_back(plan);
      fused_fields.push_back(plan.destination);
    }

    if (!selection_valid || fused_x_offset < 0 || fused_y_offset < 0 || fused_z_offset < 0) {
      if (selection_valid) {
        RCLCPP_ERROR(this->get_logger(),
                     "Output field selection must include x, y, and z; publishing full field set instead.");
      }
      use_all_fields = true;
      copy_plan.clear();
      fused_point_step = point_step;
      fused_fields.assign(input0_fields.begin(), input0_fields.end());
      fused_x_offset = x_offset;
      fused_y_offset = y_offset;
      fused_z_offset = z_offset;
    }
  }

  if (use_all_fields) {
    fused_fields.assign(input0_fields.begin(), input0_fields.end());
    fused_point_step = point_step;
    fused_x_offset = x_offset;
    fused_y_offset = y_offset;
    fused_z_offset = z_offset;
  }

  // Reserve enough space once so the fusion loop only appends into a pre-sized buffer.
  const size_t max_capacity = std::accumulate(
      msgs.begin(), msgs.end(), static_cast<size_t>(0), [this](size_t sum, const PointCloudMsg::ConstSharedPtr& cloud) {
        if (!cloud) {
          return sum;
        }
        size_t cloud_size = static_cast<size_t>(cloud->width) * static_cast<size_t>(cloud->height);
        // If user set a cap, limit each cloud to that size
        if (fixed_points_per_input_cloud_ > 0) {
          cloud_size = std::min(cloud_size, static_cast<size_t>(fixed_points_per_input_cloud_));
        }
        return sum + cloud_size;
      });

  if (max_capacity == 0) {
    valid_point_count = 0;
    return nullptr;
  }

  auto output = std::make_unique<PointCloudMsg>();
  output->header.frame_id = target_frame_;
  rclcpp::Time chosen_stamp;
  switch (output_stamp_mode_) {
    case OutputStampMode::Earliest:
      chosen_stamp = timing.earliest_stamp;
      break;
    case OutputStampMode::Mean: {
      const auto delta = timing.latest_stamp - timing.earliest_stamp;
      chosen_stamp = timing.earliest_stamp + rclcpp::Duration::from_nanoseconds(delta.nanoseconds() / 2);
      break;
    }
    case OutputStampMode::Input0:
      chosen_stamp = timing.input0_stamp;
      break;
    case OutputStampMode::Latest:
    default:
      chosen_stamp = timing.latest_stamp;
      break;
  }
  output->header.stamp = chosen_stamp;
  output->height = 1;
  output->is_bigendian = is_bigendian;
  output->point_step = fused_point_step;
  output->fields = fused_fields;
  output->is_dense = true;
  output->data.resize(max_capacity * fused_point_step);

  uint8_t* dest_ptr = output->data.data();
  valid_point_count = 0;
  std::size_t skipped_inputs = 0;

  for (const auto& msg : msgs) {
    if (!msg) {
      continue;
    }

    if (msg->point_step != point_step || msg->fields != input0_fields) {
      RCLCPP_WARN(this->get_logger(), "Skipping point cloud '%s' due to incompatible field layout.",
                  msg->header.frame_id.c_str());
      ++skipped_inputs;
      continue;
    }

    // Cache the frame transform once per cloud to avoid repeated TF queries inside the point loop.
    const bool apply_transform = msg->header.frame_id != target_frame_;
    tf2::Vector3 translation;
    tf2::Matrix3x3 rotation;
    if (apply_transform) {
      tf2::Transform tf_transform;
      geometry_msgs::msg::TransformStamped tf_stamped;
      try {
        tf_stamped = tf_buffer_->lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp,
                                                 rclcpp::Duration::from_seconds(0.1));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(), "Cannot transform point cloud from %s to %s: %s", msg->header.frame_id.c_str(),
                     target_frame_.c_str(), ex.what());
        ++skipped_inputs;
        continue;
      }
      tf2::fromMsg(tf_stamped.transform, tf_transform);
      translation = tf_transform.getOrigin();
      rotation = tf_transform.getBasis();
    }

    const auto* src_data = msg->data.data();
    const size_t total_points = static_cast<size_t>(msg->width) * static_cast<size_t>(msg->height);

    auto emit_point = [&](const uint8_t* point_ptr, float x, float y, float z) {
      if (use_all_fields) {
        std::memcpy(dest_ptr, point_ptr, point_step);
      } else {
        for (const auto& plan : copy_plan) {
          std::memcpy(dest_ptr + plan.destination.offset, point_ptr + plan.source->offset, plan.byte_length);
        }
      }

      if (apply_transform) {
        const tf2::Vector3 rotated = rotation * tf2::Vector3(x, y, z) + translation;
        auto* dest_x = reinterpret_cast<float*>(dest_ptr + fused_x_offset);
        auto* dest_y = reinterpret_cast<float*>(dest_ptr + fused_y_offset);
        auto* dest_z = reinterpret_cast<float*>(dest_ptr + fused_z_offset);
        *dest_x = static_cast<float>(rotated.x());
        *dest_y = static_cast<float>(rotated.y());
        *dest_z = static_cast<float>(rotated.z());
      } else {
        // XYZ already live in the target frame; copying the raw bytes is enough.
      }

      dest_ptr += fused_point_step;
      ++valid_point_count;
    };

    if (fixed_points_per_input_cloud_ <= 0 || static_cast<size_t>(fixed_points_per_input_cloud_) >= total_points) {
      // Fast path: when no downsampling is requested.
      for (size_t idx = 0; idx < total_points; ++idx) {
        const auto* point_ptr = src_data + idx * point_step;
        const float x = *reinterpret_cast<const float*>(point_ptr + x_offset);
        const float y = *reinterpret_cast<const float*>(point_ptr + y_offset);
        const float z = *reinterpret_cast<const float*>(point_ptr + z_offset);

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }

        emit_point(point_ptr, x, y, z);
      }
      continue;
    }

    // Downsample path: strided sampling for uniform spatial distribution.
    const size_t desired_points = static_cast<size_t>(fixed_points_per_input_cloud_);
    const double stride = static_cast<double>(total_points) / static_cast<double>(desired_points);
    const size_t num_samples = desired_points;

    for (size_t i = 0; i < num_samples; ++i) {
      const size_t idx = std::min(static_cast<size_t>(i * stride), total_points - 1);

      const auto* point_ptr = src_data + idx * point_step;
      const float x = *reinterpret_cast<const float*>(point_ptr + x_offset);
      const float y = *reinterpret_cast<const float*>(point_ptr + y_offset);
      const float z = *reinterpret_cast<const float*>(point_ptr + z_offset);

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      emit_point(point_ptr, x, y, z);
    }
  }

  if (valid_point_count == 0) {
    if (skipped_inputs == msgs.size()) {
      RCLCPP_WARN(this->get_logger(), "Skipped all point clouds in synchronized batch; no data fused.");
    }
    return nullptr;
  }

  output->width = valid_point_count;
  output->row_step = output->point_step * output->width;
  output->data.resize(valid_point_count * fused_point_step);

  output->is_dense = true;
  return output;
}

void PointCloudFusion::publishFusedCloud(PointCloudMsg::UniquePtr cloud, const FusionTiming& timing,
                                         std::size_t input_count, std::size_t total_points,
                                         std::chrono::steady_clock::time_point callback_start,
                                         std::chrono::steady_clock::time_point processing_start,
                                         std::chrono::steady_clock::time_point processing_end, const char* event_name) {
  // Publish the fused cloud and emit a compact timing summary for observability.
  cloud_publisher_->publish(std::move(cloud));
  const auto publish_end = std::chrono::steady_clock::now();

  const double prep_duration_ms = std::chrono::duration<double, std::milli>(processing_start - callback_start).count();
  const double processing_duration_ms =
      std::chrono::duration<double, std::milli>(processing_end - processing_start).count();
  const double publish_duration_ms = std::chrono::duration<double, std::milli>(publish_end - processing_end).count();
  const double e2e_duration_ms = prep_duration_ms + processing_duration_ms + publish_duration_ms;
  const double batch_dt_ms = (timing.latest_stamp - timing.earliest_stamp).seconds() * 1000.0;

  RCLCPP_DEBUG(this->get_logger(),
               "%s inputs=%zu points=%zu e2e_ms=%.3f prep_ms=%.3f process_ms=%.3f publish_ms=%.3f batch_dt_ms=%.3f "
               "max_dt_ms=%.3f",
               event_name, input_count, total_points, e2e_duration_ms, prep_duration_ms, processing_duration_ms,
               publish_duration_ms, batch_dt_ms, timing.max_dt_from_input0_sec * 1000.0);
}

void PointCloudFusion::configureOutputStampMode(const std::string& mode) {
  std::string lowered = mode;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (lowered == "earliest") {
    output_stamp_mode_ = OutputStampMode::Earliest;
  } else if (lowered == "mean" || lowered == "mid" || lowered == "midpoint") {
    output_stamp_mode_ = OutputStampMode::Mean;
  } else if (lowered == "input0") {
    output_stamp_mode_ = OutputStampMode::Input0;
  } else if (lowered == "latest") {
    output_stamp_mode_ = OutputStampMode::Latest;
  } else {
    RCLCPP_WARN(this->get_logger(), "Invalid output_stamp_mode '%s'; defaulting to 'earliest'.", mode.c_str());
    output_stamp_mode_ = OutputStampMode::Earliest;
  }
}

void PointCloudFusion::validateInputTopicsParameter() const {
  if (input_topics_.empty()) {
    RCLCPP_FATAL(this->get_logger(), "No input topics configured (parameter 'input_topics'). Exiting");
    exit(EXIT_FAILURE);
  }
  if (input_topics_.size() > kMaxInputTopics) {
    RCLCPP_FATAL(this->get_logger(), "Configured with %zu input topics, but only up to %zu inputs are supported",
                 input_topics_.size(), kMaxInputTopics);
    exit(EXIT_FAILURE);
  }
}

#ifdef ENABLE_CUDA
PointCloudFusion::PointCloudMsg::UniquePtr PointCloudFusion::fusePointCloudBatchCUDA(
    const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, const FusionTiming& timing,
    std::size_t& valid_point_count) const {
  if (msgs.empty() || !cuda_context_) {
    return nullptr;
  }

  const auto& input0_msg = msgs.front();
  if (!input0_msg) {
    return nullptr;
  }

  // Find x, y, z field offsets
  int x_offset = -1, y_offset = -1, z_offset = -1;
  for (const auto& field : input0_msg->fields) {
    if (field.name == "x")
      x_offset = field.offset;
    else if (field.name == "y")
      y_offset = field.offset;
    else if (field.name == "z")
      z_offset = field.offset;
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    RCLCPP_WARN(this->get_logger(), "Point cloud lacks x/y/z fields; skipping CUDA fusion");
    valid_point_count = 0;
    return nullptr;
  }

  const size_t point_step = input0_msg->point_step;
  const auto& input0_fields = input0_msg->fields;
  const bool is_bigendian = input0_msg->is_bigendian;

  // Determine output fields and copy plan
  bool use_all_fields = output_fields_.empty();
  std::vector<sensor_msgs::msg::PointField> fused_fields;
  fused_fields.reserve(input0_fields.size());
  std::vector<cuda::CudaFieldCopy> copy_plan;

  std::size_t fused_point_step = point_step;
  int fused_x_offset = x_offset;
  int fused_y_offset = y_offset;
  int fused_z_offset = z_offset;

  if (!use_all_fields) {
    bool selection_valid = true;
    fused_point_step = 0;
    fused_fields.clear();
    copy_plan.reserve(output_fields_.size());

    for (const auto& requested_name : output_fields_) {
      auto iter = std::find_if(
          input0_fields.begin(), input0_fields.end(),
          [&requested_name](const sensor_msgs::msg::PointField& field) { return field.name == requested_name; });
      if (iter == input0_fields.end()) {
        RCLCPP_WARN(
            this->get_logger(),
            "Requested output field '%s' not present in incoming point cloud; publishing full field set instead.",
            requested_name.c_str());
        selection_valid = false;
        break;
      }

      const std::size_t datatype_size = pointFieldDatatypeSize(iter->datatype);
      if (datatype_size == 0) {
        RCLCPP_WARN(this->get_logger(),
                    "Point field '%s' uses unsupported datatype %u; publishing full field set instead.",
                    requested_name.c_str(), static_cast<unsigned int>(iter->datatype));
        selection_valid = false;
        break;
      }

      cuda::CudaFieldCopy plan;
      plan.src_offset = static_cast<int>(iter->offset);
      plan.dst_offset = static_cast<int>(fused_point_step);
      plan.size = static_cast<int>(datatype_size * iter->count);

      sensor_msgs::msg::PointField dest_field = *iter;
      dest_field.offset = static_cast<uint32_t>(fused_point_step);

      fused_point_step += plan.size;

      if (requested_name == "x")
        fused_x_offset = dest_field.offset;
      else if (requested_name == "y")
        fused_y_offset = dest_field.offset;
      else if (requested_name == "z")
        fused_z_offset = dest_field.offset;

      copy_plan.push_back(plan);
      fused_fields.push_back(dest_field);
    }

    if (!selection_valid || fused_x_offset < 0 || fused_y_offset < 0 || fused_z_offset < 0) {
      if (selection_valid) {
        RCLCPP_ERROR(this->get_logger(),
                     "Output field selection must include x, y, and z; publishing full field set instead.");
      }
      use_all_fields = true;
      copy_plan.clear();
      fused_point_step = point_step;
      fused_fields.assign(input0_fields.begin(), input0_fields.end());
      fused_x_offset = x_offset;
      fused_y_offset = y_offset;
      fused_z_offset = z_offset;
    }
  }

  if (use_all_fields) {
    fused_fields.assign(input0_fields.begin(), input0_fields.end());
    fused_point_step = point_step;
    fused_x_offset = x_offset;
    fused_y_offset = y_offset;
    fused_z_offset = z_offset;

    // If using all fields, we can just copy the whole point step as one chunk
    // But we still need to overwrite XYZ.
    // Actually, if we use all fields, we can just create a single copy op for the whole point
    // OR we can iterate over fields if we want to be precise, but copying the whole struct is faster.
    // Let's just create one copy op for the whole point.
    cuda::CudaFieldCopy plan;
    plan.src_offset = 0;
    plan.dst_offset = 0;
    plan.size = static_cast<int>(point_step);
    copy_plan.push_back(plan);
  }

  // Calculate max points for batch reset
  // For strided sampling, slots must hold the FULL input cloud (we sample from all points)
  // But output capacity only needs space for the downsampled count
  size_t total_output_points = 0;
  size_t max_single_cloud_points = 0;  // Max points in any single input cloud (for slot sizing)
  for (const auto& msg : msgs) {
    if (msg) {
      size_t cloud_size = msg->width * msg->height;
      max_single_cloud_points = std::max(max_single_cloud_points, cloud_size);

      // Output count is limited by fixed_points_per_input_cloud_ if set
      size_t output_from_cloud = (fixed_points_per_input_cloud_ > 0)
                                     ? std::min(cloud_size, static_cast<size_t>(fixed_points_per_input_cloud_))
                                     : cloud_size;
      total_output_points += output_from_cloud;
    }
  }

  if (max_single_cloud_points == 0) {
    valid_point_count = 0;
    return nullptr;
  }

  // slot_size is the FULL max cloud size (not capped) for strided sampling to work
  size_t slot_size = max_single_cloud_points;

  // Total input capacity is slot_size * number_of_inputs
  const size_t num_inputs = msgs.size();
  size_t total_input_capacity = slot_size * num_inputs;

  // Reset batch with fixed slots
  // Note: total_input_capacity is for input buffers, but output may be smaller due to downsampling
  if (!cuda_context_->resetBatch(total_input_capacity, slot_size, point_step, fused_point_step, x_offset, y_offset,
                                 z_offset, fused_x_offset, fused_y_offset, fused_z_offset, copy_plan)) {
    RCLCPP_ERROR(this->get_logger(), "CUDA resetBatch failed");
    return nullptr;
  }

  // Process each cloud and write into its fixed slot index (preserve input order)
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& msg = msgs[i];
    if (!msg || msg->point_step != point_step || msg->fields != input0_fields) {
      continue;
    }

    const bool apply_transform = (msg->header.frame_id != target_frame_);
    float rotation_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float translation[3] = {0, 0, 0};

    if (apply_transform) {
      try {
        auto tf_stamped = tf_buffer_->lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp,
                                                      rclcpp::Duration::from_seconds(0.1));
        tf2::Transform tf_transform;
        tf2::fromMsg(tf_stamped.transform, tf_transform);
        auto rot = tf_transform.getBasis();
        auto trans = tf_transform.getOrigin();

        rotation_matrix[0] = rot[0][0];
        rotation_matrix[1] = rot[0][1];
        rotation_matrix[2] = rot[0][2];
        rotation_matrix[3] = rot[1][0];
        rotation_matrix[4] = rot[1][1];
        rotation_matrix[5] = rot[1][2];
        rotation_matrix[6] = rot[2][0];
        rotation_matrix[7] = rot[2][1];
        rotation_matrix[8] = rot[2][2];
        translation[0] = trans.x();
        translation[1] = trans.y();
        translation[2] = trans.z();
      } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR(this->get_logger(), "CUDA: Cannot transform %s to %s: %s", msg->header.frame_id.c_str(),
                     target_frame_.c_str(), ex.what());
        continue;
      }
    }

    // Send ALL points to GPU - strided sampling happens in the kernel
    size_t num_points = msg->width * msg->height;
    const uint8_t* data_ptr = msg->data.data();

    // Pass desired_points for strided sampling (0 = all points)
    const int desired_points =
        (fixed_points_per_input_cloud_ > 0) ? static_cast<int>(fixed_points_per_input_cloud_) : 0;

    if (!cuda_context_->addCloud(data_ptr, num_points, rotation_matrix, translation, apply_transform, i,
                                 desired_points)) {
      RCLCPP_ERROR(this->get_logger(), "CUDA addCloud failed");
      continue;
    }
  }

  // Get result directly into output vector
  auto output = std::make_unique<PointCloudMsg>();
  output->header.frame_id = target_frame_;
  rclcpp::Time chosen_stamp;
  switch (output_stamp_mode_) {
    case OutputStampMode::Earliest:
      chosen_stamp = timing.earliest_stamp;
      break;
    case OutputStampMode::Mean: {
      const auto delta = timing.latest_stamp - timing.earliest_stamp;
      chosen_stamp = timing.earliest_stamp + rclcpp::Duration::from_nanoseconds(delta.nanoseconds() / 2);
      break;
    }
    case OutputStampMode::Input0:
      chosen_stamp = timing.input0_stamp;
      break;
    case OutputStampMode::Latest:
    default:
      chosen_stamp = timing.latest_stamp;
      break;
  }
  output->header.stamp = chosen_stamp;
  output->height = 1;
  output->is_bigendian = is_bigendian;
  output->point_step = fused_point_step;
  output->fields = fused_fields;
  output->is_dense = true;

  if (!cuda_context_->getBatchOutput(output->data, valid_point_count)) {
    RCLCPP_ERROR(this->get_logger(), "CUDA getBatchOutput failed");
    return nullptr;
  }

  if (valid_point_count == 0) {
    return nullptr;
  }

  output->width = valid_point_count;
  output->row_step = output->point_step * output->width;

  return output;
}
#endif  // ENABLE_CUDA

}  // namespace point_cloud_fusion
