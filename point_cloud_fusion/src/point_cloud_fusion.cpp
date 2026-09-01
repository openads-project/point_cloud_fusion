// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include <point_cloud_fusion/point_cloud_fusion.hpp>
#include <point_cloud_fusion/point_cloud_layout.hpp>

#include <tf2/LinearMath/Transform.h>
#include <tf2/time.h>
#include <tracetools/tracetools.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <rmw/qos_profiles.h>
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_fusion::PointCloudFusion)

namespace {

/**
 * @brief Check whether a point lies inside the configured inclusive XYZ bounds.
 *
 * @param x Point x coordinate in the target frame.
 * @param y Point y coordinate in the target frame.
 * @param z Point z coordinate in the target frame.
 * @param x_min Minimum accepted x coordinate.
 * @param x_max Maximum accepted x coordinate.
 * @param y_min Minimum accepted y coordinate.
 * @param y_max Maximum accepted y coordinate.
 * @param z_min Minimum accepted z coordinate.
 * @param z_max Maximum accepted z coordinate.
 * @return True when the point lies within every configured bound.
 */
inline bool pointWithinRange(
    float x, float y, float z, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
  return x >= x_min && x <= x_max && y >= y_min && y <= y_max && z >= z_min && z <= z_max;
}

/**
 * @brief Advance a byte pointer by an offset.
 *
 * @tparam Byte Byte element type, optionally const-qualified.
 * @param data Base byte pointer.
 * @param offset Number of bytes to advance.
 * @return Pointer at the requested byte offset.
 */
template <typename Byte>
inline Byte* byteOffset(Byte* data, std::size_t offset) {
  return std::next(data, static_cast<std::ptrdiff_t>(offset));
}

/**
 * @brief Load a potentially unaligned float from a byte buffer.
 *
 * @param data Base byte pointer.
 * @param offset Byte offset of the float.
 * @return Loaded float value.
 */
inline float loadFloat(const uint8_t* data, std::size_t offset) {
  float value = 0.0F;
  std::memcpy(&value, byteOffset(data, offset), sizeof(value));
  return value;
}

inline uint32_t loadUint32(const uint8_t* data, std::size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, byteOffset(data, offset), sizeof(value));
  return value;
}

/**
 * @brief Store a float in a potentially unaligned byte buffer.
 *
 * @param data Base byte pointer.
 * @param offset Byte offset of the float.
 * @param value Float value to store.
 */
inline void storeFloat(uint8_t* data, std::size_t offset, float value) {
  std::memcpy(byteOffset(data, offset), &value, sizeof(value));
}

}  // namespace

namespace point_cloud_fusion {

// clang-format off
PointCloudFusion::PointCloudFusion(const rclcpp::NodeOptions& options) : Node("point_cloud_fusion", options) {
  this->declareAndLoadParameter("target_frame", target_frame_,                   // name
                                "Frame into which all input point clouds are transformed before fusion",  // description
                                false,                                           // add_to_auto_reconfigurable_params
                                true,                                            // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Must be set.");                                 // additional_constraints
  this->declareAndLoadParameter("input_topics", input_topics_,                   // name
                                "Point-cloud topics to fuse",                    // description
                                false,                                           // add_to_auto_reconfigurable_params
                                true,                                            // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Must configure between 1 and " + std::to_string(kMaxInputTopics) + " topics");
  validateInputTopicsParameter();
  this->declareAndLoadParameter("input_transport_hints", input_transport_hints_, // name
                                "Transport hint for each input topic; unspecified entries use raw",  // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Length must be zero or match input_topics; unspecified entries default to '" +
                                    std::string(kDefaultTransportHint) + "'.");  // additional_constraints
  this->declareAndLoadParameter("sync_queue_size", sync_queue_size_,             // name
                                "Queue depth for approximate-time synchronization",  // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                kMinSyncQueueSize,                               // from_value
                                kMaxSyncQueueSize,                               // to_value
                                kStepSizeSyncQueueSize,                          // step_value
                                std::string("Must be >= ") + std::to_string(kMinSyncQueueSize));  // additional_constraints
  this->declareAndLoadParameter("output_queue_size", output_queue_size_,         // name
                                "Queue depth for the fused output publisher",    // description
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                kMinOutputQueueSize,                             // from_value
                                kMaxOutputQueueSize,                             // to_value
                                kStepSizeOutputQueueSize,                        // step_value
                                std::string("Must be >= ") + std::to_string(kMinOutputQueueSize)); // additional_constraints
  this->declareAndLoadParameter(
      "output_fields", output_fields_,                                           // name
      "Fields retained in the fused output; an empty list retains all input fields",  // description
      true,                                                                      // add_to_auto_reconfigurable_params
      false,                                                                     // is_required
      false,                                                                     // read_only
      std::nullopt, std::nullopt, std::nullopt,                                  // from_value, to_value, step_value
      "Typical fields include: x, y, z, intensity, t, reflectivity, ring, ambient, range.");  // additional_constraints
  this->declareAndLoadParameter("output_stamp_mode", output_stamp_mode_param_,   // name
                                "Fused timestamp selection: earliest, latest, mean, or input0",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                std::string("Allowed values: ") + kAllowedOutputStampModes); // additional_constraints
  // Allow user to optionally limit the per-cloud point count to a maximum
  this->declareAndLoadParameter("fixed_points_per_input_cloud", fixed_points_per_input_cloud_,
                                "Runtime-reconfigurable maximum valid point count per input cloud; 0 disables the limit",
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinFixedPointsPerInputCloud,                    // from_value
                                kMaxFixedPointsPerInputCloud,                    // to_value
                                kStepSizeFixedPointsPerInputCloud,               // step_value
                                "0 = disabled; reasonable range is 0 to 10,000,000 points per input cloud");
  this->declareAndLoadParameter("use_cuda", use_cuda_,                           // name
                                "Runtime-reconfigurable backend selection; true uses CUDA and false uses CPU",
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Runtime changes apply between fusion batches.");  // additional_constraints
  this->declareAndLoadParameter("motion_compensation.enable", motion_compensation_enable_,
                                "Compensate inter-sensor capture skew and motion during each scan",
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Requires a connected TF trajectory through motion_compensation.fixed_frame; "
                                "the per-point time field is optional.");
  this->declareAndLoadParameter("motion_compensation.fixed_frame", motion_compensation_fixed_frame_,
                                "World-fixed TF frame used to compare sensor poses at different times",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "Typically map or odom; must connect target_frame and every input frame.");
  this->declareAndLoadParameter("motion_compensation.time_field", motion_compensation_time_field_,
                                "UINT32 point field containing an offset from the cloud header stamp",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "If absent, capture-time compensation remains active but scan deskewing is skipped.");
  this->declareAndLoadParameter("motion_compensation.time_scale_sec", motion_compensation_time_scale_sec_,
                                "Seconds represented by one unit of the per-point time field",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                1.0e-12,                                         // from_value
                                1.0,                                             // to_value
                                std::nullopt,                                    // step_value
                                "Use 1e-9 for Ouster nanosecond offsets.");      // additional_constraints
  this->declareAndLoadParameter("motion_compensation.tf_timeout_sec", motion_compensation_tf_timeout_sec_,
                                "Timeout for the first motion-compensation TF failure [s]",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                0.0,                                             // from_value
                                1.0,                                             // to_value
                                std::nullopt,                                    // step_value
                                "The entire batch falls back to rigid transforms; recovery probes do not wait.");
  this->declareAndLoadParameter("range_limits.enable", range_limits_enable_,     // name
                                "Enable XYZ range filtering after transformation into target_frame",
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                std::nullopt, std::nullopt, std::nullopt,        // from_value, to_value, step_value
                                "When false, no range filtering is applied.");   // additional_constraints
  this->declareAndLoadParameter("range_limits.x_min", range_limits_x_min_,       // name
                                "Minimum x coordinate in target_frame to keep [m]",  // description
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinRangeXY,                                     // from_value
                                kMaxRangeXY,                                     // to_value
                                std::nullopt,                                    // step_value
                                "Must be less than range_limits.x_max.");        // additional_constraints
  this->declareAndLoadParameter("range_limits.x_max", range_limits_x_max_,       // name
                                "Maximum x coordinate in target_frame to keep [m]",  // description
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinRangeXY,                                     // from_value
                                kMaxRangeXY,                                     // to_value
                                std::nullopt,                                    // step_value
                                "Must be greater than range_limits.x_min.");     // additional_constraints
  this->declareAndLoadParameter("range_limits.y_min", range_limits_y_min_,       // name
                                "Minimum y coordinate in target_frame to keep [m]",  // description
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinRangeXY,                                     // from_value
                                kMaxRangeXY,                                     // to_value
                                std::nullopt,                                    // step_value
                                "Must be less than range_limits.y_max.");        // additional_constraints
  this->declareAndLoadParameter("range_limits.y_max", range_limits_y_max_,       // name
                                "Maximum y coordinate in target_frame to keep [m]",  // description
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinRangeXY,                                     // from_value
                                kMaxRangeXY,                                     // to_value
                                std::nullopt,                                    // step_value
                                "Must be greater than range_limits.y_min.");     // additional_constraints
  this->declareAndLoadParameter("range_limits.z_min", range_limits_z_min_,       // name
                                "Minimum z coordinate in target_frame to keep [m]",  // description
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinRangeZ,                                      // from_value
                                kMaxRangeZ,                                      // to_value
                                std::nullopt,                                    // step_value
                                "Must be less than range_limits.z_max.");        // additional_constraints
  this->declareAndLoadParameter("range_limits.z_max", range_limits_z_max_,       // name
                                "Maximum z coordinate in target_frame to keep [m]",  // description
                                true,                                            // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                false,                                           // read_only
                                kMinRangeZ,                                      // from_value
                                kMaxRangeZ,                                      // to_value
                                std::nullopt,                                    // step_value
                                "Must be greater than range_limits.z_min.");     // additional_constraints
  validateRangeLimits();
  this->declareAndLoadParameter("max_time_diff_sec", max_time_diff_sec_,         // name
                                "Maximum timestamp spread across a synchronized input batch in seconds",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                0.0, std::nullopt, std::nullopt,                 // from_value, to_value, step_value
                                "Must be non-negative");                         // additional_constraints
  this->declareAndLoadParameter("age_penalty", age_penalty_,                     // name
                                "Age penalty used by the approximate-time synchronizer",
                                false,                                           // add_to_auto_reconfigurable_params
                                false,                                           // is_required
                                true,                                            // read_only
                                0.0,                                             // from_value
                                100.0,                                           // to_value
                                std::nullopt,                                    // step_value
                                "Valid range is [0, 100].");                     // additional_constraints
  configureOutputStampMode(output_stamp_mode_param_);

#ifdef ENABLE_CUDA
  // Keep the CUDA context available even when starting in CPU mode so the
  // backend can be switched safely at runtime.
  try {
    cuda_context_ = std::make_unique<cuda::CudaTransformContext>();
    if (use_cuda_) {
      RCLCPP_INFO(this->get_logger(), "CUDA acceleration enabled");
    } else {
      RCLCPP_INFO(this->get_logger(), "CUDA context initialized; using CPU backend by parameter");
    }
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA context: %s", e.what());
    RCLCPP_WARN(this->get_logger(), "Falling back to CPU-only implementation");
    cuda_context_.reset();
    use_cuda_ = false;
    this->set_parameter(rclcpp::Parameter("use_cuda", false));
  }
#else
  RCLCPP_INFO(this->get_logger(), "CUDA support not compiled, using CPU-only implementation");
  use_cuda_ = false;
  this->set_parameter(rclcpp::Parameter("use_cuda", false));
#endif

  // run setup after constructor has finished to enable shared_from_this()
  setup_timer_ = this->create_wall_timer(std::chrono::milliseconds(1), [this]() {
    setup();
    setup_timer_->cancel();
  });
}
// clang-format on

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
      RCLCPP_WARN(this->get_logger(),
                  "Parameter type of parameter '%s' does not support "
                  "specifying a range",
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
    std::function<void(const rclcpp::Parameter&)> setter = [&param](const rclcpp::Parameter& p) { param = p.get_value<T>(); };
    auto_reconfigurable_params_.push_back(std::make_tuple(name, setter));
  }
}

rcl_interfaces::msg::SetParametersResult PointCloudFusion::parametersCallback(const std::vector<rclcpp::Parameter>& parameters) {
  std::unique_lock<std::shared_mutex> config_lock(config_mutex_);

  // Pre-validate interdependent and runtime-sensitive parameters before
  // applying any changes.
  // Build the prospective state: current values overridden by incoming changes.
  bool any_range_param = false;
  bool prospective_use_cuda = use_cuda_;
  int64_t prospective_fixed_points_per_input_cloud = fixed_points_per_input_cloud_;
  double prospective_x_min = range_limits_x_min_;
  double prospective_x_max = range_limits_x_max_;
  double prospective_y_min = range_limits_y_min_;
  double prospective_y_max = range_limits_y_max_;
  double prospective_z_min = range_limits_z_min_;
  double prospective_z_max = range_limits_z_max_;

  for (const auto& param : parameters) {
    const auto& name = param.get_name();
    if (name == "fixed_points_per_input_cloud") {
      prospective_fixed_points_per_input_cloud = param.as_int();
    } else if (name == "use_cuda") {
      prospective_use_cuda = param.as_bool();
    } else if (name == "range_limits.x_min") {
      prospective_x_min = param.as_double();
      any_range_param = true;
    } else if (name == "range_limits.x_max") {
      prospective_x_max = param.as_double();
      any_range_param = true;
    } else if (name == "range_limits.y_min") {
      prospective_y_min = param.as_double();
      any_range_param = true;
    } else if (name == "range_limits.y_max") {
      prospective_y_max = param.as_double();
      any_range_param = true;
    } else if (name == "range_limits.z_min") {
      prospective_z_min = param.as_double();
      any_range_param = true;
    } else if (name == "range_limits.z_max") {
      prospective_z_max = param.as_double();
      any_range_param = true;
    }
  }

  if (prospective_fixed_points_per_input_cloud < kMinFixedPointsPerInputCloud ||
      prospective_fixed_points_per_input_cloud > kMaxFixedPointsPerInputCloud) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    result.reason = "fixed_points_per_input_cloud must be in [" + std::to_string(kMinFixedPointsPerInputCloud) + ", " +
                    std::to_string(kMaxFixedPointsPerInputCloud) + "]";
    RCLCPP_ERROR(this->get_logger(), "Rejecting parameter update: %s", result.reason.c_str());
    return result;
  }

  if (prospective_use_cuda) {
#ifdef ENABLE_CUDA
    if (!cuda_context_) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = false;
      result.reason =
          "CUDA backend is unavailable because CUDA context "
          "initialization failed";
      RCLCPP_ERROR(this->get_logger(), "Rejecting parameter update: %s", result.reason.c_str());
      return result;
    }
#else
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = false;
    result.reason = "CUDA backend is unavailable because CUDA support was not compiled";
    RCLCPP_ERROR(this->get_logger(), "Rejecting parameter update: %s", result.reason.c_str());
    return result;
#endif
  }

  if (any_range_param) {
    rcl_interfaces::msg::SetParametersResult result;
    std::string reason;
    if (prospective_x_min >= prospective_x_max) {
      reason += "range_limits.x_min (" + std::to_string(prospective_x_min) + ") must be less than range_limits.x_max (" +
                std::to_string(prospective_x_max) + "). ";
    }
    if (prospective_y_min >= prospective_y_max) {
      reason += "range_limits.y_min (" + std::to_string(prospective_y_min) + ") must be less than range_limits.y_max (" +
                std::to_string(prospective_y_max) + "). ";
    }
    if (prospective_z_min >= prospective_z_max) {
      reason += "range_limits.z_min (" + std::to_string(prospective_z_min) + ") must be less than range_limits.z_max (" +
                std::to_string(prospective_z_max) + "). ";
    }
    if (!reason.empty()) {
      result.successful = false;
      result.reason = reason;
      RCLCPP_ERROR(this->get_logger(), "Rejecting range_limits parameter update: %s", reason.c_str());
      return result;
    }
  }

  // All validations passed — apply changes.
  const bool previous_use_cuda = use_cuda_;
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
  if (use_cuda_ != previous_use_cuda) {
    RCLCPP_INFO(this->get_logger(), "Fusion backend switched to %s", use_cuda_ ? "CUDA" : "CPU");
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
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<3> {
  using Policy = message_filters::sync_policies::
      ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<4> {
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<5> {
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<6> {
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<7> {
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<8> {
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2>;
};

template <>
struct SyncPolicyTraits<9> {
  using Policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2,
                                                                 sensor_msgs::msg::PointCloud2>;
};

template <std::size_t N>
using SyncPolicy = typename SyncPolicyTraits<N>::Policy;

template <std::size_t N>
using SyncType = message_filters::Synchronizer<SyncPolicy<N>>;

/**
 * @brief Connect subscriber filters to the synchronizer through an index
 * sequence.
 *
 * @tparam N Number of synchronizer inputs.
 * @tparam Is Subscriber indices expanded into connectInput.
 * @param sync Synchronizer receiving the subscriber filters.
 * @param subs Subscriber filters to connect.
 */
template <std::size_t N, std::size_t... Is>
void connectInputsImpl(SyncType<N>& sync,
                       const std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>>& subs,
                       std::index_sequence<Is...> /*unused*/) {
  // Fan the subscriber filters into the synchronizer inputs.
  sync.connectInput(*subs[Is]...);
}

/**
 * @brief Connect all subscriber filters to the synchronizer for N inputs.
 *
 * @tparam N Number of synchronizer inputs.
 * @param sync Synchronizer receiving the subscriber filters.
 * @param subs Subscriber filters to connect.
 */
template <std::size_t N>
void connectInputs(SyncType<N>& sync, const std::vector<std::shared_ptr<point_cloud_transport::SubscriberFilter>>& subs) {
  connectInputsImpl<N>(sync, subs, std::make_index_sequence<N>{});
}

}  // namespace detail

void PointCloudFusion::setup() {
  // callback for dynamic parameter configuration
  parameters_callback_ =
      this->add_on_set_parameters_callback(std::bind(&PointCloudFusion::parametersCallback, this, std::placeholders::_1));

  // create transform buffer and listener
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // validate inputs
  if (!input_transport_hints_.empty() && input_transport_hints_.size() != input_topics_.size()) {
    RCLCPP_WARN(this->get_logger(),
                "'input_transport_hints' length (%zu) does not match "
                "'input_topics' (%zu). Missing hints default to "
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

  RCLCPP_INFO(this->get_logger(), "Configured %zu input subscriber callback groups for %zu input topics",
              cloud_subscriber_callback_groups_.size(), input_topics_.size());

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
  std::string point_cloud_topic_name = this->get_node_topics_interface()->resolve_topic_name("~/point_cloud");
  cloud_publisher_ = std::make_shared<point_cloud_transport::Publisher>(
      pct.advertise(point_cloud_topic_name, static_cast<uint32_t>(output_queue_size_)));
  RCLCPP_INFO(this->get_logger(), "Publishing to '%s'", cloud_publisher_->getTopic().c_str());

  // Annotate message links for tracing: Each publisher (for raw and compressed
  // point clouds) depends an all input point clouds.
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
  TRACETOOLS_TRACEPOINT(message_link_partial_sync, link_subs.data(), link_subs.size(), link_pubs.data(), link_pubs.size());
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
      RCLCPP_WARN(this->get_logger(),
                  "ApproximateTime synchronizer yielded no "
                  "valid point clouds; skipping fusion.");
    }
  });

  synchronizer_ = sync;

  RCLCPP_INFO(this->get_logger(),
              "Configured approximate time synchronizer for %zu inputs "
              "(queue=%zu, max_dt=%.3f s, age_penalty=%.6f)",
              static_cast<size_t>(N), static_cast<size_t>(sync_queue_size_), max_time_diff_sec_, age_penalty_);
}

void PointCloudFusion::handleSynchronizedPointClouds(const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>& msgs) {
  if (msgs.empty()) {
    return;
  }

  // Protect runtime-configurable parameter reads against concurrent parameter
  // updates.
  std::shared_lock<std::shared_mutex> config_lock(config_mutex_);

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
    // Guard shared CUDA pipeline state against concurrent synchronized
    // callbacks.
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
  const auto cpu_processing_start = std::chrono::steady_clock::now();
  auto fused_point_cloud = fusePointCloudBatch(msgs, timing, valid_count);

  if (!fused_point_cloud) {
    RCLCPP_WARN(this->get_logger(), "All points are invalid, skipping fusion");
    return;
  }

  const auto cpu_processing_end = std::chrono::steady_clock::now();
  publishFusedCloud(std::move(fused_point_cloud), timing, msgs.size(), valid_count, callback_start, cpu_processing_start,
                    cpu_processing_end, "cpu_fusion_complete");
}

bool PointCloudFusion::collectTimingInfo(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, FusionTiming& timing) const {
  if (msgs.empty()) {
    return false;
  }

  const auto input0_stamp = rclcpp::Time(msgs.front()->header.stamp);
  bool first_stamp = true;
  double max_dt_from_input0_sec = 0.0;
  rclcpp::Time earliest_stamp;
  rclcpp::Time latest_stamp;

  // Walk every cloud once to gather min/max stamps and the largest skew from
  // input0.
  for (const auto& pc_msg : msgs) {
    if (!pc_msg) {
      RCLCPP_WARN(this->get_logger(),
                  "Received null point cloud pointer in "
                  "synchronized batch, skipping fusion");
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

rclcpp::Time PointCloudFusion::outputStamp(const FusionTiming& timing) const {
  switch (output_stamp_mode_) {
    case OutputStampMode::Earliest:
      return timing.earliest_stamp;
    case OutputStampMode::Mean: {
      const auto delta = timing.latest_stamp - timing.earliest_stamp;
      return timing.earliest_stamp + rclcpp::Duration::from_nanoseconds(delta.nanoseconds() / 2);
    }
    case OutputStampMode::Input0:
      return timing.input0_stamp;
    case OutputStampMode::Latest:
    default:
      return timing.latest_stamp;
  }
}

bool PointCloudFusion::prepareMotionTransform(const PointCloudMsg& msg,
                                              const rclcpp::Time& reference_stamp,
                                              int time_field_offset,
                                              double timeout_sec,
                                              MotionTransform& transform) const {
  transform = MotionTransform{};

  const std::size_t total_points = static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
  if (time_field_offset >= 0 && total_points > 0) {
    const auto offset = static_cast<std::size_t>(time_field_offset);
    if (offset + sizeof(uint32_t) <= msg.point_step && msg.data.size() >= total_points * msg.point_step) {
      const auto* data = msg.data.data();
      uint32_t max_time_offset = 0;
      for (std::size_t idx = 0; idx < total_points; ++idx) {
        max_time_offset = std::max(max_time_offset, loadUint32(byteOffset(data, idx * msg.point_step), offset));
      }
      transform.max_time_offset = max_time_offset;
    }
  }

  const rclcpp::Time scan_start(msg.header.stamp);
  const auto scan_duration =
      rclcpp::Duration::from_seconds(static_cast<double>(transform.max_time_offset) * motion_compensation_time_scale_sec_);
  const rclcpp::Time scan_end = scan_start + scan_duration;
  const auto timeout = rclcpp::Duration::from_seconds(timeout_sec);

  auto lookup_at = [&](const rclcpp::Time& source_stamp) {
    return tf_buffer_->lookupTransform(target_frame_, reference_stamp, msg.header.frame_id, source_stamp,
                                       motion_compensation_fixed_frame_, timeout);
  };

  geometry_msgs::msg::TransformStamped start_stamped;
  geometry_msgs::msg::TransformStamped end_stamped;
  try {
    start_stamped = lookup_at(scan_start);
    end_stamped = transform.max_time_offset > 0 ? lookup_at(scan_end) : start_stamped;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Motion compensation unavailable for '%s' through "
                         "fixed frame '%s': %s; "
                         "falling back to rigid transforms for this batch",
                         msg.header.frame_id.c_str(), motion_compensation_fixed_frame_.c_str(), ex.what());
    return false;
  }

  auto copy_transform = [](const geometry_msgs::msg::Transform& source, std::array<float, 3>& translation,
                           std::array<float, 4>& quaternion) {
    translation[0] = static_cast<float>(source.translation.x);
    translation[1] = static_cast<float>(source.translation.y);
    translation[2] = static_cast<float>(source.translation.z);
    quaternion[0] = static_cast<float>(source.rotation.x);
    quaternion[1] = static_cast<float>(source.rotation.y);
    quaternion[2] = static_cast<float>(source.rotation.z);
    quaternion[3] = static_cast<float>(source.rotation.w);
  };
  copy_transform(start_stamped.transform, transform.start_translation, transform.start_quaternion);
  copy_transform(end_stamped.transform, transform.end_translation, transform.end_quaternion);

  const float quaternion_dot =
      transform.start_quaternion[0] * transform.end_quaternion[0] + transform.start_quaternion[1] * transform.end_quaternion[1] +
      transform.start_quaternion[2] * transform.end_quaternion[2] + transform.start_quaternion[3] * transform.end_quaternion[3];
  if (quaternion_dot < 0.0F) {
    for (float& component : transform.end_quaternion) {
      component = -component;
    }
  }
  return true;
}

bool PointCloudFusion::prepareBatchMotionTransforms(const std::vector<PointCloudMsg::ConstSharedPtr>& msgs,
                                                    const rclcpp::Time& reference_stamp,
                                                    const std::vector<int>& time_field_offsets,
                                                    std::vector<MotionTransform>& transforms) const {
  transforms.clear();
  transforms.resize(msgs.size());
  if (!motion_compensation_enable_ || motion_compensation_fixed_frame_.empty()) {
    return false;
  }

  if (time_field_offsets.size() != msgs.size() ||
      std::any_of(time_field_offsets.begin(), time_field_offsets.end(), [](int offset) { return offset == -1; })) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Motion compensation requested, but at least one input lacks a valid UINT32 '%s' field; "
                         "using rigid transforms for this heterogeneous batch",
                         motion_compensation_time_field_.c_str());
    return false;
  }

  const bool recovery_probe = !motion_tf_available_.load(std::memory_order_relaxed);
  const double timeout_sec = recovery_probe ? 0.0 : motion_compensation_tf_timeout_sec_;
  bool prepared_any = false;
  for (std::size_t i = 0; i < msgs.size(); ++i) {
    if (!msgs[i] || time_field_offsets[i] == -2) {
      continue;
    }
    prepared_any = true;
    if (!prepareMotionTransform(*msgs[i], reference_stamp, time_field_offsets[i], timeout_sec, transforms[i])) {
      motion_tf_available_.store(false, std::memory_order_relaxed);
      return false;
    }
  }

  if (prepared_any && !motion_tf_available_.exchange(true, std::memory_order_relaxed)) {
    RCLCPP_INFO(this->get_logger(),
                "Motion-compensation TF is available "
                "again; resuming compensated fusion");
  }
  return prepared_any;
}

PointCloudFusion::PointCloudMsg::UniquePtr PointCloudFusion::fusePointCloudBatch(
    const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, const FusionTiming& timing, std::size_t& valid_point_count) const {
  valid_point_count = 0;
  if (msgs.empty()) return nullptr;

  const detail::BatchLayout layout = detail::buildBatchLayout(msgs, output_fields_, motion_compensation_time_field_);
  if (layout.point_step == 0) {
    missing_xyz_count_.fetch_add(msgs.size(), std::memory_order_relaxed);
    RCLCPP_WARN(this->get_logger(), "No input cloud has valid FLOAT32 x/y/z fields; skipping batch");
    return nullptr;
  }

  if (!layout.conflicting_fields.empty()) {
    incompatible_field_count_.fetch_add(layout.conflicting_fields.size(), std::memory_order_relaxed);
    std::ostringstream names;
    for (std::size_t i = 0; i < layout.conflicting_fields.size(); ++i) {
      if (i != 0U) names << ", ";
      names << layout.conflicting_fields[i];
    }
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Omitting same-name fields with incompatible datatype/count: %s", names.str().c_str());
  }

  std::size_t max_capacity = 0;
  std::vector<int> time_offsets(msgs.size(), -2);
  for (std::size_t i = 0; i < msgs.size(); ++i) {
    if (!layout.inputs[i].valid || !msgs[i]) {
      if (msgs[i]) {
        missing_xyz_count_.fetch_add(1, std::memory_order_relaxed);
        skipped_cloud_count_.fetch_add(1, std::memory_order_relaxed);
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Skipping cloud from '%s': %s (missing_xyz_total=%llu, skipped_total=%llu)",
                             msgs[i]->header.frame_id.c_str(), layout.inputs[i].rejection_reason.c_str(),
                             static_cast<unsigned long long>(missing_xyz_count_.load()),
                             static_cast<unsigned long long>(skipped_cloud_count_.load()));
      }
      continue;
    }
    time_offsets[i] = layout.inputs[i].time_offset;
    std::size_t count = static_cast<std::size_t>(msgs[i]->width) * msgs[i]->height;
    count = std::min(count, msgs[i]->point_step == 0 ? std::size_t{0} : msgs[i]->data.size() / msgs[i]->point_step);
    if (fixed_points_per_input_cloud_ > 0) {
      count = std::min(count, static_cast<std::size_t>(fixed_points_per_input_cloud_));
    }
    max_capacity += count;
    if (!layout.inputs[i].zero_filled_fields.empty()) {
      zero_filled_field_count_.fetch_add(layout.inputs[i].zero_filled_fields.size(), std::memory_order_relaxed);
      RCLCPP_DEBUG(this->get_logger(), "Cloud from '%s' zero-fills %zu output fields (total=%llu)",
                   msgs[i]->header.frame_id.c_str(), layout.inputs[i].zero_filled_fields.size(),
                   static_cast<unsigned long long>(zero_filled_field_count_.load()));
    }
  }
  if (max_capacity == 0) return nullptr;

  auto output = std::make_unique<PointCloudMsg>();
  output->header.frame_id = target_frame_;
  const rclcpp::Time chosen_stamp = outputStamp(timing);
  output->header.stamp = chosen_stamp;
  output->height = 1;
  output->is_bigendian = layout.is_bigendian;
  output->point_step = layout.point_step;
  output->fields = layout.fields;
  output->is_dense = true;
  output->data.resize(max_capacity * layout.point_step);

  const bool check_range = range_limits_enable_;
  const float x_min = static_cast<float>(range_limits_x_min_);
  const float x_max = static_cast<float>(range_limits_x_max_);
  const float y_min = static_cast<float>(range_limits_y_min_);
  const float y_max = static_cast<float>(range_limits_y_max_);
  const float z_min = static_cast<float>(range_limits_z_min_);
  const float z_max = static_cast<float>(range_limits_z_max_);

  std::vector<MotionTransform> motion_transforms;
  const bool batch_motion = prepareBatchMotionTransforms(msgs, chosen_stamp, time_offsets, motion_transforms);
  uint8_t* destination = output->data.data();

  for (std::size_t input_index = 0; input_index < msgs.size(); ++input_index) {
    const auto& msg = msgs[input_index];
    const auto& input = layout.inputs[input_index];
    if (!msg || !input.valid || msg->point_step == 0) continue;

    const bool apply_transform = batch_motion || msg->header.frame_id != target_frame_;
    tf2::Vector3 translation(0.0, 0.0, 0.0);
    tf2::Matrix3x3 rotation = tf2::Matrix3x3::getIdentity();
    if (!batch_motion && apply_transform) {
      try {
        const auto stamped = tf_buffer_->lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp,
                                                         rclcpp::Duration::from_seconds(0.1));
        tf2::Transform transform;
        tf2::fromMsg(stamped.transform, transform);
        translation = transform.getOrigin();
        rotation = transform.getBasis();
      } catch (const tf2::TransformException& exception) {
        skipped_cloud_count_.fetch_add(1, std::memory_order_relaxed);
        RCLCPP_ERROR(this->get_logger(), "Cannot transform point cloud from %s to %s: %s (skipped_total=%llu)",
                     msg->header.frame_id.c_str(), target_frame_.c_str(), exception.what(),
                     static_cast<unsigned long long>(skipped_cloud_count_.load()));
        continue;
      }
    }

    const std::size_t declared_points = static_cast<std::size_t>(msg->width) * msg->height;
    const std::size_t total_points = std::min(declared_points, msg->data.size() / msg->point_step);
    const std::size_t samples = fixed_points_per_input_cloud_ > 0
                                    ? std::min(total_points, static_cast<std::size_t>(fixed_points_per_input_cloud_))
                                    : total_points;
    if (samples == 0) continue;
    const double stride = static_cast<double>(total_points) / static_cast<double>(samples);

    for (std::size_t sample = 0; sample < samples; ++sample) {
      const std::size_t point_index =
          samples == total_points ? sample
                                  : std::min(static_cast<std::size_t>(static_cast<double>(sample) * stride), total_points - 1);
      const uint8_t* source = byteOffset(msg->data.data(), point_index * msg->point_step);
      const float x = loadFloat(source, input.x_offset);
      const float y = loadFloat(source, input.y_offset);
      const float z = loadFloat(source, input.z_offset);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

      float transformed_x = x;
      float transformed_y = y;
      float transformed_z = z;
      if (batch_motion) {
        const uint32_t point_time = loadUint32(source, static_cast<std::size_t>(input.time_offset));
        transformPointInterpolated(motion_transforms[input_index], point_time, x, y, z, transformed_x, transformed_y,
                                   transformed_z);
      } else if (apply_transform) {
        const tf2::Vector3 transformed = rotation * tf2::Vector3(x, y, z) + translation;
        transformed_x = static_cast<float>(transformed.x());
        transformed_y = static_cast<float>(transformed.y());
        transformed_z = static_cast<float>(transformed.z());
      }
      if (check_range &&
          !pointWithinRange(transformed_x, transformed_y, transformed_z, x_min, x_max, y_min, y_max, z_min, z_max)) {
        continue;
      }

      if (input.whole_point_copy) {
        std::memcpy(destination, source, layout.point_step);
      } else {
        std::memset(destination, 0, layout.point_step);
        for (const auto& copy : input.copies) {
          std::memcpy(byteOffset(destination, copy.destination_offset), byteOffset(source, copy.source_offset), copy.byte_length);
        }
      }
      storeFloat(destination, layout.x_offset, transformed_x);
      storeFloat(destination, layout.y_offset, transformed_y);
      storeFloat(destination, layout.z_offset, transformed_z);
      destination = byteOffset(destination, layout.point_step);
      ++valid_point_count;
    }
  }

  if (valid_point_count == 0) return nullptr;
  output->width = valid_point_count;
  output->row_step = output->point_step * output->width;
  output->data.resize(valid_point_count * output->point_step);
  return output;
}

void PointCloudFusion::publishFusedCloud(PointCloudMsg::UniquePtr cloud,
                                         const FusionTiming& timing,
                                         std::size_t input_count,
                                         std::size_t total_points,
                                         std::chrono::steady_clock::time_point callback_start,
                                         std::chrono::steady_clock::time_point processing_start,
                                         std::chrono::steady_clock::time_point processing_end,
                                         const char* event_name) {
  // Publish the fused cloud and emit a compact timing summary for
  // observability.
  cloud_publisher_->publish(std::move(cloud));
  const auto publish_end = std::chrono::steady_clock::now();

  const double prep_duration_ms = std::chrono::duration<double, std::milli>(processing_start - callback_start).count();
  const double processing_duration_ms = std::chrono::duration<double, std::milli>(processing_end - processing_start).count();
  const double publish_duration_ms = std::chrono::duration<double, std::milli>(publish_end - processing_end).count();
  const double e2e_duration_ms = prep_duration_ms + processing_duration_ms + publish_duration_ms;
  const double batch_dt_ms = (timing.latest_stamp - timing.earliest_stamp).seconds() * 1000.0;

  RCLCPP_DEBUG(this->get_logger(),
               "%s inputs=%zu points=%zu e2e_ms=%.3f prep_ms=%.3f "
               "process_ms=%.3f publish_ms=%.3f batch_dt_ms=%.3f "
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
    RCLCPP_FATAL(this->get_logger(),
                 "Configured with %zu input topics, but only up to %zu inputs "
                 "are supported",
                 input_topics_.size(), kMaxInputTopics);
    exit(EXIT_FAILURE);
  }
}

void PointCloudFusion::validateRangeLimits() {
  bool valid = true;
  if (range_limits_x_min_ >= range_limits_x_max_) {
    RCLCPP_ERROR(this->get_logger(),
                 "range_limits.x_min (%.3f) must be less than "
                 "range_limits.x_max (%.3f); disabling range filtering",
                 range_limits_x_min_, range_limits_x_max_);
    valid = false;
  }
  if (range_limits_y_min_ >= range_limits_y_max_) {
    RCLCPP_ERROR(this->get_logger(),
                 "range_limits.y_min (%.3f) must be less than "
                 "range_limits.y_max (%.3f); disabling range filtering",
                 range_limits_y_min_, range_limits_y_max_);
    valid = false;
  }
  if (range_limits_z_min_ >= range_limits_z_max_) {
    RCLCPP_ERROR(this->get_logger(),
                 "range_limits.z_min (%.3f) must be less than "
                 "range_limits.z_max (%.3f); disabling range filtering",
                 range_limits_z_min_, range_limits_z_max_);
    valid = false;
  }
  if (!valid) {
    range_limits_enable_ = false;
  }
}

#ifdef ENABLE_CUDA

PointCloudFusion::PointCloudMsg::UniquePtr PointCloudFusion::fusePointCloudBatchCUDA(
    const std::vector<PointCloudMsg::ConstSharedPtr>& msgs, const FusionTiming& timing, std::size_t& valid_point_count) const {
  valid_point_count = 0;
  if (msgs.empty() || !cuda_context_) return nullptr;

  const detail::BatchLayout layout = detail::buildBatchLayout(msgs, output_fields_, motion_compensation_time_field_);
  if (layout.point_step == 0) {
    RCLCPP_WARN(this->get_logger(), "CUDA: no input cloud has valid FLOAT32 x/y/z fields; skipping batch");
    return nullptr;
  }
  for (const auto& name : layout.conflicting_fields) {
    incompatible_field_count_.fetch_add(1, std::memory_order_relaxed);
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "CUDA: omitting incompatible same-name field '%s'",
                         name.c_str());
  }

  // Heterogeneous clouds are packed into the common output layout on the host.
  // The existing coalesced CUDA transform/filter kernel then sees identical
  // slots. Homogeneous all-field batches retain the original zero-copy host path.
  std::vector<std::vector<uint8_t>> packed(msgs.size());
  std::vector<const uint8_t*> input_data(msgs.size(), nullptr);
  std::vector<std::size_t> point_counts(msgs.size(), 0);
  std::vector<int> source_time_offsets(msgs.size(), -2);
  std::size_t max_points = 0;
  for (std::size_t i = 0; i < msgs.size(); ++i) {
    const auto& msg = msgs[i];
    const auto& input = layout.inputs[i];
    if (!msg || !input.valid || msg->point_step == 0) {
      if (msg) {
        missing_xyz_count_.fetch_add(1, std::memory_order_relaxed);
        skipped_cloud_count_.fetch_add(1, std::memory_order_relaxed);
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "CUDA: skipping cloud from '%s': %s",
                             msg->header.frame_id.c_str(), input.rejection_reason.c_str());
      }
      continue;
    }
    source_time_offsets[i] = input.time_offset;
    point_counts[i] = std::min(static_cast<std::size_t>(msg->width) * msg->height, msg->data.size() / msg->point_step);
    max_points = std::max(max_points, point_counts[i]);
    if (input.whole_point_copy) {
      input_data[i] = msg->data.data();
      continue;
    }
    packed[i].assign(point_counts[i] * layout.point_step, 0);
    for (std::size_t point = 0; point < point_counts[i]; ++point) {
      const uint8_t* source = byteOffset(msg->data.data(), point * msg->point_step);
      uint8_t* destination = byteOffset(packed[i].data(), point * layout.point_step);
      for (const auto& copy : input.copies) {
        std::memcpy(byteOffset(destination, copy.destination_offset), byteOffset(source, copy.source_offset), copy.byte_length);
      }
    }
    input_data[i] = packed[i].data();
    if (!input.zero_filled_fields.empty()) {
      zero_filled_field_count_.fetch_add(input.zero_filled_fields.size(), std::memory_order_relaxed);
    }
  }
  if (max_points == 0) return nullptr;

  int packed_time_offset = -1;
  for (const auto& field : layout.fields) {
    if (field.name == motion_compensation_time_field_ && field.datatype == sensor_msgs::msg::PointField::UINT32 &&
        field.count == 1) {
      packed_time_offset = static_cast<int>(field.offset);
      break;
    }
  }
  if (packed_time_offset < 0) {
    for (std::size_t i = 0; i < source_time_offsets.size(); ++i) {
      if (source_time_offsets[i] >= 0) source_time_offsets[i] = -1;
    }
  }

  const rclcpp::Time chosen_stamp = outputStamp(timing);
  std::vector<MotionTransform> motion_transforms;
  const bool batch_motion = prepareBatchMotionTransforms(msgs, chosen_stamp, source_time_offsets, motion_transforms);
  const std::size_t total_capacity = max_points * msgs.size();
  std::vector<cuda::CudaFieldCopy> copy_plan{{0, 0, static_cast<int>(layout.point_step)}};
  if (!cuda_context_->resetBatch(
          total_capacity, max_points, layout.point_step, layout.point_step, static_cast<int>(layout.x_offset),
          static_cast<int>(layout.y_offset), static_cast<int>(layout.z_offset), static_cast<int>(layout.x_offset),
          static_cast<int>(layout.y_offset), static_cast<int>(layout.z_offset), copy_plan,
          static_cast<float>(range_limits_x_min_), static_cast<float>(range_limits_x_max_),
          static_cast<float>(range_limits_y_min_), static_cast<float>(range_limits_y_max_),
          static_cast<float>(range_limits_z_min_), static_cast<float>(range_limits_z_max_), range_limits_enable_)) {
    RCLCPP_ERROR(this->get_logger(), "CUDA resetBatch failed for heterogeneous layout");
    return nullptr;
  }

  for (std::size_t i = 0; i < msgs.size(); ++i) {
    if (!input_data[i] || point_counts[i] == 0) continue;
    const auto& msg = msgs[i];
    const bool apply_transform = batch_motion || msg->header.frame_id != target_frame_;
    float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float translation[3] = {0, 0, 0};
    if (!batch_motion && apply_transform) {
      try {
        const auto stamped = tf_buffer_->lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp,
                                                         rclcpp::Duration::from_seconds(0.1));
        tf2::Transform transform;
        tf2::fromMsg(stamped.transform, transform);
        const auto basis = transform.getBasis();
        const auto origin = transform.getOrigin();
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) rotation[row * 3 + column] = basis[row][column];
        }
        translation[0] = origin.x();
        translation[1] = origin.y();
        translation[2] = origin.z();
      } catch (const tf2::TransformException& exception) {
        skipped_cloud_count_.fetch_add(1, std::memory_order_relaxed);
        RCLCPP_ERROR(this->get_logger(), "CUDA: cannot transform %s to %s: %s", msg->header.frame_id.c_str(),
                     target_frame_.c_str(), exception.what());
        continue;
      }
    }
    const int desired = fixed_points_per_input_cloud_ > 0 ? static_cast<int>(fixed_points_per_input_cloud_) : 0;
    const auto& motion = motion_transforms[i];
    if (!cuda_context_->addCloud(input_data[i], point_counts[i], rotation, translation, apply_transform, i, desired, batch_motion,
                                 packed_time_offset, motion.max_time_offset, motion.start_translation.data(),
                                 motion.end_translation.data(), motion.start_quaternion.data(), motion.end_quaternion.data())) {
      RCLCPP_ERROR(this->get_logger(), "CUDA addCloud failed for input %zu", i);
    }
  }

  auto output = std::make_unique<PointCloudMsg>();
  output->header.frame_id = target_frame_;
  output->header.stamp = chosen_stamp;
  output->height = 1;
  output->is_bigendian = layout.is_bigendian;
  output->point_step = layout.point_step;
  output->fields = layout.fields;
  output->is_dense = true;
  if (!cuda_context_->getBatchOutput(output->data, valid_point_count) || valid_point_count == 0) return nullptr;
  output->width = valid_point_count;
  output->row_step = output->point_step * output->width;
  return output;
}
#endif  // ENABLE_CUDA

}  // namespace point_cloud_fusion
