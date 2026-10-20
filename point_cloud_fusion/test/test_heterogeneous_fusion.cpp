// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <rcl/time.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <rcl_interfaces/msg/log.hpp>

#include <point_cloud_fusion/point_cloud_fusion.hpp>
#include <point_cloud_fusion/point_cloud_layout.hpp>

namespace {

using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;
using namespace std::chrono_literals;

Field field(const std::string& name, uint32_t offset, uint8_t datatype = Field::FLOAT32, uint32_t count = 1) {
  return Field().set__name(name).set__offset(offset).set__datatype(datatype).set__count(count);
}

template <typename T>
void write(Cloud& cloud, std::size_t point, uint32_t offset, const T& value) {
  std::memcpy(cloud.data.data() + point * cloud.point_step + offset, &value, sizeof(T));
}

template <typename T>
T read(const Cloud& cloud, std::size_t point, uint32_t offset) {
  T value;
  std::memcpy(&value, cloud.data.data() + point * cloud.point_step + offset, sizeof(T));
  return value;
}

// Synthetic layouts deliberately differ in field order, offsets, padding and
// point count. They model heterogeneous sensors, not a specific driver version.
Cloud syntheticCloud(bool aeva) {
  Cloud cloud;
  cloud.header.frame_id = "base_link";
  cloud.header.stamp.sec = 100;
  cloud.height = 1;
  cloud.width = aeva ? 5 : 3;
  cloud.point_step = aeva ? 64 : 24;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.fields = aeva ? std::vector<Field>{field("velocity", 0),
                                           field("intensity", 4),
                                           field("z", 12),
                                           field("x", 20),
                                           field("y", 28),
                                           field("t", 32, Field::UINT32),
                                           field("velocity_vector", 36, Field::FLOAT32, 3),
                                           field("velocity_double", 48, Field::FLOAT64)}
                      : std::vector<Field>{field("x", 0),
                                           field("y", 4),
                                           field("z", 8),
                                           field("intensity", 12),
                                           field("t", 16, Field::UINT32),
                                           field("ring", 20, Field::UINT16)};
  // Nonzero padding makes accidental whole-record copies easier to detect.
  cloud.data.assign(cloud.row_step, 0xA5);
  const std::array<float, 5> velocities{-7.5F, 0.0F, 12.25F, std::numeric_limits<float>::quiet_NaN(), 999.0F};
  for (std::size_t i = 0; i < cloud.width; ++i) {
    const float x =
        aeva ? (i == 4 ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(i + 1)) : static_cast<float>(i + 11);
    write(cloud, i, aeva ? 20 : 0, x);
    write(cloud, i, aeva ? 28 : 4, 0.5F);
    write(cloud, i, aeva ? 12 : 8, 1.0F);
    write(cloud, i, aeva ? 4 : 12, static_cast<float>(i + (aeva ? 20 : 40)));
    write(cloud, i, aeva ? 32 : 16, static_cast<uint32_t>((i + 1) * 1000));
    if (aeva) {
      write(cloud, i, 0, velocities[i]);
      write(cloud, i, 36, std::array<float, 3>{velocities[i], -2.0F, 3.5F});
      write(cloud, i, 48, static_cast<double>(velocities[i]));
    } else {
      write(cloud, i, 20, static_cast<uint16_t>(i + 5));
    }
  }
  return cloud;
}

struct Configuration {
  bool cuda;
  bool aeva_first;
};

class HeterogeneousFusion : public testing::TestWithParam<Configuration> {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }

  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    static unsigned int test_index = 0;
    namespace_ = "/synthetic_" + std::to_string(test_index++);
    driver_ = std::make_shared<rclcpp::Node>("synthetic_lidars", namespace_);
    executor_->add_node(driver_);
    aeva_publisher_ = driver_->create_publisher<Cloud>(namespace_ + "/aeva", rclcpp::SensorDataQoS());
    ouster_publisher_ = driver_->create_publisher<Cloud>(namespace_ + "/ouster", rclcpp::QoS(10).reliable());
    output_subscription_ =
        driver_->create_subscription<Cloud>(namespace_ + "/point_cloud_fusion/point_cloud", rclcpp::SensorDataQoS(),
                                            [this](Cloud::ConstSharedPtr msg) { output_ = msg; });
  }

  void TearDown() override {
    executor_.reset();
    fusion_.reset();
    output_subscription_.reset();
    aeva_publisher_.reset();
    ouster_publisher_.reset();
    driver_.reset();
  }

  bool spinUntil(const std::function<bool()>& ready) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!ready() && std::chrono::steady_clock::now() < deadline) executor_->spin_once(10ms);
    return ready();
  }

  void start(std::vector<rclcpp::Parameter> parameters = {}) {
    std::vector<std::string> topics{namespace_ + "/ouster", namespace_ + "/aeva"};
    if (GetParam().aeva_first) std::reverse(topics.begin(), topics.end());
    parameters.emplace_back("target_frame", "base_link");
    parameters.emplace_back("input_topics", topics);
    parameters.emplace_back("use_cuda", GetParam().cuda);
    rclcpp::NodeOptions options;
    options.parameter_overrides(parameters).arguments({"--ros-args", "-r", "__ns:=" + namespace_});
    fusion_ = std::make_shared<point_cloud_fusion::PointCloudFusion>(options);
    if (GetParam().cuda && !fusion_->get_parameter("use_cuda").as_bool()) {
      GTEST_SKIP() << "CUDA backend unavailable; do not count CPU fallback as CUDA coverage";
    }
    executor_->add_node(fusion_);
    ASSERT_TRUE(spinUntil([this] {
      return aeva_publisher_->get_subscription_count() > 0 && ouster_publisher_->get_subscription_count() > 0 &&
             output_subscription_->get_publisher_count() > 0;
    })) << "Fusion topics did not become ready";
  }

  void fuse(Cloud aeva = syntheticCloud(true), Cloud ouster = syntheticCloud(false)) {
    output_.reset();
    const auto first_stamp = next_stamp_;
    aeva.header.stamp.sec = ouster.header.stamp.sec = next_stamp_;
    // Retry best-effort sensor messages while middleware endpoints finish
    // matching. Equal stamps let ApproximateTime emit without a later scan.
    auto next_publish = std::chrono::steady_clock::now();
    ASSERT_TRUE(spinUntil([&] {
      if (output_ && output_->header.stamp.sec >= first_stamp) return true;
      if (std::chrono::steady_clock::now() >= next_publish) {
        aeva_publisher_->publish(aeva);
        ouster_publisher_->publish(ouster);
        ++next_stamp_;
        aeva.header.stamp.sec = ouster.header.stamp.sec = next_stamp_;
        next_publish = std::chrono::steady_clock::now() + 50ms;
      }
      return false;
    })) << "No fused output received";
  }

  void checkOutput(std::vector<float> expected_x,
                   bool velocity_published = true,
                   float aeva_translation = 0.0F,
                   const Cloud& aeva = syntheticCloud(true),
                   const std::string& velocity_name = "velocity") {
    ASSERT_NE(output_, nullptr);
    EXPECT_EQ(output_->header.frame_id, "base_link");
    EXPECT_EQ(output_->height, 1U);
    ASSERT_EQ(output_->width, expected_x.size());
    EXPECT_EQ(output_->row_step, output_->point_step * output_->width);
    ASSERT_EQ(output_->data.size(), output_->row_step);
    if (velocity_published) {
      EXPECT_FALSE(output_->is_dense);
    }
    const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
    ASSERT_NE(x, nullptr);
    std::vector<float> actual_x;
    for (std::size_t i = 0; i < output_->width; ++i) {
      const float position = read<float>(*output_, i, x->offset);
      actual_x.push_back(position);
      const bool from_aeva = aeva_translation == 0.0F ? position < 10.0F : position > 100.0F;
      const auto aeva_index = from_aeva ? static_cast<std::size_t>(position - aeva_translation - 1.0F) : 0;
      for (const auto& name : {velocity_name, std::string("velocity_vector"), std::string("velocity_double")}) {
        const auto* destination = point_cloud_fusion::detail::findField(*output_, name);
        if (!velocity_published) {
          EXPECT_EQ(destination, nullptr);
          continue;
        }
        ASSERT_NE(destination, nullptr) << name;
        const auto* source = point_cloud_fusion::detail::findField(aeva, name);
        ASSERT_TRUE(point_cloud_fusion::detail::sameType(*source, *destination));
        const auto length = point_cloud_fusion::detail::fieldDatatypeSize(source->datatype) * source->count;
        const auto* bytes = output_->data.data() + i * output_->point_step + destination->offset;
        if (from_aeva) {
          ASSERT_LT(aeva_index, 4U);
          EXPECT_EQ(std::memcmp(bytes, aeva.data.data() + aeva_index * aeva.point_step + source->offset, length), 0)
              << name << " at x=" << position;
        } else if (destination->datatype == Field::FLOAT32 || destination->datatype == Field::FLOAT64) {
          for (uint32_t element = 0; element < destination->count; ++element) {
            const uint32_t offset =
                destination->offset + element * point_cloud_fusion::detail::fieldDatatypeSize(destination->datatype);
            if (destination->datatype == Field::FLOAT32) {
              EXPECT_TRUE(std::isnan(read<float>(*output_, i, offset)));
            } else {
              EXPECT_TRUE(std::isnan(read<double>(*output_, i, offset)));
            }
          }
        } else {
          EXPECT_TRUE(std::all_of(bytes, bytes + length, [](uint8_t value) { return value == 0; }))
              << name << " must be zero-filled for Ouster";
        }
      }
    }
    // CUDA compaction need not preserve point order.
    std::sort(actual_x.begin(), actual_x.end());
    std::sort(expected_x.begin(), expected_x.end());
    EXPECT_EQ(actual_x, expected_x);
  }

  int32_t next_stamp_{100};
  std::string namespace_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::Node::SharedPtr driver_;
  std::shared_ptr<point_cloud_fusion::PointCloudFusion> fusion_;
  rclcpp::Publisher<Cloud>::SharedPtr aeva_publisher_;
  rclcpp::Publisher<Cloud>::SharedPtr ouster_publisher_;
  rclcpp::Subscription<Cloud>::SharedPtr output_subscription_;
  Cloud::ConstSharedPtr output_;
};

TEST_P(HeterogeneousFusion, UnionRetainsSensorSpecificVelocityValues) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  fuse();
  checkOutput({1, 2, 3, 4, 11, 12, 13});
  ASSERT_NE(output_, nullptr);
  EXPECT_NE(point_cloud_fusion::detail::findField(*output_, "ring"), nullptr);
}

TEST_P(HeterogeneousFusion, RawSpcVelocityNameDefaultsToNaN) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = syntheticCloud(true);
  aeva.fields[0].name = "v";
  fuse(aeva);
  checkOutput({1, 2, 3, 4, 11, 12, 13}, true, 0.0F, aeva, "v");
}

TEST_P(HeterogeneousFusion, ArbitraryFieldNameDefaultsToNaN) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = syntheticCloud(true);
  aeva.fields[0].name = "custom_speed";
  fuse(aeva);
  checkOutput({1, 2, 3, 4, 11, 12, 13}, true, 0.0F, aeva, "custom_speed");
}

TEST_P(HeterogeneousFusion, PerFieldPoliciesSupportNaNConstantsIntegersArraysAndTime) {
  start({rclcpp::Parameter(
      "missing_field_fill_values",
      std::vector<std::string>{"velocity=0", "velocity_vector=nan", "velocity_double=-123.5", "ring=65535", "t=777"})});
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = syntheticCloud(true);
  aeva.fields.erase(std::remove_if(aeva.fields.begin(), aeva.fields.end(), [](const auto& f) { return f.name == "t"; }),
                    aeva.fields.end());
  fuse(aeva);
  ASSERT_NE(output_, nullptr);
  ASSERT_EQ(output_->width, 7U);
  EXPECT_FALSE(output_->is_dense);
  const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
  const auto* velocity = point_cloud_fusion::detail::findField(*output_, "velocity");
  const auto* vector = point_cloud_fusion::detail::findField(*output_, "velocity_vector");
  const auto* floating = point_cloud_fusion::detail::findField(*output_, "velocity_double");
  const auto* ring = point_cloud_fusion::detail::findField(*output_, "ring");
  const auto* time = point_cloud_fusion::detail::findField(*output_, "t");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(velocity, nullptr);
  ASSERT_NE(vector, nullptr);
  ASSERT_NE(floating, nullptr);
  ASSERT_NE(ring, nullptr);
  ASSERT_NE(time, nullptr);
  for (std::size_t i = 0; i < output_->width; ++i) {
    const float position = read<float>(*output_, i, x->offset);
    if (position < 10) {
      EXPECT_EQ(read<uint16_t>(*output_, i, ring->offset), 65535U);
      EXPECT_EQ(read<uint32_t>(*output_, i, time->offset), 777U);
    } else {
      EXPECT_FLOAT_EQ(read<float>(*output_, i, velocity->offset), 0);
      EXPECT_DOUBLE_EQ(read<double>(*output_, i, floating->offset), -123.5);
      for (uint32_t element = 0; element < 3; ++element) {
        EXPECT_TRUE(std::isnan(read<float>(*output_, i, vector->offset + element * sizeof(float))));
      }
      EXPECT_EQ(read<uint16_t>(*output_, i, ring->offset), static_cast<uint16_t>(position - 11 + 5));
      EXPECT_EQ(read<uint32_t>(*output_, i, time->offset), static_cast<uint32_t>((position - 11 + 1) * 1000));
    }
  }
}

TEST_P(HeterogeneousFusion, EmptyOverridesUseDatatypeDefaults) {
  start({rclcpp::Parameter("missing_field_fill_values", std::vector<std::string>{})});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse();
  checkOutput({1, 2, 3, 4, 11, 12, 13});
}

TEST_P(HeterogeneousFusion, UnusedOverridesWarnWithThrottlingAndPreserveMeasurements) {
  start({rclcpp::Parameter("missing_field_fill_values", std::vector<std::string>{"intensity=999", "t=nan"}),
         rclcpp::Parameter("output_fields", std::vector<std::string>{"x", "y", "z", "intensity"}),
         rclcpp::Parameter("motion_compensation.enable", true)});
  if (IsSkipped() || HasFatalFailure()) return;

  // Advance the ROS clock for each parameterized case so the shared logging
  // call site's throttle can be checked without wall-clock sleeps.
  static int64_t clock_offset_ns = 0;
  clock_offset_ns += 10'000'000'000LL;
  const auto log_time = fusion_->get_clock()->now().nanoseconds() + clock_offset_ns;
  {
    std::lock_guard<std::mutex> clock_lock(fusion_->get_clock()->get_clock_mutex());
    auto* clock = fusion_->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock), RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock, log_time), RCL_RET_OK);
  }

  std::vector<std::string> warnings;
  bool barrier_received = false;
  const std::string logger_name = fusion_->get_logger().get_name();
  auto logs = driver_->create_subscription<rcl_interfaces::msg::Log>(
      "/rosout", rclcpp::QoS(1000).reliable().transient_local(), [&](const rcl_interfaces::msg::Log& message) {
        if (message.name != logger_name) return;
        if (message.msg.find("Ignoring unused missing_field_fill_values") != std::string::npos) {
          EXPECT_EQ(message.level, rcl_interfaces::msg::Log::WARN);
          warnings.push_back(message.msg);
        }
        if (message.msg == "unused fill warning test barrier") barrier_received = true;
      });
  auto aeva = syntheticCloud(true);
  aeva.fields.erase(std::remove_if(aeva.fields.begin(), aeva.fields.end(), [](const auto& f) { return f.name == "t"; }),
                    aeva.fields.end());
  // Intensity exists in both clouds. The time field is unpublished, so its
  // invalid NaN-for-UINT32 override must also be ignored in CUDA's private layout.
  fuse(aeva);
  checkOutput({1, 2, 3, 4, 11, 12, 13}, false);
  ASSERT_TRUE(spinUntil([&] { return !warnings.empty(); }));
  ASSERT_EQ(warnings.size(), 1U);
  EXPECT_NE(warnings.front().find("intensity, t"), std::string::npos);

  fuse(aeva);
  ASSERT_NE(output_, nullptr);
  const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
  const auto* intensity = point_cloud_fusion::detail::findField(*output_, "intensity");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(intensity, nullptr);
  for (std::size_t i = 0; i < output_->width; ++i) {
    const float position = read<float>(*output_, i, x->offset);
    EXPECT_FLOAT_EQ(read<float>(*output_, i, intensity->offset), position < 10 ? position + 19 : position + 29);
  }
  RCLCPP_WARN(fusion_->get_logger(), "unused fill warning test barrier");
  ASSERT_TRUE(spinUntil([&] { return barrier_received; }));
  EXPECT_EQ(warnings.size(), 1U);
}

TEST_P(HeterogeneousFusion, FillPolicyUpdatesAreValidatedAndApplyToLaterBatches) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  const auto previous = fusion_->get_parameter("output_fields").as_string_array();
  const auto invalid =
      fusion_->set_parameters_atomically({rclcpp::Parameter("missing_field_fill_values", std::vector<std::string>{"v=invalid"}),
                                          rclcpp::Parameter("output_fields", std::vector<std::string>{"x", "y", "z"})});
  EXPECT_FALSE(invalid.successful);
  EXPECT_EQ(fusion_->get_parameter("output_fields").as_string_array(), previous);
  fuse();
  checkOutput({1, 2, 3, 4, 11, 12, 13});
  const auto valid = fusion_->set_parameters_atomically(
      {rclcpp::Parameter("missing_field_fill_values", std::vector<std::string>{"velocity=-99.5"})});
  ASSERT_TRUE(valid.successful) << valid.reason;
  fuse();
  ASSERT_NE(output_, nullptr);
  ASSERT_EQ(output_->width, 7U);
  const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
  const auto* velocity = point_cloud_fusion::detail::findField(*output_, "velocity");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(velocity, nullptr);
  for (std::size_t i = 0; i < output_->width; ++i) {
    if (read<float>(*output_, i, x->offset) >= 10) {
      EXPECT_FLOAT_EQ(read<float>(*output_, i, velocity->offset), -99.5F);
    }
  }
}

TEST_P(HeterogeneousFusion, ExplicitFieldsRetainVelocityWithReorderedOffsets) {
  start({rclcpp::Parameter("output_fields",
                           std::vector<std::string>{"velocity_double", "velocity_vector", "velocity", "z", "x", "y"})});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse();
  checkOutput({1, 2, 3, 4, 11, 12, 13});
  ASSERT_NE(output_, nullptr);
  EXPECT_EQ(output_->fields.front().name, "velocity_double");
  EXPECT_EQ(output_->fields.front().offset, 0U);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "intensity"), nullptr);
}

TEST_P(HeterogeneousFusion, MissingFloat64VelocityGetsNaN) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = syntheticCloud(true);
  // Exchange names to make the FLOAT64 field the actual velocity channel.
  std::swap(aeva.fields[0].name, aeva.fields[7].name);
  fuse(aeva);
  checkOutput({1, 2, 3, 4, 11, 12, 13}, true, 0.0F, aeva);
}

TEST_P(HeterogeneousFusion, MissingVelocityArrayGetsNaNInEveryElement) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = syntheticCloud(true);
  std::swap(aeva.fields[0].name, aeva.fields[6].name);
  fuse(aeva);
  checkOutput({1, 2, 3, 4, 11, 12, 13}, true, 0.0F, aeva);
}

TEST_P(HeterogeneousFusion, ExplicitFieldsCanExcludeVelocity) {
  start({rclcpp::Parameter("output_fields", std::vector<std::string>{"x", "y", "z"})});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse();
  checkOutput({1, 2, 3, 4, 11, 12, 13}, false);
}

TEST_P(HeterogeneousFusion, FilteringKeepsVelocityAttachedToSurvivingPoints) {
  start({rclcpp::Parameter("range_limits.enable", true), rclcpp::Parameter("range_limits.x_min", 2.5),
         rclcpp::Parameter("range_limits.x_max", 12.5)});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse();
  checkOutput({3, 4, 11, 12});
}

TEST_P(HeterogeneousFusion, SubsamplingKeepsVelocityAttachedToSelectedPoints) {
  // The parameter's allowed step is 100. Tile the small fixtures so sampling
  // selects every second Aeva point and every third Ouster point.
  const auto repeat = [](const Cloud& source, uint32_t repetitions) {
    Cloud result = source;
    result.width *= repetitions;
    result.row_step = result.width * result.point_step;
    result.data.clear();
    for (uint32_t i = 0; i < repetitions; ++i) {
      result.data.insert(result.data.end(), source.data.begin(), source.data.end());
    }
    return result;
  };
  start({rclcpp::Parameter("fixed_points_per_input_cloud", 100)});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse(repeat(syntheticCloud(true), 40), repeat(syntheticCloud(false), 100));
  std::vector<float> expected(100, 11.0F);
  for (int i = 0; i < 20; ++i) expected.insert(expected.end(), {1, 2, 3, 4});
  checkOutput(expected);
}

TEST_P(HeterogeneousFusion, MotionCompensationWithPrivateTimeFieldRetainsVelocity) {
  start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "base_link"),
         rclcpp::Parameter("output_fields",
                           std::vector<std::string>{"x", "y", "z", "velocity", "velocity_vector", "velocity_double"})});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse();
  checkOutput({1, 2, 3, 4, 11, 12, 13});
  ASSERT_NE(output_, nullptr);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "t"), nullptr);
}

TEST_P(HeterogeneousFusion, RigidTransformPreservesScalarVelocityBytes) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  tf2_ros::StaticTransformBroadcaster broadcaster(driver_);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "base_link";
  transform.child_frame_id = "aeva";
  transform.transform.translation.x = 100.0;
  transform.transform.rotation.w = 1.0;
  broadcaster.sendTransform(transform);
  // Let the fusion node's dedicated TF listener consume the static transform.
  executor_->spin_all(100ms);
  auto aeva = syntheticCloud(true);
  aeva.header.frame_id = "aeva";
  fuse(aeva);
  checkOutput({101, 102, 103, 104, 11, 12, 13}, true, 100.0F);
}

const std::vector<Configuration> configurations{
    {false, false},
    {false, true},
#ifdef ENABLE_CUDA
    {true, false},
    {true, true},
#endif
};

INSTANTIATE_TEST_SUITE_P(BackendsAndInputOrders,
                         HeterogeneousFusion,
                         testing::ValuesIn(configurations),
                         [](const testing::TestParamInfo<Configuration>& info) {
                           return std::string(info.param.cuda ? "Cuda" : "Cpu") +
                                  (info.param.aeva_first ? "AevaFirst" : "OusterFirst");
                         });

}  // namespace
