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
#include <tf2_ros/transform_broadcaster.h>
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

Cloud objectCloud() {
  auto cloud = syntheticCloud(true);
  const std::array<float, 4> velocities{5.0F, 0.0F, -5.0F, std::numeric_limits<float>::quiet_NaN()};
  for (std::size_t i = 0; i < velocities.size(); ++i) {
    write(cloud, i, 20, 10.0F);
    write(cloud, i, 28, 0.0F);
    write(cloud, i, 12, 0.0F);
    write(cloud, i, 0, velocities[i]);
    write(cloud, i, 32, uint32_t{100000000});
    write(cloud, i, 48, static_cast<double>(velocities[i]));
  }
  return cloud;
}

std::vector<rclcpp::Parameter> objectParameters() {
  return {rclcpp::Parameter("motion_compensation.enable", true),
          rclcpp::Parameter("motion_compensation.fixed_frame", "base_link"),
          rclcpp::Parameter("object_motion_compensation.enable", true)};
}

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
    const auto ouster_second_offset = ouster.header.stamp.sec - aeva.header.stamp.sec;
    aeva.header.stamp.sec = next_stamp_;
    ouster.header.stamp.sec = next_stamp_ + ouster_second_offset;
    // Retry best-effort sensor messages while middleware endpoints finish
    // matching. Most fixtures use equal stamps to avoid needing a later scan.
    auto next_publish = std::chrono::steady_clock::now();
    ASSERT_TRUE(spinUntil([&] {
      if (output_ && output_->header.stamp.sec >= first_stamp) return true;
      if (std::chrono::steady_clock::now() >= next_publish) {
        aeva_publisher_->publish(aeva);
        ouster_publisher_->publish(ouster);
        ++next_stamp_;
        aeva.header.stamp.sec = next_stamp_;
        ouster.header.stamp.sec = next_stamp_ + ouster_second_offset;
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

  void checkObjectPoint(float intensity, float x, float y = 0.0F, float z = 0.0F) {
    ASSERT_NE(output_, nullptr);
    const auto* feature = point_cloud_fusion::detail::findField(*output_, "intensity");
    ASSERT_NE(feature, nullptr);
    const std::array<float, 3> expected{x, y, z};
    for (std::size_t i = 0; i < output_->width; ++i) {
      if (read<float>(*output_, i, feature->offset) != intensity) continue;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto* coordinate = point_cloud_fusion::detail::findField(*output_, std::array<const char*, 3>{"x", "y", "z"}[axis]);
        ASSERT_NE(coordinate, nullptr);
        EXPECT_NEAR(read<float>(*output_, i, coordinate->offset), expected[axis], 2.0e-5F) << "intensity=" << intensity;
      }
      return;
    }
    FAIL() << "Point with intensity=" << intensity << " is missing";
  }

  void checkObjectTimestamp(const std::string& mode, float expected_x) {
    auto params = objectParameters();
    params.emplace_back("output_stamp_mode", mode);
    params.emplace_back("max_time_diff_sec", 0.5);
    start(params);
    if (IsSkipped() || HasFatalFailure()) return;
    next_stamp_ = 1800000000;
    auto aeva = objectCloud();
    auto ouster = syntheticCloud(false);
    aeva.header.stamp.nanosec = 950000000;
    ouster.header.stamp.sec += 1;
    ouster.header.stamp.nanosec = 50000000;
    // Acquisition crosses the second boundary; converting epoch timestamps to
    // float before subtraction would erase these intervals entirely.
    for (std::size_t i = 0; i < aeva.width; ++i) write(aeva, i, 32, uint32_t{50000000});
    fuse(aeva, ouster);
    checkObjectPoint(20, expected_x);
    checkObjectPoint(21, 10.0F);
  }

  void checkMixedDatatypeCalibration(bool explicit_fields) {
    std::vector<rclcpp::Parameter> parameters{
        rclcpp::Parameter("feature_calibration.enable", true),
        rclcpp::Parameter("feature_calibration.field", "intensity"),
        rclcpp::Parameter("feature_calibration.leading_inputs", std::vector<std::string>{namespace_ + "/ouster"}),
        rclcpp::Parameter("feature_calibration.follower_inputs", std::vector<std::string>{namespace_ + "/aeva"}),
        rclcpp::Parameter("feature_calibration.minimum_samples", 100),
        rclcpp::Parameter("feature_calibration.window_samples", 100),
        rclcpp::Parameter("feature_calibration.update_interval_sec", 0.1),
        rclcpp::Parameter("missing_field_fill_values", std::vector<std::string>{"intensity=65535"})};
    if (explicit_fields) {
      parameters.emplace_back("output_fields", std::vector<std::string>{"intensity", "z", "x", "y"});
    }
    start(parameters);
    if (IsSkipped() || HasFatalFailure()) return;
    const auto make_cloud = [](bool follower) {
      auto cloud = syntheticCloud(follower);
      cloud.width = 100;
      cloud.row_step = cloud.width * cloud.point_step;
      cloud.data.assign(cloud.row_step, 0);
      if (!follower) {
        for (auto& f : cloud.fields) {
          if (f.name == "intensity") f.datatype = Field::UINT16;
        }
      }
      for (std::size_t point = 0; point < cloud.width; ++point) {
        write(cloud, point, follower ? 20 : 0, static_cast<float>(point + (follower ? 1 : 1001)));
        if (follower) write(cloud, point, 4, static_cast<float>(point));
        else write(cloud, point, 12, static_cast<uint16_t>(point + 100));
      }
      return cloud;
    };
    const auto follower = make_cloud(true);
    const auto leader = make_cloud(false);
    bool calibrated = false;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!calibrated && std::chrono::steady_clock::now() < deadline) {
      fuse(follower, leader);
      if (HasFatalFailure()) return;
      const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
      const auto* intensity = point_cloud_fusion::detail::findField(*output_, "intensity");
      ASSERT_NE(x, nullptr);
      ASSERT_NE(intensity, nullptr);
      ASSERT_EQ(intensity->datatype, Field::UINT16);
      ASSERT_EQ(output_->width, 200U);
      for (std::size_t point = 0; point < output_->width; ++point) {
        const float position = read<float>(*output_, point, x->offset);
        const uint16_t value = read<uint16_t>(*output_, point, intensity->offset);
        if (position >= 1001.0F) EXPECT_EQ(value, static_cast<uint16_t>(position - 1001.0F + 100.0F));
        if (position == 1.0F && value == 100U) calibrated = true;
      }
    }
    ASSERT_TRUE(calibrated) << "Mixed-datatype feature mapping did not become active";
    const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
    const auto* intensity = point_cloud_fusion::detail::findField(*output_, "intensity");
    for (std::size_t point = 0; point < output_->width; ++point) {
      const float position = read<float>(*output_, point, x->offset);
      if (position < 1001.0F) {
        EXPECT_EQ(read<uint16_t>(*output_, point, intensity->offset), static_cast<uint16_t>(position - 1.0F + 100.0F));
      }
    }
  }

  void checkDeskewedInput(bool aeva_deskewed) {
    // deskewed_inputs follows input_topics order, which depends on the fixture.
    const std::vector<bool> deskewed =
        GetParam().aeva_first ? std::vector<bool>{aeva_deskewed, false} : std::vector<bool>{false, aeva_deskewed};
    start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "map"),
           rclcpp::Parameter("motion_compensation.deskewed_inputs", deskewed),
           rclcpp::Parameter("object_motion_compensation.enable", true)});
    if (IsSkipped() || HasFatalFailure()) return;
    tf2_ros::TransformBroadcaster broadcaster(driver_);
    for (const int32_t seconds : {98, 107}) {
      geometry_msgs::msg::TransformStamped transform;
      transform.header.stamp.sec = seconds;
      transform.header.frame_id = "map";
      transform.child_frame_id = "base_link";
      transform.transform.translation.x = 2.0 * seconds;
      transform.transform.rotation.w = 1.0;
      broadcaster.sendTransform(transform);
    }
    executor_->spin_all(100ms);
    fuse(objectCloud());
    // Same scene as ObjectMotionCombinesWithEgoTranslationWithoutDoubleCorrection:
    // a deskewed input must not get the 0.2 m ego shift again, while object
    // motion still uses the per-point acquisition time.
    const float ego_shift = aeva_deskewed ? 0.0F : 0.2F;
    checkObjectPoint(20, 9.5F + ego_shift);
    checkObjectPoint(21, 10.0F + ego_shift);
    checkObjectPoint(22, 10.5F + ego_shift);
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
         rclcpp::Parameter("motion_compensation.enable", true),
         rclcpp::Parameter("motion_compensation.fixed_frame", "base_link")});
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

TEST_P(HeterogeneousFusion, MixedDatatypeCalibrationRetainsFieldInUnion) { checkMixedDatatypeCalibration(false); }

TEST_P(HeterogeneousFusion, MixedDatatypeCalibrationPreservesLeadersWithExplicitFields) { checkMixedDatatypeCalibration(true); }

TEST_P(HeterogeneousFusion, FeatureCalibrationPreservesNonFiniteMeasurements) {
  start({rclcpp::Parameter("feature_calibration.enable", true),
         rclcpp::Parameter("feature_calibration.field", "intensity"),
         rclcpp::Parameter("feature_calibration.leading_inputs", std::vector<std::string>{namespace_ + "/ouster"}),
         rclcpp::Parameter("feature_calibration.follower_inputs", std::vector<std::string>{namespace_ + "/aeva"}),
         rclcpp::Parameter("feature_calibration.minimum_samples", 100),
         rclcpp::Parameter("feature_calibration.window_samples", 100),
         rclcpp::Parameter("feature_calibration.update_interval_sec", 0.1)});
  if (IsSkipped() || HasFatalFailure()) return;

  const auto make_cloud = [](bool follower) {
    auto cloud = syntheticCloud(follower);
    cloud.width = follower ? 103 : 100;
    cloud.row_step = cloud.width * cloud.point_step;
    cloud.data.assign(cloud.row_step, 0);
    for (std::size_t point = 0; point < cloud.width; ++point) {
      write(cloud, point, follower ? 20 : 0, follower ? 1.0F : 11.0F);
      write(cloud, point, follower ? 4 : 12, static_cast<float>(point + (follower ? 0 : 100)));
    }
    if (follower) {
      const std::array<float, 3> invalid{std::numeric_limits<float>::quiet_NaN(),
                                        std::numeric_limits<float>::infinity(),
                                        -std::numeric_limits<float>::infinity()};
      for (std::size_t index = 0; index < invalid.size(); ++index) {
        write(cloud, 100 + index, 20, static_cast<float>(2 + index));
        write(cloud, 100 + index, 4, invalid[index]);
      }
    }
    return cloud;
  };
  const auto follower = make_cloud(true);
  const auto leader = make_cloud(false);
  bool calibrated = false;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!calibrated && std::chrono::steady_clock::now() < deadline) {
    fuse(follower, leader);
    if (HasFatalFailure()) return;
    const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
    const auto* intensity = point_cloud_fusion::detail::findField(*output_, "intensity");
    ASSERT_NE(x, nullptr);
    ASSERT_NE(intensity, nullptr);
    for (std::size_t point = 0; point < output_->width; ++point) {
      if (read<float>(*output_, point, x->offset) == 1.0F && read<float>(*output_, point, intensity->offset) >= 100.0F) {
        calibrated = true;
      }
    }
  }
  ASSERT_TRUE(calibrated) << "Feature mapping did not become active";
  const auto* x = point_cloud_fusion::detail::findField(*output_, "x");
  const auto* intensity = point_cloud_fusion::detail::findField(*output_, "intensity");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(intensity, nullptr);
  std::size_t invalid_count = 0;
  for (std::size_t point = 0; point < output_->width; ++point) {
    const float position = read<float>(*output_, point, x->offset);
    const float value = read<float>(*output_, point, intensity->offset);
    if (position == 2.0F) {
      EXPECT_TRUE(std::isnan(value));
      ++invalid_count;
    } else if (position == 3.0F || position == 4.0F) {
      EXPECT_TRUE(std::isinf(value));
      EXPECT_EQ(std::signbit(value), position == 4.0F);
      ++invalid_count;
    }
  }
  EXPECT_EQ(invalid_count, 3U);
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

TEST_P(HeterogeneousFusion, ObjectMotionUsesAbsoluteVelocityAndPreservesUnknownMeasurements) {
  start(objectParameters());
  if (IsSkipped() || HasFatalFailure()) return;
  fuse(objectCloud());
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(21, 10.0F);
  checkObjectPoint(22, 10.5F);
  checkObjectPoint(23, 10.0F);
  checkObjectPoint(40, 11.0F, 0.5F, 1.0F);
  ASSERT_EQ(output_->width, 7U);
  const auto* velocity = point_cloud_fusion::detail::findField(*output_, "velocity");
  ASSERT_NE(velocity, nullptr);
  const auto* intensity = point_cloud_fusion::detail::findField(*output_, "intensity");
  for (std::size_t i = 0; i < output_->width; ++i) {
    if (read<float>(*output_, i, intensity->offset) == 20.0F) {
      EXPECT_FLOAT_EQ(read<float>(*output_, i, velocity->offset), 5.0F);
    }
  }
}

TEST_P(HeterogeneousFusion, ObjectMotionUsesPrivateMeasuredFieldsAndNeverOutputFills) {
  auto params = objectParameters();
  params.emplace_back("output_fields", std::vector<std::string>{"x", "y", "z", "intensity"});
  params.emplace_back("missing_field_fill_values", std::vector<std::string>{"velocity=100", "t=nan"});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  fuse(objectCloud());
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(40, 11.0F, 0.5F, 1.0F);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "velocity"), nullptr);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "t"), nullptr);
}

TEST_P(HeterogeneousFusion, ObjectMotionDoesNotTreatNumericFillsAsOusterMeasurements) {
  auto params = objectParameters();
  params.emplace_back("missing_field_fill_values", std::vector<std::string>{"velocity=100"});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto ouster = syntheticCloud(false);
  for (std::size_t i = 0; i < ouster.width; ++i) write(ouster, i, 16, uint32_t{100000000});
  fuse(objectCloud(), ouster);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(40, 11.0F, 0.5F, 1.0F);
}

TEST_P(HeterogeneousFusion, ObjectMotionSupportsFloat64AndCustomVelocityNames) {
  auto params = objectParameters();
  params.emplace_back("object_motion_compensation.velocity_field", "custom_speed");
  params.emplace_back("output_fields", std::vector<std::string>{"x", "y", "z", "intensity"});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = objectCloud();
  for (auto& field : aeva.fields)
    if (field.name == "velocity_double") field.name = "custom_speed";
  fuse(aeva);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(22, 10.5F);
}

TEST_P(HeterogeneousFusion, ObjectMotionReadsPublishedFloat64Measurements) {
  auto params = objectParameters();
  params.emplace_back("object_motion_compensation.velocity_field", "velocity_double");
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = objectCloud();
  write(aeva, 0, 48, 5.125);
  write(aeva, 1, 48, std::numeric_limits<double>::max());
  write(aeva, 2, 48, -std::numeric_limits<double>::infinity());
  fuse(aeva);
  checkObjectPoint(20, 9.4875F);
  checkObjectPoint(21, 10.0F);
  checkObjectPoint(22, 10.0F);
  const auto* velocity = point_cloud_fusion::detail::findField(*output_, "velocity_double");
  ASSERT_NE(velocity, nullptr);
  EXPECT_EQ(velocity->datatype, Field::FLOAT64);
}

TEST_P(HeterogeneousFusion, ObjectMotionSurvivesConflictingAndChangingVelocitySchemas) {
  start(objectParameters());
  if (IsSkipped() || HasFatalFailure()) return;
  auto other = objectCloud();
  other.fields.erase(other.fields.begin());
  for (auto& field : other.fields)
    if (field.name == "velocity_double") field.name = "velocity";
  for (std::size_t i = 0; i < other.width; ++i) write(other, i, 4, static_cast<float>(i + 40));
  write(other, 0, 48, -10.0);
  fuse(objectCloud(), other);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(40, 11.0F);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "velocity"), nullptr);

  // Exercise the homogeneous fast path, then absent and restored measurements
  // on the same running node. CUDA metadata must not leak across batches.
  other = objectCloud();
  for (std::size_t i = 0; i < other.width; ++i) write(other, i, 4, static_cast<float>(i + 40));
  fuse(objectCloud(), other);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(40, 9.5F);
  ASSERT_NE(point_cloud_fusion::detail::findField(*output_, "velocity"), nullptr);
  fuse(objectCloud());
  checkObjectPoint(40, 11.0F, 0.5F, 1.0F);
  fuse(objectCloud(), other);
  checkObjectPoint(40, 9.5F);
}

TEST_P(HeterogeneousFusion, ObjectMotionRejectsNonScalarAndIntegerVelocityFields) {
  start(objectParameters());
  if (IsSkipped() || HasFatalFailure()) return;
  for (const auto datatype : {Field::FLOAT32, Field::UINT32}) {
    auto aeva = objectCloud();
    aeva.fields[0].datatype = datatype;
    aeva.fields[0].count = datatype == Field::FLOAT32 ? 3 : 1;
    fuse(aeva);
    checkObjectPoint(20, 10.0F);
  }
}

TEST_P(HeterogeneousFusion, ObjectMotionChecksSpeedTimeAndFiniteLimits) {
  start(objectParameters());
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = objectCloud();
  write(aeva, 0, 0, std::numeric_limits<float>::infinity());
  write(aeva, 1, 0, 101.0F);
  write(aeva, 2, 32, uint32_t{300000000});
  fuse(aeva);
  for (float intensity : {20.0F, 21.0F, 22.0F, 23.0F}) checkObjectPoint(intensity, 10.0F);
}

TEST_P(HeterogeneousFusion, ObjectMotionSupportsApproachingPositiveDoppler) {
  auto params = objectParameters();
  params.emplace_back("object_motion_compensation.doppler_sign", -1.0);
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  fuse(objectCloud());
  checkObjectPoint(20, 10.5F);
  checkObjectPoint(22, 9.5F);
}

TEST_P(HeterogeneousFusion, ObjectMotionRetainsMeasuredTimingDespiteConflictingTimeTypes) {
  start(objectParameters());
  if (IsSkipped() || HasFatalFailure()) return;
  auto ouster = syntheticCloud(false);
  ouster.fields[4].datatype = Field::INT32;
  fuse(objectCloud(), ouster);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(22, 10.5F);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "t"), nullptr);
}

TEST_P(HeterogeneousFusion, ObjectMotionRetainsPrivateTimingDespiteConflictingTimeTypes) {
  auto params = objectParameters();
  params.emplace_back("output_fields", std::vector<std::string>{"x", "y", "z", "intensity"});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto ouster = syntheticCloud(false);
  ouster.fields[4].datatype = Field::INT32;
  fuse(objectCloud(), ouster);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(22, 10.5F);
  EXPECT_EQ(point_cloud_fusion::detail::findField(*output_, "t"), nullptr);
}

TEST_P(HeterogeneousFusion, ObjectMotionSupportsPerInputFieldsAndDisabledEntries) {
  auto params = objectParameters();
  params.emplace_back("object_motion_compensation.velocity_fields", GetParam().aeva_first
                                                                        ? std::vector<std::string>{"", "intensity"}
                                                                        : std::vector<std::string>{"intensity", ""});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto ouster = syntheticCloud(false);
  write(ouster, 0, 4, 0.0F);
  write(ouster, 0, 8, 0.0F);
  write(ouster, 0, 16, uint32_t{100000000});
  fuse(objectCloud(), ouster);
  checkObjectPoint(20, 10.0F);
  checkObjectPoint(40, 7.0F);
}

TEST_P(HeterogeneousFusion, ObjectMotionUsesPrivateValidityFlagsAndSkipsMissingFlags) {
  auto params = objectParameters();
  params.emplace_back("object_motion_compensation.invalid_flags_field", "point_flags_msb");
  params.emplace_back("object_motion_compensation.invalid_flags_mask", int64_t{1});
  params.emplace_back("output_fields", std::vector<std::string>{"x", "y", "z", "intensity"});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = objectCloud();
  aeva.fields.push_back(field("point_flags_msb", 56, Field::UINT32));
  for (std::size_t i = 0; i < aeva.width; ++i) write(aeva, i, 56, uint32_t{0});
  write(aeva, 2, 56, uint32_t{1});
  fuse(aeva);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(22, 10.0F);
  fuse(objectCloud());
  checkObjectPoint(20, 10.0F);
}

TEST_P(HeterogeneousFusion, ObjectMotionUsesCloudTimestampWithoutPerPointTime) {
  auto params = objectParameters();
  params.emplace_back("output_stamp_mode", "latest");
  params.emplace_back("max_time_diff_sec", 0.5);
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = objectCloud();
  auto ouster = syntheticCloud(false);
  aeva.fields.erase(std::remove_if(aeva.fields.begin(), aeva.fields.end(), [](const auto& field) { return field.name == "t"; }),
                    aeva.fields.end());
  ouster.header.stamp.nanosec = 100000000;
  fuse(aeva, ouster);
  checkObjectPoint(20, 10.5F);
  checkObjectPoint(22, 9.5F);
}

TEST_P(HeterogeneousFusion, ObjectMotionUsesEarliestEpochTimestampAcrossSecondBoundary) {
  checkObjectTimestamp("earliest", 9.75F);
}

TEST_P(HeterogeneousFusion, ObjectMotionUsesLatestEpochTimestampAcrossSecondBoundary) { checkObjectTimestamp("latest", 10.25F); }

TEST_P(HeterogeneousFusion, ObjectMotionUsesMeanEpochTimestampAcrossSecondBoundary) { checkObjectTimestamp("mean", 10.0F); }

TEST_P(HeterogeneousFusion, ObjectMotionUsesInputZeroEpochTimestampAcrossSecondBoundary) {
  checkObjectTimestamp("input0", GetParam().aeva_first ? 9.75F : 10.25F);
}

TEST_P(HeterogeneousFusion, ObjectMotionChecksHighBitFlagsAndMalformedDescriptors) {
  auto params = objectParameters();
  params.emplace_back("object_motion_compensation.invalid_flags_field", "flags");
  params.emplace_back("object_motion_compensation.invalid_flags_mask", int64_t{0x80000000});
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  auto aeva = objectCloud();
  aeva.fields.push_back(field("flags", 56, Field::UINT32));
  for (std::size_t i = 0; i < aeva.width; ++i) write(aeva, i, 56, uint32_t{1});
  write(aeva, 2, 56, uint32_t{0x80000000});
  fuse(aeva);
  checkObjectPoint(20, 9.5F);
  checkObjectPoint(22, 10.0F);
  aeva.fields.back().offset = aeva.point_step - 1;
  fuse(aeva);
  checkObjectPoint(20, 10.0F);
  aeva.fields.back().offset = 56;
  aeva.fields.front().offset = aeva.point_step - 1;
  fuse(aeva);
  checkObjectPoint(20, 10.0F);
}

TEST_P(HeterogeneousFusion, ObjectMotionUsesAcquisitionRayBeforeTranslationAndRotation) {
  start(objectParameters());
  if (IsSkipped() || HasFatalFailure()) return;
  tf2_ros::StaticTransformBroadcaster broadcaster(driver_);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "base_link";
  transform.child_frame_id = "aeva";
  transform.transform.translation.x = 100.0;
  transform.transform.rotation.z = std::sqrt(0.5);
  transform.transform.rotation.w = std::sqrt(0.5);
  broadcaster.sendTransform(transform);
  executor_->spin_all(100ms);
  auto aeva = objectCloud();
  aeva.header.frame_id = "aeva";
  fuse(aeva);
  checkObjectPoint(20, 100.0F, 9.5F);
  checkObjectPoint(22, 100.0F, 10.5F);
}

TEST_P(HeterogeneousFusion, ObjectMotionCombinesWithEgoTranslationWithoutDoubleCorrection) {
  start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "map"),
         rclcpp::Parameter("object_motion_compensation.enable", true)});
  if (IsSkipped() || HasFatalFailure()) return;
  tf2_ros::TransformBroadcaster broadcaster(driver_);
  // Keep both samples inside TF's ten-second cache, including when DDS
  // delivers the newer sample first.
  for (const int32_t seconds : {98, 107}) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp.sec = seconds;
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.transform.translation.x = 2.0 * seconds;
    transform.transform.rotation.w = 1.0;
    broadcaster.sendTransform(transform);
  }
  executor_->spin_all(100ms);
  fuse(objectCloud());
  checkObjectPoint(20, 9.7F);
  checkObjectPoint(21, 10.2F);
  checkObjectPoint(22, 10.7F);
}

TEST_P(HeterogeneousFusion, DeskewedObjectMotionUsesAcquisitionOriginDuringSidewaysEgoMotion) {
  const std::vector<bool> deskewed = GetParam().aeva_first ? std::vector<bool>{true, false} : std::vector<bool>{false, true};
  start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "map"),
         rclcpp::Parameter("motion_compensation.deskewed_inputs", deskewed),
         rclcpp::Parameter("object_motion_compensation.enable", true)});
  if (IsSkipped() || HasFatalFailure()) return;
  tf2_ros::TransformBroadcaster broadcaster(driver_);
  for (const int32_t seconds : {98, 107}) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp.sec = seconds;
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.transform.translation.y = 10.0 * (seconds - 100);
    transform.transform.rotation.w = 1.0;
    broadcaster.sendTransform(transform);
  }
  executor_->spin_all(100ms);
  auto aeva = objectCloud();
  // At acquisition the sensor was 1m sideways from its header pose.
  // The driver publishes the raw (10,0,0) ray as deskewed (10,1,0).
  for (std::size_t point = 0; point < 4; ++point) write(aeva, point, 28, 1.0F);
  fuse(aeva);
  checkObjectPoint(20, 9.5F, 1.0F);
  checkObjectPoint(21, 10.0F, 1.0F);
  checkObjectPoint(22, 10.5F, 1.0F);
}

TEST_P(HeterogeneousFusion, DeskewedInputIsNotDeskewedTwice) { checkDeskewedInput(true); }

TEST_P(HeterogeneousFusion, InputNotMarkedDeskewedIsDeskewed) { checkDeskewedInput(false); }

TEST_P(HeterogeneousFusion, DeskewedInputsMustMatchInputTopics) {
  EXPECT_THROW(start({rclcpp::Parameter("motion_compensation.deskewed_inputs", std::vector<bool>{true})}), std::invalid_argument);
}

TEST_P(HeterogeneousFusion, ObjectMotionIsDisabledDuringTfFallback) {
  start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "missing"),
         rclcpp::Parameter("motion_compensation.tf_timeout_sec", 0.0),
         rclcpp::Parameter("object_motion_compensation.enable", true)});
  if (IsSkipped() || HasFatalFailure()) return;
  fuse(objectCloud());
  checkObjectPoint(20, 10.0F);
  tf2_ros::StaticTransformBroadcaster broadcaster(driver_);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "missing";
  transform.child_frame_id = "base_link";
  transform.transform.rotation.w = 1.0;
  broadcaster.sendTransform(transform);
  executor_->spin_all(100ms);
  fuse(objectCloud());
  checkObjectPoint(20, 9.5F);
}

TEST_P(HeterogeneousFusion, MissingTfWaitIsBoundedWhileRosTimeIsPaused) {
  start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "missing"),
         rclcpp::Parameter("motion_compensation.tf_timeout_sec", 0.02),
         rclcpp::Parameter("object_motion_compensation.enable", true)});
  if (IsSkipped() || HasFatalFailure()) return;
  {
    std::lock_guard<std::mutex> clock_lock(fusion_->get_clock()->get_clock_mutex());
    auto* clock = fusion_->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock), RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock, 100'000'000'000LL), RCL_RET_OK);
  }
  const auto start_time = std::chrono::steady_clock::now();
  // Both the advanced motion lookup and the rigid fallback lookup fail for
  // Aeva. Ouster is already in the output frame and must still be published.
  auto aeva = objectCloud();
  aeva.header.frame_id = "missing_sensor";
  fuse(aeva);
  ASSERT_NE(output_, nullptr);
  EXPECT_EQ(output_->width, 3U);
  checkObjectPoint(40, 11.0F, 0.5F, 1.0F);
  EXPECT_LT(std::chrono::steady_clock::now() - start_time, 1s);
}

TEST_P(HeterogeneousFusion, ObjectMotionCombinesWithEgoRotation) {
  start({rclcpp::Parameter("motion_compensation.enable", true), rclcpp::Parameter("motion_compensation.fixed_frame", "map"),
         rclcpp::Parameter("object_motion_compensation.enable", true)});
  if (IsSkipped() || HasFatalFailure()) return;
  tf2_ros::TransformBroadcaster broadcaster(driver_);
  for (const int32_t seconds : {98, 107}) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp.sec = seconds;
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    const double angle = (seconds - 98) * std::acos(-1.0) / 20.0;
    transform.transform.rotation.z = std::sin(angle / 2.0);
    transform.transform.rotation.w = std::cos(angle / 2.0);
    broadcaster.sendTransform(transform);
  }
  executor_->spin_all(100ms);
  fuse(objectCloud());
  const float angle = static_cast<float>(std::acos(-1.0) * 0.1 / 20.0);
  checkObjectPoint(20, 9.5F * std::cos(angle), 9.5F * std::sin(angle));
  checkObjectPoint(21, 10.0F * std::cos(angle), 10.0F * std::sin(angle));
  checkObjectPoint(22, 10.5F * std::cos(angle), 10.5F * std::sin(angle));
}

TEST_P(HeterogeneousFusion, ObjectMotionAppliesRangeFilteringAfterDisplacement) {
  auto params = objectParameters();
  params.emplace_back("range_limits.enable", true);
  params.emplace_back("range_limits.x_max", 9.75);
  start(params);
  if (IsSkipped() || HasFatalFailure()) return;
  fuse(objectCloud());
  ASSERT_NE(output_, nullptr);
  EXPECT_EQ(output_->width, 1U);
  checkObjectPoint(20, 9.5F);
}

TEST_P(HeterogeneousFusion, ObjectMotionParameterUpdatesAreAtomicWithEgoMotion) {
  start({rclcpp::Parameter("motion_compensation.fixed_frame", "base_link")});
  if (IsSkipped() || HasFatalFailure()) return;
  EXPECT_FALSE(fusion_->set_parameter(rclcpp::Parameter("object_motion_compensation.enable", true)).successful);
  EXPECT_TRUE(fusion_
                  ->set_parameters_atomically({rclcpp::Parameter("motion_compensation.enable", true),
                                               rclcpp::Parameter("object_motion_compensation.enable", true)})
                  .successful);
  fuse(objectCloud());
  checkObjectPoint(20, 9.5F);
  EXPECT_FALSE(fusion_->set_parameter(rclcpp::Parameter("motion_compensation.enable", false)).successful);
  EXPECT_TRUE(fusion_
                  ->set_parameters_atomically({rclcpp::Parameter("motion_compensation.enable", false),
                                               rclcpp::Parameter("object_motion_compensation.enable", false)})
                  .successful);
  fuse(objectCloud());
  checkObjectPoint(20, 10.0F);
}

TEST_P(HeterogeneousFusion, ObjectMotionRejectsInvalidConfigurationAtStartup) {
  start();
  if (IsSkipped() || HasFatalFailure()) return;
  const std::vector<std::vector<rclcpp::Parameter>> invalid{
      {rclcpp::Parameter("object_motion_compensation.enable", true)},
      {rclcpp::Parameter("object_motion_compensation.doppler_sign", 0.0)},
      {rclcpp::Parameter("object_motion_compensation.velocity_fields", std::vector<std::string>{"velocity"})},
      {rclcpp::Parameter("object_motion_compensation.max_speed_mps", std::numeric_limits<double>::quiet_NaN())},
      {rclcpp::Parameter("object_motion_compensation.invalid_flags_mask", int64_t{1})}};
  for (const auto& params : invalid) EXPECT_THROW(start(params), std::invalid_argument);
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
