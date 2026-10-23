// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <point_cloud_fusion/point_cloud_fusion.hpp>

namespace point_cloud_fusion {
namespace {

using namespace std::chrono_literals;
using Cloud = sensor_msgs::msg::PointCloud2;

using Collector = PartialBatchCollector<std::shared_ptr<const int>>;

class PartialBatchCollectorTest : public ::testing::Test {
 protected:
  Collector collector{3, {10ms, 5ms, 200ms, 50ms}};

  static bool push(Collector& target, std::size_t topic, Collector::Time stamp, Collector::Time arrival, int id = 1) {
    return target.push(topic, stamp, arrival, std::make_shared<const int>(id));
  }
};

TEST_F(PartialBatchCollectorTest, AlternatingMissingScansDoNotLimitOutputToSlowerInput) {
  for (int cycle = 0; cycle < 10; ++cycle) {
    const auto now = 100ms + cycle * 50ms;
    ASSERT_TRUE(push(collector, 0, now, now));
    if (cycle % 2 == 0) {
      ASSERT_TRUE(push(collector, 1, now, now));
    }
    const auto batch = collector.take(now + 5ms);
    EXPECT_EQ(batch.size(), cycle % 2 == 0 ? 2U : 1U);
    EXPECT_TRUE(collector.take(now + 6ms).empty());
  }
}

TEST_F(PartialBatchCollectorTest, EnforcesFullBatchSpreadRatherThanDistanceFromTrigger) {
  ASSERT_TRUE(push(collector, 0, 91ms, 100ms));
  ASSERT_TRUE(push(collector, 1, 109ms, 101ms));
  ASSERT_TRUE(push(collector, 2, 100ms, 102ms));
  const auto batch = collector.take(105ms);
  ASSERT_EQ(batch.size(), 2U);
  EXPECT_EQ(batch[0].topic_index, 1U);
  EXPECT_EQ(batch[1].topic_index, 2U);
  EXPECT_LE(batch[0].stamp - batch[1].stamp, 10ms);
}

TEST_F(PartialBatchCollectorTest, PhaseShiftedTwentyHertzInputsCannotProduceFortyHertzOutput) {
  Collector sync{2, {10ms, 5ms, 200ms, 50ms}};
  std::vector<Collector::Time> outputs;
  for (int time_ms = 0; time_ms < 1000; ++time_ms) {
    const auto now = time_ms * 1ms;
    if (time_ms % 25 == 0) {
      ASSERT_TRUE(push(sync, (time_ms / 25) % 2, now, now));
    }
    if (!sync.take(now).empty()) {
      outputs.push_back(now);
    }
  }
  ASSERT_EQ(outputs.size(), 20U);
  for (std::size_t i = 1; i < outputs.size(); ++i) {
    EXPECT_GE(outputs[i] - outputs[i - 1], 50ms);
  }
}

TEST_F(PartialBatchCollectorTest, PendingUpdatesRespectCollectionAndPublicationDeadlines) {
  ASSERT_TRUE(push(collector, 0, 100ms, 100ms, 1));
  ASSERT_TRUE(push(collector, 0, 104ms, 104ms, 2));
  ASSERT_TRUE(collector.deadline());
  EXPECT_EQ(*collector.deadline(), 105ms);
  EXPECT_TRUE(collector.take(104ms).empty());
  const auto first = collector.take(105ms);
  ASSERT_EQ(first.size(), 1U);
  EXPECT_EQ(*first[0].message, 2);

  collector.published(130ms);  // Include processing time in the publication interval.
  ASSERT_TRUE(push(collector, 0, 140ms, 140ms, 3));
  EXPECT_TRUE(collector.take(145ms).empty());
  ASSERT_TRUE(push(collector, 0, 175ms, 175ms, 4));
  ASSERT_TRUE(push(collector, 1, 176ms, 176ms, 5));
  ASSERT_TRUE(collector.deadline());
  EXPECT_EQ(*collector.deadline(), 180ms);
  EXPECT_TRUE(collector.take(179ms).empty());
  const auto next = collector.take(180ms);
  ASSERT_EQ(next.size(), 2U);
  EXPECT_EQ(*next[0].message, 4);
  EXPECT_EQ(*next[1].message, 5);
}

/**
 * @brief Exercise real subscriptions and timers using an explicitly advanced ROS clock.
 */
class PartialFusionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }

  void SetUp() override {
    namespace_ = "/fusion_test/" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
    topics_ = {namespace_ + "/input0", namespace_ + "/input1"};
    io_ = std::make_shared<rclcpp::Node>("fusion_test_io", namespace_);
    clock_pub_ = io_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    for (const auto& topic : topics_) {
      inputs_.push_back(io_->create_publisher<Cloud>(topic, 10));
    }
    output_ = io_->create_subscription<Cloud>(namespace_ + "/point_cloud_fusion/point_cloud", rclcpp::SensorDataQoS(),
                                              [this](Cloud::ConstSharedPtr msg) { outputs_.push_back(*msg); });
    executor_.add_node(io_);
  }

  void TearDown() override {
    if (fusion_) {
      executor_.remove_node(fusion_);
      fusion_.reset();
    }
    executor_.remove_node(io_);
  }

  rclcpp::NodeOptions options(const std::string& mode = "partial") {
    rclcpp::NodeOptions result;
    result.arguments({"--ros-args", "-r", "__ns:=" + namespace_});
    result.parameter_overrides({
        {"target_frame", "base_link"},
        {"input_topics", topics_},
        {"use_cuda", false},
        {"use_sim_time", true},
        {"sync_mode", mode},
        {"max_time_diff_sec", 0.01},
        {"partial_sync.publish_rate_hz", 20.0},
        {"partial_sync.timeout_sec", 0.01},
        {"partial_sync.max_age_sec", 0.2},
    });
    return result;
  }

  void start(const std::string& mode = "partial") {
    fusion_ = std::make_shared<PointCloudFusion>(options(mode));
    executor_.add_node(fusion_);
    ASSERT_TRUE(spinUntil([this]() {
      return clock_pub_->get_subscription_count() > 0 && output_->get_publisher_count() > 0 &&
             inputs_[0]->get_subscription_count() > 0 && inputs_[1]->get_subscription_count() > 0;
    }));
    advance(1s);
  }

  bool spinUntil(const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate() && std::chrono::steady_clock::now() < deadline) {
      executor_.spin_some();
      std::this_thread::sleep_for(1ms);
    }
    return predicate();
  }

  void drain() {
    // Allow DDS delivery and executor work to settle without advancing ROS time.
    const auto deadline = std::chrono::steady_clock::now() + 30ms;
    while (std::chrono::steady_clock::now() < deadline) {
      executor_.spin_some();
      std::this_thread::sleep_for(1ms);
    }
  }

  void advance(std::chrono::nanoseconds time) {
    rosgraph_msgs::msg::Clock clock;
    clock.clock = rclcpp::Time(time.count());
    clock_pub_->publish(clock);
    ASSERT_TRUE(spinUntil([this, time]() { return fusion_->now().nanoseconds() == time.count(); }));
    drain();
  }

  void send(std::size_t topic_index, std::chrono::nanoseconds stamp, float id) {
    Cloud cloud;
    cloud.header.frame_id = "base_link";
    cloud.header.stamp = rclcpp::Time(stamp.count());
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(1);
    *sensor_msgs::PointCloud2Iterator<float>(cloud, "x") = id;
    *sensor_msgs::PointCloud2Iterator<float>(cloud, "y") = 1.0F;
    *sensor_msgs::PointCloud2Iterator<float>(cloud, "z") = 2.0F;
    inputs_.at(topic_index)->publish(cloud);
    drain();
  }

  static std::vector<float> pointIds(const Cloud& cloud) {
    std::vector<float> result;
    for (sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"); x != x.end(); ++x) {
      result.push_back(*x);
    }
    return result;
  }

  std::string namespace_;
  std::vector<std::string> topics_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Node::SharedPtr io_;
  std::shared_ptr<PointCloudFusion> fusion_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  std::vector<rclcpp::Publisher<Cloud>::SharedPtr> inputs_;
  rclcpp::Subscription<Cloud>::SharedPtr output_;
  std::vector<Cloud> outputs_;
};

TEST_F(PartialFusionTest, PublishesFreshBestEffortInputAtDeadlineWithoutReusingScans) {
  inputs_[1] = io_->create_publisher<Cloud>(topics_[1], rclcpp::SensorDataQoS());
  ASSERT_NO_FATAL_FAILURE(start());
  send(1, 700ms, 10.0F);  // A newly received cloud can already be stale.
  send(1, 1s, 11.0F);     // Configured input 0 never arrives.
  advance(1009ms);
  EXPECT_TRUE(outputs_.empty());
  advance(1011ms);
  ASSERT_EQ(outputs_.size(), 1U);
  EXPECT_EQ(pointIds(outputs_.back()), std::vector<float>{11.0F});
  EXPECT_EQ(outputs_.back().header.frame_id, "base_link");
  EXPECT_EQ(rclcpp::Time(outputs_.back().header.stamp).nanoseconds(), 1000000000LL);
  send(0, 1s, 12.0F);  // A late peer must not reopen the finalized cycle.
  advance(2s);
  EXPECT_EQ(outputs_.size(), 1U);
}

TEST_F(PartialFusionTest, RateLimitedBatchUsesNewInputsAndIncludesReturningSensor) {
  ASSERT_NO_FATAL_FAILURE(start());
  send(0, 1s, 1.0F);
  advance(1011ms);
  ASSERT_EQ(outputs_.size(), 1U);
  advance(1020ms);
  send(0, 1020ms, 2.0F);
  advance(1031ms);
  EXPECT_EQ(outputs_.size(), 1U);
  advance(1050ms);
  send(0, 1050ms, 3.0F);
  send(1, 1050ms, 4.0F);
  advance(1060ms);
  EXPECT_EQ(outputs_.size(), 1U);
  advance(1062ms);
  ASSERT_EQ(outputs_.size(), 2U);
  EXPECT_EQ(pointIds(outputs_.back()), (std::vector<float>{3.0F, 4.0F}));
}

TEST_F(PartialFusionTest, ClockJumpsClearPendingDataAndPublicationHistory) {
  ASSERT_NO_FATAL_FAILURE(start());
  send(0, 1s, 1.0F);
  advance(500ms);  // Jump backwards while a collection timer is pending.
  send(1, 500ms, 2.0F);
  advance(511ms);
  ASSERT_EQ(outputs_.size(), 1U);
  EXPECT_EQ(pointIds(outputs_.back()), std::vector<float>{2.0F});

  advance(250ms);  // Replay an earlier time after a completed publication.
  send(0, 250ms, 3.0F);
  advance(261ms);
  ASSERT_EQ(outputs_.size(), 2U);
  EXPECT_EQ(pointIds(outputs_.back()), std::vector<float>{3.0F});

  send(0, 261ms, 4.0F);
  advance(2s);  // Jump forwards while rate limited; expire the pending cloud.
  EXPECT_EQ(outputs_.size(), 2U);
  send(1, 2s, 5.0F);
  advance(2011ms);
  ASSERT_EQ(outputs_.size(), 3U);
  EXPECT_EQ(pointIds(outputs_.back()), std::vector<float>{5.0F});
}

TEST_F(PartialFusionTest, ApproximateModeStillRequiresEveryInput) {
  ASSERT_NO_FATAL_FAILURE(start("approximate"));
  send(0, 1s, 1.0F);
  advance(1100ms);
  EXPECT_TRUE(outputs_.empty());
  send(1, 1s, 2.0F);
  // Equal stamps form an immediately provable ApproximateTime candidate.
  ASSERT_TRUE(spinUntil([this]() { return !outputs_.empty(); }));
  EXPECT_EQ(pointIds(outputs_.back()), (std::vector<float>{1.0F, 2.0F}));
}

TEST_F(PartialFusionTest, InvalidPartialConfigurationsFailAtStartup) {
  const std::vector<rclcpp::Parameter> invalid_parameters{
      {"partial_sync.publish_rate_hz", 0.0},
      {"partial_sync.timeout_sec", -0.01},
      {"output_stamp_mode", "INPUT0"},
      {"sync_mode", "unknown"},
  };
  for (const auto& parameter : invalid_parameters) {
    SCOPED_TRACE(parameter.get_name());
    auto invalid = options();
    invalid.append_parameter_override(parameter.get_name(), parameter.get_parameter_value());
    EXPECT_THROW(std::make_shared<PointCloudFusion>(invalid), std::invalid_argument);
  }
}

}  // namespace
}  // namespace point_cloud_fusion
