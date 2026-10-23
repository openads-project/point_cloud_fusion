// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <thread>

#include <gtest/gtest.h>

#include <point_cloud_fusion/feature_calibration.hpp>

namespace point_cloud_fusion {
namespace {

sensor_msgs::msg::PointCloud2::SharedPtr makeFloatCloud(const std::vector<float>& values) {
  auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  cloud->height = 1;
  cloud->width = values.size();
  cloud->point_step = 4 * sizeof(float);
  cloud->row_step = cloud->width * cloud->point_step;
  for (std::size_t index = 0; index < 4; ++index) {
    sensor_msgs::msg::PointField field;
    field.name = std::vector<std::string>{"x", "y", "z", "reflectivity"}[index];
    field.offset = index * sizeof(float);
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    cloud->fields.push_back(field);
  }
  cloud->data.resize(values.size() * cloud->point_step);
  for (std::size_t point = 0; point < values.size(); ++point) {
    const std::array<float, 4> record{1.0F, 2.0F, 3.0F, values[point]};
    std::memcpy(cloud->data.data() + point * cloud->point_step, record.data(), cloud->point_step);
  }
  return cloud;
}

TEST(FeatureCalibration, QuantileMappingMatchesReferenceDistribution) {
  const auto mapping = DistributionFeatureCalibrator::buildMapping({0.0F, 10.0F, 20.0F, 30.0F, 40.0F},
                                                                   {100.0F, 120.0F, 140.0F, 160.0F, 180.0F}, 4);
  ASSERT_TRUE(mapping);
  EXPECT_FLOAT_EQ(mapping->apply(0.0F), 100.0F);
  EXPECT_FLOAT_EQ(mapping->apply(15.0F), 130.0F);
  EXPECT_FLOAT_EQ(mapping->apply(40.0F), 180.0F);
}

TEST(FeatureCalibration, MappingIsMonotonicAndClampsOutliers) {
  const auto mapping = DistributionFeatureCalibrator::buildMapping({0.0F, 0.0F, 10.0F, 20.0F}, {5.0F, 7.0F, 9.0F, 11.0F}, 8);
  ASSERT_TRUE(mapping);
  EXPECT_LE(mapping->apply(1.0F), mapping->apply(10.0F));
  EXPECT_FLOAT_EQ(mapping->apply(-100.0F), mapping->target.front());
  EXPECT_FLOAT_EQ(mapping->apply(100.0F), mapping->target.back());
}

TEST(FeatureCalibration, TailDenseKnotsPreserveResolutionBelowExtremeOutlier) {
  std::vector<float> source;
  std::vector<float> target;
  for (int value = 0; value < 1000; ++value) {
    source.push_back(static_cast<float>(value));
    target.push_back(static_cast<float>(value));
  }
  source.push_back(1000000.0F);
  target.push_back(2000.0F);

  const auto mapping = DistributionFeatureCalibrator::buildMapping(std::move(source), std::move(target), 32);

  ASSERT_TRUE(mapping);
  EXPECT_NEAR(mapping->apply(990.0F), 990.0F, 5.0F);
}

TEST(FeatureCalibration, ScalarConversionSaturatesIntegerOutput) {
  std::uint8_t bytes[4]{};
  ASSERT_TRUE(writeScalarField(bytes, sensor_msgs::msg::PointField::UINT16, false, 70000.0F));
  float value = 0.0F;
  ASSERT_TRUE(readScalarField(bytes, sensor_msgs::msg::PointField::UINT16, false, value));
  EXPECT_FLOAT_EQ(value, 65535.0F);
}

TEST(FeatureCalibration, WorkerPublishesFollowerMappingAsynchronously) {
  DistributionFeatureCalibrator::Config config;
  config.field = "reflectivity";
  config.leaders = {0};
  config.followers = {1};
  config.input_count = 3;
  config.samples_per_cloud = 4;
  config.window_samples = 16;
  config.minimum_samples = 4;
  config.quantile_count = 4;
  config.update_interval_sec = 0.001;
  DistributionFeatureCalibrator calibrator(config);
  const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> clouds{makeFloatCloud({100.0F, 120.0F, 140.0F, 160.0F}),
                                                                          makeFloatCloud({0.0F, 10.0F, 20.0F, 30.0F}),
                                                                          makeFloatCloud({500.0F, 500.0F, 500.0F, 500.0F})};

  std::shared_ptr<const FeatureMapping> mapping;
  for (int attempt = 0; attempt < 100 && !mapping; ++attempt) {
    calibrator.enqueue(clouds);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mapping = calibrator.mappings()[1];
  }

  ASSERT_TRUE(mapping);
  EXPECT_FLOAT_EQ(mapping->apply(0.0F), 100.0F);
  EXPECT_FLOAT_EQ(mapping->apply(30.0F), 160.0F);
  EXPECT_EQ(calibrator.mappings()[0], nullptr);
  EXPECT_EQ(calibrator.mappings()[2], nullptr);
}

TEST(FeatureCalibration, WorkerExcludesPointsWithInvalidCoordinates) {
  DistributionFeatureCalibrator::Config config;
  config.field = "reflectivity";
  config.leaders = {0};
  config.followers = {1};
  config.input_count = 2;
  config.samples_per_cloud = 4;
  config.window_samples = 16;
  config.minimum_samples = 2;
  config.quantile_count = 2;
  config.update_interval_sec = 0.001;
  DistributionFeatureCalibrator calibrator(config);

  auto leader = makeFloatCloud({0.0F, 0.0F, 100.0F, 200.0F});
  const float invalid_coordinate = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(leader->data.data(), &invalid_coordinate, sizeof(invalid_coordinate));
  std::memcpy(std::next(leader->data.data(), static_cast<std::ptrdiff_t>(leader->point_step)), &invalid_coordinate,
              sizeof(invalid_coordinate));
  const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr> clouds{leader, makeFloatCloud({10.0F, 20.0F, 30.0F, 40.0F})};

  std::shared_ptr<const FeatureMapping> mapping;
  for (int attempt = 0; attempt < 100 && !mapping; ++attempt) {
    calibrator.enqueue(clouds);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    mapping = calibrator.mappings()[1];
  }

  ASSERT_TRUE(mapping);
  EXPECT_FLOAT_EQ(mapping->target.front(), 100.0F);
  EXPECT_FLOAT_EQ(mapping->target.back(), 200.0F);
}

}  // namespace
}  // namespace point_cloud_fusion
