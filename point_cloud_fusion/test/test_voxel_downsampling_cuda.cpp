// copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <point_cloud_fusion/point_cloud_fusion_cuda.hpp>
#include <point_cloud_fusion/voxel_downsampling.hpp>
#include <random>
#include <vector>

namespace {
using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;
using Context = point_cloud_fusion::cuda::CudaTransformContext;

Cloud cloud(const std::vector<std::array<float, 3>>& xyz) {
  Cloud result;
  result.width = xyz.size();
  result.height = 1;
  result.point_step = 37;
  result.row_step = result.width * result.point_step;
  result.fields = {Field().set__name("x").set__offset(4).set__datatype(Field::FLOAT32).set__count(1),
                   Field().set__name("y").set__offset(12).set__datatype(Field::FLOAT32).set__count(1),
                   Field().set__name("z").set__offset(20).set__datatype(Field::FLOAT32).set__count(1),
                   Field().set__name("metadata").set__offset(24).set__datatype(Field::UINT8).set__count(13)};
  result.data.resize(result.row_step);
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    for (std::size_t b = 0; b < result.point_step; ++b) result.data[i * result.point_step + b] = (i * 71 + b) % 256;
    for (std::size_t a = 0; a < 3; ++a) {
      std::memcpy(result.data.data() + i * result.point_step + 4 + a * 8, &xyz[i][a], sizeof(float));
    }
  }
  return result;
}

std::vector<std::vector<uint8_t>> sortedRecords(const Cloud& input) {
  std::vector<std::vector<uint8_t>> result;
  for (std::size_t i = 0; i < input.width; ++i) {
    const auto first = input.data.begin() + i * input.point_step;
    result.emplace_back(first, first + input.point_step);
  }
  std::sort(result.begin(), result.end());
  return result;
}

class CudaVoxelDownsampling : public testing::Test {
 protected:
  void SetUp() override {
    try {
      context_ = std::make_unique<Context>();
    } catch (const std::exception& error) {
      GTEST_SKIP() << "CUDA unavailable: " << error.what();
    }
  }

  bool process(const Cloud& input, bool enabled, double size, Cloud& output, std::size_t& fused, double& duration) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<point_cloud_fusion::cuda::CudaFieldCopy> copies{{0, 0, static_cast<int>(input.point_step)}};
    if (!context_->resetBatch(input.width, input.width, input.point_step, input.point_step, 4, 12, 20, 4, 12, 20, -1, copies,
                              -inf, inf, -inf, inf, -inf, inf, false))
      return false;
    const float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float translation[3] = {0, 0, 0};
    if (input.width && !context_->addCloud(input.data.data(), input.width, rotation, translation, false, 0)) return false;
    output = input;
    std::size_t count = 0;
    if (!context_->getBatchOutput(output.data, count, enabled, size, fused, duration)) return false;
    output.width = count;
    output.row_step = count * input.point_step;
    return true;
  }

  void compare(const Cloud& input, double size) {
    Cloud expected = input;
    point_cloud_fusion::detail::downsampleVoxels(expected, size);
    Cloud actual;
    std::size_t fused = 0;
    double duration = 0;
    ASSERT_TRUE(process(input, true, size, actual, fused, duration));
    EXPECT_EQ(fused, input.width);
    EXPECT_GE(duration, 0);
    EXPECT_EQ(sortedRecords(actual), sortedRecords(expected));
  }

  std::unique_ptr<Context> context_;
};

TEST_F(CudaVoxelDownsampling, MatchesCpuForBoundariesTiesAndUnalignedCompleteRecords) {
  const auto input = cloud({{.75F, .5F, .5F},
                            {.25F, .5F, .5F},
                            {.25F, .5F, .5F},
                            {-.5F, -.5F, -.5F},
                            {-1, 0, 0},
                            {0, 0, 0},
                            {1, 0, 0},
                            {-0.0F, .5F, .5F},
                            {0.0F, .5F, .5F}});
  for (double size : {.05, .1, .2, 1.0}) compare(input, size);
}

TEST_F(CudaVoxelDownsampling, MatchesCpuAcrossDenseSparseAndRepeatedBatches) {
  std::mt19937 random(381);
  std::uniform_real_distribution<float> coordinate(-2.0F, 2.0F);
  std::vector<std::array<float, 3>> xyz;
  for (int i = 0; i < 50000; ++i) xyz.push_back({coordinate(random), coordinate(random), coordinate(random)});
  for (double size : {.05, .1, .2, .05}) compare(cloud(xyz), size);
  compare(cloud({{10000, -10000, 20000}, {-10000, 20000, -30000}}), .1);
  xyz.assign(4096, {.25F, .5F, .5F});
  compare(cloud(xyz), 1.0);
  // exercise workspace reuse after shrinking and growing the record layout.
  auto larger = cloud(xyz);
  larger.point_step = 74;
  larger.row_step = larger.width * larger.point_step;
  larger.data.assign(larger.row_step, 0);
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    for (std::size_t a = 0; a < 3; ++a) std::memcpy(larger.data.data() + i * larger.point_step + 4 + a * 8, &xyz[i][a], 4);
  }
  compare(larger, 1.0);
  compare(cloud(xyz), .1);
}

TEST_F(CudaVoxelDownsampling, RejectsOverflowAndRecoversOnNextBatch) {
  Cloud output;
  std::size_t fused = 0;
  double duration = 0;
  auto input = cloud({{1, 2, 3}});
  EXPECT_FALSE(process(input, true, 1e-30, output, fused, duration));
  EXPECT_FALSE(process(input, true, 0, output, fused, duration));
  EXPECT_FALSE(process(input, true, std::numeric_limits<double>::quiet_NaN(), output, fused, duration));
  EXPECT_FALSE(process(cloud({{1, 0, 0}}), true, 0x1p-63, output, fused, duration));
  compare(cloud({{-1, 0, 0}}), 0x1p-63);
  compare(input, .1);
}

TEST_F(CudaVoxelDownsampling, DisabledAndEmptyBatchesPreserveCountsAndSkipFilter) {
  Cloud output;
  std::size_t fused = 0;
  double duration = -1;
  const auto input = cloud({{1, 2, 3}, {1, 2, 3}, {4, 5, 6}});
  ASSERT_TRUE(process(input, false, .1, output, fused, duration));
  EXPECT_EQ(sortedRecords(output), sortedRecords(input));
  EXPECT_EQ(fused, input.width);
  EXPECT_DOUBLE_EQ(duration, 0);
  ASSERT_TRUE(process(cloud({}), true, .1, output, fused, duration));
  EXPECT_TRUE(output.data.empty());
  EXPECT_EQ(output.width, 0U);
  EXPECT_EQ(fused, 0U);
  EXPECT_DOUBLE_EQ(duration, 0);
}
}  // namespace
