// copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <point_cloud_fusion/voxel_downsampling.hpp>

namespace {
using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;

Cloud cloud(const std::vector<std::array<float, 3>>& xyz, bool bigendian = false) {
  Cloud result;
  result.header.frame_id = "target";
  result.header.stamp.sec = 123;
  result.height = 1;
  result.width = xyz.size();
  result.point_step = 16;
  result.row_step = result.width * result.point_step;
  result.is_bigendian = bigendian;
  result.is_dense = false;
  result.fields = {Field().set__name("x").set__offset(0).set__datatype(Field::FLOAT32).set__count(1),
                   Field().set__name("y").set__offset(4).set__datatype(Field::FLOAT32).set__count(1),
                   Field().set__name("z").set__offset(8).set__datatype(Field::FLOAT32).set__count(1),
                   Field().set__name("metadata").set__offset(12).set__datatype(Field::UINT8).set__count(4)};
  result.data.resize(result.row_step);
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    for (std::size_t a = 0; a < 3; ++a) {
      std::array<uint8_t, 4> bytes;
      std::memcpy(bytes.data(), &xyz[i][a], 4);
      if (bigendian) std::reverse(bytes.begin(), bytes.end());
      std::memcpy(result.data.data() + i * 16 + a * 4, bytes.data(), 4);
    }
    std::fill_n(result.data.data() + i * 16 + 12, 4, static_cast<uint8_t>(i));
  }
  return result;
}

TEST(VoxelDownsampling, NearestCenterPreservesRecordsOrderAndMetadata) {
  for (bool bigendian : {false, true}) {
    auto input = cloud({{1.5F, .5F, .5F}, {.1F, .5F, .5F}, {.5F, .5F, .5F}, {1.9F, .5F, .5F}}, bigendian);
    const auto original = input;
    point_cloud_fusion::detail::downsampleVoxels(input, 1.0);
    EXPECT_EQ(input.header, original.header);
    EXPECT_EQ(input.fields, original.fields);
    EXPECT_EQ(input.point_step, original.point_step);
    EXPECT_EQ(input.is_bigendian, bigendian);
    EXPECT_EQ(input.is_dense, original.is_dense);
    ASSERT_EQ(input.width, 2U);
    EXPECT_EQ(input.height, 1U);
    EXPECT_EQ(input.row_step, 32U);
    ASSERT_EQ(input.data.size(), 32U);
    EXPECT_EQ(std::memcmp(input.data.data(), original.data.data(), 16), 0);
    EXPECT_EQ(std::memcmp(input.data.data() + 16, original.data.data() + 32, 16), 0);
  }
}

TEST(VoxelDownsampling, NegativeCellsAndExactBoundariesUseFloor) {
  auto input = cloud({{-1.0F, 0, 0}, {-.5F, .5F, .5F}, {0, 0, 0}, {.5F, .5F, .5F}, {1, 0, 0}});
  point_cloud_fusion::detail::downsampleVoxels(input, 1.0);
  ASSERT_EQ(input.width, 3U);
  EXPECT_EQ(input.data[12], 1U);
  EXPECT_EQ(input.data[28], 3U);
  EXPECT_EQ(input.data[44], 4U);
}

TEST(VoxelDownsampling, TiesUseCoordinatesThenRecordBytes) {
  auto input = cloud({{.75F, .5F, .5F}, {.25F, .5F, .5F}, {.25F, .5F, .5F}});
  input.data[16 + 12] = 9;
  input.data[32 + 12] = 3;
  const auto original = input;
  point_cloud_fusion::detail::downsampleVoxels(input, 1.0);
  ASSERT_EQ(input.width, 1U);
  EXPECT_EQ(std::memcmp(input.data.data(), original.data.data() + 32, 16), 0);
}

TEST(VoxelDownsampling, RejectsInvalidSizesAndOverflowWithoutMutating) {
  auto input = cloud({{1, 2, 3}});
  const auto original = input;
  for (double size : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(point_cloud_fusion::detail::downsampleVoxels(input, size), std::invalid_argument);
  }
  EXPECT_THROW(point_cloud_fusion::detail::downsampleVoxels(input, 1e-30), std::overflow_error);
  EXPECT_EQ(input, original);
  input = cloud({{0, 0, 0}, {std::numeric_limits<float>::infinity(), 0, 0}});
  const auto nonfinite = input;
  EXPECT_THROW(point_cloud_fusion::detail::downsampleVoxels(input, 1), std::overflow_error);
  EXPECT_EQ(input, nonfinite);
}

TEST(VoxelDownsampling, EmptyCloudRetainsLayout) {
  auto input = cloud({});
  const auto original = input;
  point_cloud_fusion::detail::downsampleVoxels(input, .1);
  EXPECT_EQ(input, original);
}
}  // namespace
