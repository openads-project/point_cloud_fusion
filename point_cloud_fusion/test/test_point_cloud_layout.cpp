// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <point_cloud_fusion/point_cloud_layout.hpp>

namespace {

using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;

Field field(std::string name, uint32_t offset, uint8_t datatype = Field::FLOAT32, uint32_t count = 1) {
  Field result;
  result.name = std::move(name);
  result.offset = offset;
  result.datatype = datatype;
  result.count = count;
  return result;
}

Cloud::ConstSharedPtr cloud(uint32_t point_step, std::vector<Field> fields) {
  auto result = std::make_shared<Cloud>();
  result->header.frame_id = "base_link";
  result->height = 1;
  result->width = 1;
  result->point_step = point_step;
  result->row_step = point_step;
  result->fields = std::move(fields);
  result->data.resize(point_step);
  return result;
}

std::vector<Field> ousterFields() {
  return {field("x", 0),
          field("y", 4),
          field("z", 8),
          field("intensity", 12),
          field("t", 16, Field::UINT32),
          field("ring", 20, Field::UINT16)};
}

TEST(PointCloudLayout, HomogeneousCloudsKeepWholePointFastPath) {
  const auto first = cloud(24, ousterFields());
  const auto second = cloud(24, ousterFields());
  const auto layout = point_cloud_fusion::detail::buildBatchLayout({first, second}, {}, "t");
  EXPECT_TRUE(layout.homogeneous_fast_path);
  EXPECT_EQ(layout.point_step, 24U);
  EXPECT_TRUE(layout.inputs[0].whole_point_copy);
  EXPECT_TRUE(layout.inputs[1].whole_point_copy);
  EXPECT_EQ(layout.fields, first->fields);
}

TEST(PointCloudLayout, HeterogeneousOffsetsAndPointStepsGetIndependentPlans) {
  const auto ouster = cloud(24, ousterFields());
  const auto aeva = cloud(32, {field("intensity", 0), field("z", 8), field("x", 16), field("y", 24)});
  const auto layout =
      point_cloud_fusion::detail::buildBatchLayout({ouster, aeva}, {"x", "y", "z", "intensity", "t", "ring"}, "t");
  ASSERT_TRUE(layout.inputs[0].valid);
  ASSERT_TRUE(layout.inputs[1].valid);
  EXPECT_FALSE(layout.homogeneous_fast_path);
  EXPECT_EQ(layout.inputs[1].x_offset, 16U);
  EXPECT_EQ(layout.inputs[1].y_offset, 24U);
  EXPECT_EQ(layout.inputs[1].z_offset, 8U);
  EXPECT_EQ(layout.inputs[1].copies.size(), 4U);
}

TEST(PointCloudLayout, MissingOptionalFieldsAreMarkedForZeroFill) {
  const auto ouster = cloud(24, ousterFields());
  const auto aeva = cloud(16, {field("x", 0), field("y", 4), field("z", 8), field("intensity", 12)});
  const auto layout =
      point_cloud_fusion::detail::buildBatchLayout({ouster, aeva}, {"x", "y", "z", "intensity", "t", "ring"}, "t");
  EXPECT_NE(std::find(layout.inputs[1].zero_filled_fields.begin(), layout.inputs[1].zero_filled_fields.end(), "t"),
            layout.inputs[1].zero_filled_fields.end());
  EXPECT_NE(std::find(layout.inputs[1].zero_filled_fields.begin(), layout.inputs[1].zero_filled_fields.end(), "ring"),
            layout.inputs[1].zero_filled_fields.end());
}

TEST(PointCloudLayout, MissingXyzSkipsOnlyThatCloud) {
  const auto valid = cloud(16, {field("x", 0), field("y", 4), field("z", 8), field("intensity", 12)});
  const auto invalid = cloud(8, {field("intensity", 0), field("t", 4, Field::UINT32)});
  const auto layout = point_cloud_fusion::detail::buildBatchLayout({invalid, valid}, {}, "t");
  EXPECT_FALSE(layout.inputs[0].valid);
  EXPECT_TRUE(layout.inputs[1].valid);
  EXPECT_GT(layout.point_step, 0U);
}

TEST(PointCloudLayout, UnionOmitsIncompatibleSameNameFieldDeterministically) {
  const auto first = cloud(16, {field("x", 0), field("y", 4), field("z", 8), field("signal", 12)});
  const auto second =
      cloud(20, {field("x", 0), field("y", 4), field("z", 8), field("signal", 12, Field::UINT16), field("range", 16)});
  const auto layout = point_cloud_fusion::detail::buildBatchLayout({first, second}, {}, "t");
  ASSERT_EQ(layout.conflicting_fields, std::vector<std::string>{"signal"});
  EXPECT_EQ(std::count_if(layout.fields.begin(), layout.fields.end(), [](const auto& item) { return item.name == "signal"; }), 0);
  EXPECT_EQ(std::count_if(layout.fields.begin(), layout.fields.end(), [](const auto& item) { return item.name == "range"; }), 1);
}

}  // namespace
