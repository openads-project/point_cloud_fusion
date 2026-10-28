// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
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

TEST(PointCloudLayout, CalibrationUsesLeaderDatatypeAndConvertsFollowerMeasurements) {
  const auto leader = cloud(16, {field("x", 0), field("y", 4), field("z", 8), field("reflectivity", 12, Field::UINT16)});
  const auto follower = cloud(20, {field("reflectivity", 0), field("x", 4), field("y", 8), field("z", 12)});
  const auto* descriptor = point_cloud_fusion::detail::findField(*leader, "reflectivity");
  for (const bool leader_first : {false, true}) {
    const std::vector<Cloud::ConstSharedPtr> inputs = leader_first ? std::vector<Cloud::ConstSharedPtr>{leader, follower}
                                                                   : std::vector<Cloud::ConstSharedPtr>{follower, leader};
    for (const auto& selection : {std::vector<std::string>{}, std::vector<std::string>{"reflectivity", "x", "y", "z"}}) {
      const auto layout =
          point_cloud_fusion::detail::buildBatchLayout(inputs, selection, "t", {{"reflectivity", 65535}}, descriptor);
      ASSERT_TRUE(layout.fill_error.empty());
      EXPECT_TRUE(layout.conflicting_fields.empty());
      const auto output =
          std::find_if(layout.fields.begin(), layout.fields.end(), [](const auto& f) { return f.name == "reflectivity"; });
      ASSERT_NE(output, layout.fields.end());
      EXPECT_EQ(output->datatype, Field::UINT16);
      const auto& reference = layout.inputs[leader_first ? 0 : 1];
      const auto& converted = layout.inputs[leader_first ? 1 : 0];
      EXPECT_TRUE(reference.conversions.empty());
      EXPECT_TRUE(reference.field_fills.empty());
      ASSERT_EQ(converted.conversions.size(), 1U);
      EXPECT_EQ(converted.conversions[0].source_offset, 0U);
      EXPECT_EQ(converted.conversions[0].destination_offset, output->offset);
      EXPECT_EQ(converted.conversions[0].source_datatype, Field::FLOAT32);
      EXPECT_EQ(converted.conversions[0].destination_datatype, Field::UINT16);
      EXPECT_TRUE(converted.field_fills.empty());
    }
  }
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

TEST(PointCloudLayout, MissingFloatingFieldsDefaultToEndianCorrectNaNForEveryElement) {
  for (const auto& name : {std::string("velocity"), std::string("v"), std::string("temperature")}) {
    for (const auto datatype : {Field::FLOAT32, Field::FLOAT64}) {
      for (const uint32_t count : {1U, 3U}) {
        for (const bool bigendian : {false, true}) {
          SCOPED_TRACE(testing::Message() << "field=" << name << " datatype=" << static_cast<int>(datatype) << " count=" << count
                                          << " bigendian=" << bigendian);
          const auto element_size = point_cloud_fusion::detail::fieldDatatypeSize(datatype);
          auto missing = std::make_shared<Cloud>(*cloud(12, {field("x", 0), field("y", 4), field("z", 8)}));
          auto measured = std::make_shared<Cloud>(
              *cloud(12 + element_size * count, {field("x", 0), field("y", 4), field("z", 8), field(name, 12, datatype, count)}));
          missing->is_dense = measured->is_dense = true;
          missing->is_bigendian = measured->is_bigendian = bigendian;
          const auto layout = point_cloud_fusion::detail::buildBatchLayout({missing, measured}, {}, "t");
          ASSERT_EQ(layout.inputs[0].field_fills.size(), 1U);
          EXPECT_TRUE(layout.inputs[0].zero_filled_fields.empty());
          EXPECT_TRUE(layout.inputs[1].field_fills.empty());
          EXPECT_FALSE(layout.is_dense);
          std::vector<uint8_t> point(layout.point_step, 0);
          point_cloud_fusion::detail::fillMissingFields(point.data(), layout.inputs[0]);
          const uint16_t endian_check = 1;
          const bool host_bigendian = *reinterpret_cast<const uint8_t*>(&endian_check) == 0;
          for (uint32_t element = 0; element < count; ++element) {
            auto bytes = std::vector<uint8_t>(point.begin() + 12 + element * element_size,
                                              point.begin() + 12 + (element + 1) * element_size);
            if (bigendian != host_bigendian) std::reverse(bytes.begin(), bytes.end());
            if (datatype == Field::FLOAT32) {
              float value;
              std::memcpy(&value, bytes.data(), sizeof(value));
              EXPECT_TRUE(std::isnan(value));
            } else {
              double value;
              std::memcpy(&value, bytes.data(), sizeof(value));
              EXPECT_TRUE(std::isnan(value));
            }
          }
        }
      }
    }
  }
}

TEST(PointCloudLayout, MissingIntegersDefaultToZeroAndFloatsToNaN) {
  auto missing = std::make_shared<Cloud>(*cloud(12, {field("x", 0), field("y", 4), field("z", 8)}));
  auto measured =
      std::make_shared<Cloud>(*cloud(22, {field("x", 0), field("y", 4), field("z", 8), field("velocity", 12, Field::INT16),
                                          field("intensity", 14), field("velocity_uncertainty", 18)}));
  missing->is_dense = measured->is_dense = true;
  const auto layout = point_cloud_fusion::detail::buildBatchLayout({missing, measured}, {}, "t");
  EXPECT_FALSE(layout.is_dense);
  EXPECT_EQ(layout.inputs[0].field_fills.size(), 2U);
  EXPECT_EQ(layout.inputs[0].zero_filled_fields, (std::vector<std::string>{"velocity"}));
}

TEST(PointCloudLayout, NonDenseInputIsPropagatedThroughHomogeneousFastPath) {
  auto first = std::make_shared<Cloud>(*cloud(24, ousterFields()));
  auto second = std::make_shared<Cloud>(*cloud(24, ousterFields()));
  first->is_dense = second->is_dense = true;
  EXPECT_TRUE(point_cloud_fusion::detail::buildBatchLayout({first, second}, {}, "t").is_dense);
  second->is_dense = false;
  const auto layout = point_cloud_fusion::detail::buildBatchLayout({first, second}, {}, "t");
  EXPECT_TRUE(layout.homogeneous_fast_path);
  EXPECT_FALSE(layout.is_dense);
}

TEST(PointCloudLayout, FillPolicyParserValidatesEntries) {
  using point_cloud_fusion::detail::parseMissingFieldFillValues;
  const auto values = parseMissingFieldFillValues({"velocity=nan", " v = -2.5 ", "ring=65535", "ambient=1e2"});
  ASSERT_EQ(values.size(), 4U);
  EXPECT_TRUE(std::isnan(values.at("velocity")));
  EXPECT_DOUBLE_EQ(values.at("v"), -2.5);
  EXPECT_DOUBLE_EQ(values.at("ring"), 65535);
  EXPECT_DOUBLE_EQ(values.at("ambient"), 100);
  EXPECT_TRUE(parseMissingFieldFillValues({}).empty());
  for (const auto* entry : {"velocity", "=nan", "v=", "v=unknown", "v=1extra", "v=inf", "v=-inf", "v=1e999"}) {
    EXPECT_THROW(parseMissingFieldFillValues({entry}), std::invalid_argument) << entry;
  }
  EXPECT_THROW(parseMissingFieldFillValues({"v=0", " v=nan"}), std::invalid_argument);
}

template <typename T>
double decodeFillElement(const std::vector<uint8_t>& bytes, std::size_t offset, bool reverse) {
  std::vector<uint8_t> element(bytes.begin() + offset, bytes.begin() + offset + sizeof(T));
  if (reverse) std::reverse(element.begin(), element.end());
  T value;
  std::memcpy(&value, element.data(), sizeof(value));
  return static_cast<double>(value);
}

TEST(PointCloudLayout, NumericFillsSupportAllDatatypesArraysAndByteOrders) {
  const std::vector<std::pair<uint8_t, double>> cases{{Field::INT8, -128},          {Field::INT8, 127},
                                                      {Field::UINT8, 255},          {Field::INT16, -32768},
                                                      {Field::UINT16, 65535},       {Field::INT32, -2147483648.0},
                                                      {Field::INT32, 2147483647.0}, {Field::UINT32, 4294967295.0},
                                                      {Field::FLOAT32, -12.25},     {Field::FLOAT64, -1.5e200}};
  const uint16_t endian_check = 1;
  const bool host_bigendian = *reinterpret_cast<const uint8_t*>(&endian_check) == 0;
  for (const auto& item : cases) {
    for (const bool bigendian : {false, true}) {
      SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(item.first) << " value=" << item.second);
      const auto fill = point_cloud_fusion::detail::makeFieldFill(field("custom", 12, item.first, 3), item.second, bigendian);
      ASSERT_EQ(fill.destination_offset, 12U);
      const auto size = point_cloud_fusion::detail::fieldDatatypeSize(item.first);
      ASSERT_EQ(fill.bytes.size(), 3 * size);
      for (std::size_t element = 0; element < 3; ++element) {
        double actual = 0;
        const auto offset = element * size;
        const auto reverse = bigendian != host_bigendian;
        switch (item.first) {
          case Field::INT8:
            actual = decodeFillElement<int8_t>(fill.bytes, offset, reverse);
            break;
          case Field::UINT8:
            actual = decodeFillElement<uint8_t>(fill.bytes, offset, reverse);
            break;
          case Field::INT16:
            actual = decodeFillElement<int16_t>(fill.bytes, offset, reverse);
            break;
          case Field::UINT16:
            actual = decodeFillElement<uint16_t>(fill.bytes, offset, reverse);
            break;
          case Field::INT32:
            actual = decodeFillElement<int32_t>(fill.bytes, offset, reverse);
            break;
          case Field::UINT32:
            actual = decodeFillElement<uint32_t>(fill.bytes, offset, reverse);
            break;
          case Field::FLOAT32:
            actual = decodeFillElement<float>(fill.bytes, offset, reverse);
            break;
          case Field::FLOAT64:
            actual = decodeFillElement<double>(fill.bytes, offset, reverse);
            break;
        }
        EXPECT_DOUBLE_EQ(actual, item.second);
      }
    }
  }
}

TEST(PointCloudLayout, UnrepresentableFillRejectsLayoutWithoutThrowing) {
  const std::vector<std::pair<uint8_t, double>> cases{{Field::INT8, -129},
                                                      {Field::INT8, 128},
                                                      {Field::UINT8, -1},
                                                      {Field::UINT8, 256},
                                                      {Field::INT16, 32768},
                                                      {Field::UINT16, 65536},
                                                      {Field::INT32, -2147483649.0},
                                                      {Field::INT32, 2147483648.0},
                                                      {Field::UINT32, 4294967296.0},
                                                      {Field::UINT32, 1.5},
                                                      {Field::UINT16, std::numeric_limits<double>::quiet_NaN()},
                                                      {Field::FLOAT32, 1e100}};
  for (const auto& item : cases) {
    const auto missing = cloud(12, {field("x", 0), field("y", 4), field("z", 8)});
    const auto measured = cloud(20, {field("x", 0), field("y", 4), field("z", 8), field("custom", 12, item.first)});
    const auto layout = point_cloud_fusion::detail::buildBatchLayout({missing, measured}, {}, "t", {{"custom", item.second}});
    EXPECT_NE(layout.fill_error.find("custom"), std::string::npos);
  }
}

TEST(PointCloudLayout, FillPoliciesOnlyAffectMissingFieldsAndDoNotCreateFields) {
  const auto input = cloud(24, ousterFields());
  const auto layout = point_cloud_fusion::detail::buildBatchLayout(
      {input, input}, {}, "t", {{"ring", std::numeric_limits<double>::quiet_NaN()}, {"absent", -999}});
  EXPECT_TRUE(layout.fill_error.empty());
  EXPECT_TRUE(layout.homogeneous_fast_path);
  EXPECT_EQ(layout.fields, input->fields);
  EXPECT_TRUE(layout.inputs[0].field_fills.empty());
  EXPECT_EQ(layout.unused_fill_fields, (std::vector<std::string>{"absent", "ring"}));
}

TEST(PointCloudLayout, OverrideIsUsedWhenAnyValidInputNeedsItsFill) {
  const auto measured = cloud(16, {field("x", 0), field("y", 4), field("z", 8), field("velocity", 12)});
  const auto missing = cloud(12, {field("x", 0), field("y", 4), field("z", 8)});
  for (const auto& inputs :
       {std::vector<Cloud::ConstSharedPtr>{measured, missing}, std::vector<Cloud::ConstSharedPtr>{missing, measured}}) {
    const auto layout = point_cloud_fusion::detail::buildBatchLayout(inputs, {}, "t", {{"velocity", -999}});
    EXPECT_TRUE(layout.fill_error.empty());
    EXPECT_TRUE(layout.unused_fill_fields.empty());
  }
}

TEST(PointCloudLayout, OverrideForAnUnpublishedFieldIsUnusedEvenWhenAnInputLacksIt) {
  const auto measured = cloud(16, {field("x", 0), field("y", 4), field("z", 8), field("velocity", 12)});
  const auto missing = cloud(12, {field("x", 0), field("y", 4), field("z", 8)});
  const auto layout =
      point_cloud_fusion::detail::buildBatchLayout({measured, missing}, {"x", "y", "z"}, "t", {{"velocity", -999}});
  EXPECT_EQ(layout.unused_fill_fields, (std::vector<std::string>{"velocity"}));
  EXPECT_TRUE(layout.inputs[0].field_fills.empty());
  EXPECT_TRUE(layout.inputs[1].field_fills.empty());
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

TEST(PointCloudLayout, AddsHiddenTimeFieldForMotionCompensation) {
  const std::vector<std::string> requested{"x", "y", "z", "intensity"};
  const auto processing = point_cloud_fusion::detail::processingFields(requested, "t", true);
  EXPECT_EQ(processing, (std::vector<std::string>{"x", "y", "z", "intensity", "t"}));
  EXPECT_FALSE(point_cloud_fusion::detail::fieldIsPublished(requested, "t"));
}

TEST(PointCloudLayout, DoesNotAddTimeFieldWhenMotionCompensationIsDisabled) {
  const std::vector<std::string> requested{"x", "y", "z"};
  EXPECT_EQ(point_cloud_fusion::detail::processingFields(requested, "t", false), requested);
  EXPECT_TRUE(point_cloud_fusion::detail::processingFields({}, "t", true).empty());
}

}  // namespace
