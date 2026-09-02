// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <point_cloud_fusion/motion_compensation.hpp>

namespace point_cloud_fusion {
namespace {

struct TestCloud {
  std::size_t point_step;
  std::vector<int> fields;
};

TEST(MotionCompensation, ExcludesIncompatibleCloudsWithoutChangingBatchIndices) {
  using CloudPtr = std::shared_ptr<const TestCloud>;
  const std::vector<int> reference_fields{1, 2, 3};
  const CloudPtr compatible = std::make_shared<TestCloud>(TestCloud{16, reference_fields});
  const CloudPtr incompatible_point_step = std::make_shared<TestCloud>(TestCloud{20, reference_fields});
  const CloudPtr incompatible_fields = std::make_shared<TestCloud>(TestCloud{16, std::vector<int>{1, 2}});
  const std::vector<CloudPtr> msgs{compatible, incompatible_point_step, nullptr, incompatible_fields};

  const auto motion_msgs = selectMotionCompatibleClouds(msgs, 16, reference_fields);

  ASSERT_EQ(motion_msgs.size(), msgs.size());
  EXPECT_EQ(motion_msgs[0], compatible);
  EXPECT_EQ(motion_msgs[1], nullptr);
  EXPECT_EQ(motion_msgs[2], nullptr);
  EXPECT_EQ(motion_msgs[3], nullptr);
}

TEST(MotionCompensation, InterpolatesTranslationAtPointTime) {
  MotionTransform transform;
  transform.end_translation[0] = 10.0F;
  transform.max_time_offset = 100;

  float x, y, z;
  transformPointInterpolated(transform, 25, 1.0F, 2.0F, 3.0F, x, y, z);

  EXPECT_FLOAT_EQ(x, 3.5F);
  EXPECT_FLOAT_EQ(y, 2.0F);
  EXPECT_FLOAT_EQ(z, 3.0F);
}

TEST(MotionCompensation, InterpolatesRotationAtPointTime) {
  MotionTransform transform;
  // Identity to 180 degrees around Z; normalized lerp is 90 degrees halfway.
  transform.end_quaternion[2] = 1.0F;
  transform.end_quaternion[3] = 0.0F;
  transform.max_time_offset = 100;

  float x, y, z;
  transformPointInterpolated(transform, 50, 1.0F, 0.0F, 0.0F, x, y, z);

  EXPECT_NEAR(x, 0.0F, 1.0e-5F);
  EXPECT_NEAR(y, 1.0F, 1.0e-5F);
  EXPECT_NEAR(z, 0.0F, 1.0e-5F);
}

TEST(MotionCompensation, ClampsOffsetsBeyondScanEnd) {
  MotionTransform transform;
  transform.end_translation[1] = 4.0F;
  transform.max_time_offset = 100;

  float x, y, z;
  transformPointInterpolated(transform, 150, 0.0F, 0.0F, 0.0F, x, y, z);

  EXPECT_FLOAT_EQ(x, 0.0F);
  EXPECT_FLOAT_EQ(y, 4.0F);
  EXPECT_FLOAT_EQ(z, 0.0F);
}

TEST(MotionCompensation, UsesStartPoseWithoutPointTiming) {
  MotionTransform transform;
  transform.start_translation[2] = 2.0F;
  transform.end_translation[2] = 8.0F;

  float x, y, z;
  transformPointInterpolated(transform, 0, 0.0F, 0.0F, 1.0F, x, y, z);

  EXPECT_FLOAT_EQ(x, 0.0F);
  EXPECT_FLOAT_EQ(y, 0.0F);
  EXPECT_FLOAT_EQ(z, 3.0F);
}

}  // namespace
}  // namespace point_cloud_fusion
