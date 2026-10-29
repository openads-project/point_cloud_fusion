// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <point_cloud_fusion/motion_compensation.hpp>
#include <point_cloud_fusion/object_motion_compensation.hpp>

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

TEST(ObjectMotion, DeskewedTranslationPreservesAcquisitionRay) {
  MotionTransform transform;
  transform.start_translation = {10.0F, 20.0F, 0.0F};
  transform.end_translation = {10.0F, 21.0F, 0.0F};
  transform.max_time_offset = 100000000;
  retainDeskewedPose(transform);
  EXPECT_EQ(transform.max_time_offset, 0U);
  EXPECT_EQ(transform.end_translation, transform.start_translation);

  ObjectMotion motion;
  motion.ray_origin_max_time_offset = transform.ray_origin_max_time_offset;
  motion.end_ray_origin = transform.end_ray_origin;
  for (const uint32_t point_time : {50000000U, 100000000U}) {
    float x = 10.0F, y = static_cast<float>(point_time) * 1.0e-8F, z = 0.0F;
    const float expected_y = y;
    EXPECT_TRUE(compensateObjectMotion(motion, 10.0F, point_time, x, y, z));
    EXPECT_NEAR(x, 10.0F - static_cast<float>(point_time) * 1.0e-8F, 1.0e-6F);
    EXPECT_FLOAT_EQ(y, expected_y);
  }
}

TEST(ObjectMotion, DeskewedOriginIsExpressedInHeaderOrientation) {
  MotionTransform transform;
  transform.start_translation = {10.0F, 20.0F, 0.0F};
  transform.end_translation = {10.0F, 21.0F, 0.0F};
  transform.start_quaternion = {0.0F, 0.0F, std::sqrt(0.5F), std::sqrt(0.5F)};
  transform.end_quaternion = transform.start_quaternion;
  transform.max_time_offset = 100000000;
  retainDeskewedPose(transform);
  EXPECT_NEAR(transform.end_ray_origin[0], 1.0F, 1.0e-6F);
  EXPECT_NEAR(transform.end_ray_origin[1], 0.0F, 1.0e-6F);
  EXPECT_NEAR(transform.end_ray_origin[2], 0.0F, 1.0e-6F);
}

TEST(ObjectMotion, DeskewedAndRawCorrectionAgreeWithTranslationAndRotation) {
  MotionTransform raw;
  raw.start_translation = {10.0F, 20.0F, 3.0F};
  raw.end_translation = {12.0F, 21.0F, 4.0F};
  raw.start_quaternion = {0.0F, 0.0F, std::sqrt(0.5F), std::sqrt(0.5F)};
  raw.end_quaternion = {0.0F, 0.0F, 1.0F, 0.0F};
  raw.max_time_offset = 100000000;
  auto deskewed = raw;
  retainDeskewedPose(deskewed);
  MotionTransform inverse_rotation;
  inverse_rotation.start_quaternion = {0.0F, 0.0F, -std::sqrt(0.5F), std::sqrt(0.5F)};
  for (const uint32_t time : {25000000U, 50000000U, 100000000U}) {
    float x, y, z;
    transformPointInterpolated(raw, time, 3.0F, 4.0F, 1.0F, x, y, z);
    transformPointInterpolated(inverse_rotation, 0, x - 10.0F, y - 20.0F, z - 3.0F, x, y, z);
    ObjectMotion motion;
    motion.ray_origin_max_time_offset = deskewed.ray_origin_max_time_offset;
    motion.end_ray_origin = deskewed.end_ray_origin;
    ASSERT_TRUE(compensateObjectMotion(motion, 5.0F, time, x, y, z));
    transformPointInterpolated(deskewed, time, x, y, z, x, y, z);
    float expected_x = 3.0F, expected_y = 4.0F, expected_z = 1.0F;
    ASSERT_TRUE(compensateObjectMotion(ObjectMotion{}, 5.0F, time, expected_x, expected_y, expected_z));
    transformPointInterpolated(raw, time, expected_x, expected_y, expected_z, expected_x, expected_y, expected_z);
    EXPECT_NEAR(x, expected_x, 1.0e-5F);
    EXPECT_NEAR(y, expected_y, 1.0e-5F);
    EXPECT_NEAR(z, expected_z, 1.0e-5F);
  }
}

TEST(ObjectMotion, MovesAlongOriginalObliqueRayInBothTimeDirections) {
  ObjectMotion motion;
  motion.reference_offset_sec = 0.1F;
  float x = 3.0F, y = 4.0F, z = 0.0F;
  EXPECT_TRUE(compensateObjectMotion(motion, 5.0F, 0, x, y, z));
  EXPECT_NEAR(x, 3.3F, 1.0e-6F);
  EXPECT_NEAR(y, 4.4F, 1.0e-6F);
  motion.reference_offset_sec = 0.0F;
  x = 3.0F;
  y = 4.0F;
  EXPECT_TRUE(compensateObjectMotion(motion, 5.0F, 100000000, x, y, z));
  EXPECT_NEAR(x, 2.7F, 1.0e-6F);
  EXPECT_NEAR(y, 3.6F, 1.0e-6F);
}

TEST(ObjectMotion, LeavesDegenerateRayAndUnknownMeasurementsUnchanged) {
  ObjectMotion motion;
  motion.reference_offset_sec = 0.1F;
  float x = 0.0F, y = 0.0F, z = 0.0F;
  EXPECT_FALSE(compensateObjectMotion(motion, 5.0F, 0, x, y, z));
  x = 10.0F;
  for (float velocity : {0.0F, 101.0F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    EXPECT_FALSE(compensateObjectMotion(motion, velocity, 0, x, y, z));
    EXPECT_FLOAT_EQ(x, 10.0F);
  }
}

TEST(ObjectMotion, IncludesExactLimitsAndRejectsValuesImmediatelyOutside) {
  ObjectMotion motion;
  motion.max_time_offset_sec = 0.125F;
  motion.max_speed_mps = 64.0F;
  for (const float direction : {-1.0F, 1.0F}) {
    motion.reference_offset_sec = direction * motion.max_time_offset_sec;
    float x = 10.0F, y = 0.0F, z = 0.0F;
    EXPECT_TRUE(compensateObjectMotion(motion, direction * 64.0F, 0, x, y, z));
    EXPECT_FLOAT_EQ(x, 18.0F);
    x = 10.0F;
    EXPECT_FALSE(compensateObjectMotion(motion, direction * std::nextafter(64.0F, 65.0F), 0, x, y, z));
    EXPECT_FLOAT_EQ(x, 10.0F);
    motion.reference_offset_sec = direction * std::nextafter(0.125F, 1.0F);
    EXPECT_FALSE(compensateObjectMotion(motion, 5.0F, 0, x, y, z));
    EXPECT_FLOAT_EQ(x, 10.0F);
  }
  for (float dt : {0.0F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    motion.reference_offset_sec = dt;
    float x = 10.0F, y = 0.0F, z = 0.0F;
    EXPECT_FALSE(compensateObjectMotion(motion, 5.0F, 0, x, y, z));
    EXPECT_FLOAT_EQ(x, 10.0F);
  }
}

TEST(ObjectMotion, MatchesDoublePrecisionReferenceAcrossRaysSignsAndTimeScales) {
  std::mt19937 random(1701);
  std::uniform_real_distribution<float> coordinate(-100.0F, 100.0F);
  std::uniform_real_distribution<float> speed(-100.0F, 100.0F);
  std::uniform_real_distribution<float> reference(-0.05F, 0.05F);
  std::uniform_int_distribution<uint32_t> time(0, 100000);
  for (const float time_scale : {1.0e-9F, 1.0e-6F}) {
    for (const float sign : {-1.0F, 1.0F}) {
      for (int sample = 0; sample < 1000; ++sample) {
        ObjectMotion motion;
        motion.time_scale_sec = time_scale;
        motion.doppler_sign = sign;
        motion.reference_offset_sec = reference(random);
        const auto point_time = time(random);
        const float velocity = speed(random);
        float x = coordinate(random), y = coordinate(random), z = coordinate(random);
        const std::array<double, 3> ray{x, y, z};
        const double range = std::sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
        const double dt = static_cast<double>(motion.reference_offset_sec) - static_cast<double>(point_time) * time_scale;
        ASSERT_TRUE(compensateObjectMotion(motion, velocity, point_time, x, y, z));
        const std::array<float, 3> actual{x, y, z};
        for (std::size_t axis = 0; axis < ray.size(); ++axis) {
          const double expected = ray[axis] + sign * velocity * dt * ray[axis] / range;
          EXPECT_NEAR(actual[axis], expected, 3.0e-5) << "sample=" << sample << " axis=" << axis;
        }
      }
    }
  }
}

}  // namespace
}  // namespace point_cloud_fusion
