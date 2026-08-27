// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace point_cloud_fusion {

/**
 * @brief Endpoint transforms for fast scan motion interpolation.
 *
 * Quaternions use x/y/z/w order. The end quaternion is kept in the same
 * hemisphere as the start quaternion by the TF preparation code.
 */
struct MotionTransform {
  std::array<float, 3> start_translation{0.0F, 0.0F, 0.0F};
  std::array<float, 3> end_translation{0.0F, 0.0F, 0.0F};
  std::array<float, 4> start_quaternion{0.0F, 0.0F, 0.0F, 1.0F};
  std::array<float, 4> end_quaternion{0.0F, 0.0F, 0.0F, 1.0F};
  uint32_t max_time_offset{0};
};

/**
 * @brief Transform a point using normalized quaternion interpolation.
 */
inline void transformPointInterpolated(const MotionTransform& transform,
                                       uint32_t time_offset,
                                       float x,
                                       float y,
                                       float z,
                                       float& transformed_x,
                                       float& transformed_y,
                                       float& transformed_z) {
  const float alpha = transform.max_time_offset > 0
                          ? std::min(1.0F, static_cast<float>(time_offset) / static_cast<float>(transform.max_time_offset))
                          : 0.0F;
  const float inverse_alpha = 1.0F - alpha;

  float qx = inverse_alpha * transform.start_quaternion[0] + alpha * transform.end_quaternion[0];
  float qy = inverse_alpha * transform.start_quaternion[1] + alpha * transform.end_quaternion[1];
  float qz = inverse_alpha * transform.start_quaternion[2] + alpha * transform.end_quaternion[2];
  float qw = inverse_alpha * transform.start_quaternion[3] + alpha * transform.end_quaternion[3];
  const float inverse_norm = 1.0F / std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  qx *= inverse_norm;
  qy *= inverse_norm;
  qz *= inverse_norm;
  qw *= inverse_norm;

  // Rotate with q * p * conjugate(q), avoiding temporary tf2 objects.
  const float tx = 2.0F * (qy * z - qz * y);
  const float ty = 2.0F * (qz * x - qx * z);
  const float tz = 2.0F * (qx * y - qy * x);
  transformed_x =
      x + qw * tx + (qy * tz - qz * ty) + inverse_alpha * transform.start_translation[0] + alpha * transform.end_translation[0];
  transformed_y =
      y + qw * ty + (qz * tx - qx * tz) + inverse_alpha * transform.start_translation[1] + alpha * transform.end_translation[1];
  transformed_z =
      z + qw * tz + (qx * ty - qy * tx) + inverse_alpha * transform.start_translation[2] + alpha * transform.end_translation[2];
}

}  // namespace point_cloud_fusion
