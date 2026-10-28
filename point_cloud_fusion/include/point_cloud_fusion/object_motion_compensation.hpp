// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>

namespace point_cloud_fusion {

/** Per-input absolute radial velocity policy; offsets always refer to measured fields. */
struct ObjectMotion {
  int velocity_offset{-1};
  int velocity_datatype{0};
  int invalid_flags_offset{-1};
  uint32_t invalid_flags_mask{0};
  float reference_offset_sec{0.0F};
  float time_scale_sec{1.0e-9F};
  float doppler_sign{1.0F};
  float max_time_offset_sec{0.2F};
  float max_speed_mps{100.0F};
  float end_ray_origin[3]{0.0F, 0.0F, 0.0F};
  uint32_t ray_origin_max_time_offset{0};
};

/**
 * @brief Advance a measured acquisition ray using absolute radial velocity.
 *
 * Add displacement along the acquisition ray before the rigid/deskew transform.
 * Deskewed inputs subtract the acquisition origin in the header frame before
 * computing the ray. Unknown, implausible, and zero measurements leave XYZ unchanged.
 */
// clang-format off
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool compensateObjectMotion(const ObjectMotion& motion, float velocity, uint32_t point_time,
                                    float& x, float& y, float& z) {
  // clang-format on
  const float dt = motion.reference_offset_sec - static_cast<float>(point_time) * motion.time_scale_sec;
  if (!std::isfinite(velocity) || std::abs(velocity) > motion.max_speed_mps || !std::isfinite(dt) ||
      std::abs(dt) > motion.max_time_offset_sec || velocity == 0.0F || dt == 0.0F) {
    return false;
  }
  const float origin_alpha = motion.ray_origin_max_time_offset > 0
                                 ? (point_time >= motion.ray_origin_max_time_offset
                                        ? 1.0F : static_cast<float>(point_time) / motion.ray_origin_max_time_offset)
                                 : 0.0F;
  const float ray_x = x - origin_alpha * motion.end_ray_origin[0];
  const float ray_y = y - origin_alpha * motion.end_ray_origin[1];
  const float ray_z = z - origin_alpha * motion.end_ray_origin[2];
  const float range_squared = ray_x * ray_x + ray_y * ray_y + ray_z * ray_z;
  if (!std::isfinite(range_squared) || range_squared <= 0.0F) return false;
#ifdef __CUDA_ARCH__
  const float inverse_range = rsqrtf(range_squared);
#else
  const float inverse_range = 1.0F / std::sqrt(range_squared);
#endif
  const float displacement_scale = motion.doppler_sign * velocity * dt * inverse_range;
  if (!std::isfinite(displacement_scale)) return false;
  x += ray_x * displacement_scale;
  y += ray_y * displacement_scale;
  z += ray_z * displacement_scale;
  return true;
}

}  // namespace point_cloud_fusion
