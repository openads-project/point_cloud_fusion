// copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <point_cloud_fusion/point_cloud_layout.hpp>
#include <stdexcept>
#include <unordered_map>

namespace point_cloud_fusion::detail {

inline void validateVoxelSize(double size) {
  if (!std::isfinite(size) || size <= 0.0) {
    throw std::invalid_argument("downsampling.voxel_size must be finite and positive");
  }
}

struct VoxelKey {
  std::array<int64_t, 3> cell;
  bool operator==(const VoxelKey& other) const { return cell == other.cell; }
};

struct VoxelHash {
  std::size_t operator()(const VoxelKey& key) const {
    std::size_t hash = 0;
    for (auto axis : key.cell) {
      hash ^= std::hash<int64_t>{}(axis) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
    }
    return hash;
  }
};

// read coordinates without alignment or host-endianness assumptions.
inline float voxelCoordinate(const uint8_t* source, bool bigendian) {
  std::array<uint8_t, sizeof(float)> bytes{};
  std::memcpy(bytes.data(), source, bytes.size());
  const uint16_t endian = 1;
  uint8_t first_byte{};
  std::memcpy(&first_byte, &endian, sizeof(first_byte));
  if (bigendian != (first_byte == 0)) {
    std::reverse(bytes.begin(), bytes.end());
  }
  float value = 0.0F;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

// select complete fused records, then compact in input order using O(occupied cells) memory.
inline void downsampleVoxels(sensor_msgs::msg::PointCloud2& cloud, double size) {
  validateVoxelSize(size);
  std::array<uint32_t, 3> offsets{};
  std::size_t axis = 0;
  for (const auto* name : {"x", "y", "z"}) {
    const auto* field = findField(cloud, name);
    if (field == nullptr || field->datatype != sensor_msgs::msg::PointField::FLOAT32 || field->count != 1 ||
        field->offset > cloud.point_step || cloud.point_step - field->offset < sizeof(float)) {
      throw std::invalid_argument("voxel downsampling requires scalar FLOAT32 XYZ");
    }
    offsets.at(axis++) = field->offset;
  }
  if (cloud.height != 1 || cloud.point_step == 0 ||
      cloud.data.size() != static_cast<std::size_t>(cloud.width) * cloud.point_step) {
    throw std::invalid_argument("voxel downsampling requires a packed fused cloud");
  }
  struct Candidate {
    std::size_t index;
    double distance;
    std::array<double, 3> xyz;
  };
  std::unordered_map<VoxelKey, Candidate, VoxelHash> voxels;
  const auto record_at = [&cloud](std::size_t index) {
    return std::next(cloud.data.data(), static_cast<std::ptrdiff_t>(index * cloud.point_step));
  };
  const auto locate = [&](const uint8_t* record, std::array<double, 3>& xyz, double& distance) {
    VoxelKey key{};
    distance = 0.0;
    for (std::size_t a = 0; a < 3; ++a) {
      xyz.at(a) = voxelCoordinate(std::next(record, static_cast<std::ptrdiff_t>(offsets.at(a))), cloud.is_bigendian);
      const double scaled = xyz.at(a) / size;
      const double cell = std::floor(scaled);
      // 2^63 is exactly representable; INT64_MAX converted to double rounds up to it.
      if (!std::isfinite(cell) || cell < -0x1p63 || cell >= 0x1p63) {
        throw std::overflow_error("voxel cell index is non-finite or outside int64 range");
      }
      key.cell.at(a) = static_cast<int64_t>(cell);
      // compare in cell units to avoid overflow/underflow when squaring metric distances.
      const double delta = scaled - cell - 0.5;
      distance += delta * delta;
    }
    return key;
  };
  for (std::size_t i = 0; i < cloud.width; ++i) {
    const auto* record = record_at(i);
    Candidate candidate{i, 0.0, {}};
    const auto key = locate(record, candidate.xyz, candidate.distance);
    auto [entry, inserted] = voxels.emplace(key, candidate);
    if (!inserted) {
      const auto& best = entry->second;
      if (candidate.distance < best.distance ||
          (candidate.distance == best.distance &&
           (candidate.xyz < best.xyz ||
            (candidate.xyz == best.xyz && std::memcmp(record, record_at(best.index), cloud.point_step) < 0)))) {
        entry->second = candidate;
      }
    }
  }
  std::size_t output_count = 0;
  for (std::size_t i = 0; i < cloud.width; ++i) {
    const auto* record = record_at(i);
    std::array<double, 3> xyz{};
    double distance = 0.0;
    const auto key = locate(record, xyz, distance);
    if (voxels.at(key).index == i) {
      std::memmove(record_at(output_count), record, cloud.point_step);
      ++output_count;
    }
  }
  cloud.width = static_cast<uint32_t>(output_count);
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(output_count * cloud.point_step);
}

}  // namespace point_cloud_fusion::detail
