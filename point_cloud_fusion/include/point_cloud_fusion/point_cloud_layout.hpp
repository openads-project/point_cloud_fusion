// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace point_cloud_fusion::detail {

/**
 * @brief Return the size in bytes of one PointField datatype element.
 *
 * @param datatype PointField datatype identifier.
 * @return Element size in bytes, or zero if the datatype is unsupported.
 */
inline std::size_t fieldDatatypeSize(uint8_t datatype) {
  using sensor_msgs::msg::PointField;
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      return 1;
    case PointField::INT16:
    case PointField::UINT16:
      return 2;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      return 4;
    case PointField::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

struct FieldCopy {
  uint32_t source_offset{0};
  uint32_t destination_offset{0};
  std::size_t byte_length{0};
};

struct InputLayout {
  bool valid{false};
  uint32_t x_offset{0};
  uint32_t y_offset{0};
  uint32_t z_offset{0};
  int time_offset{-1};
  bool whole_point_copy{false};
  std::vector<FieldCopy> copies;
  std::vector<std::string> zero_filled_fields;
  std::string rejection_reason;
};

struct BatchLayout {
  std::vector<sensor_msgs::msg::PointField> fields;
  uint32_t point_step{0};
  uint32_t x_offset{0};
  uint32_t y_offset{0};
  uint32_t z_offset{0};
  bool homogeneous_fast_path{false};
  bool is_bigendian{false};
  std::vector<InputLayout> inputs;
  std::vector<std::string> conflicting_fields;
};

/**
 * @brief Check whether two fields have the same name and element type.
 *
 * @param lhs First field descriptor.
 * @param rhs Second field descriptor.
 * @return True when the field names, datatypes, and element counts match.
 */
inline bool sameType(const sensor_msgs::msg::PointField& lhs, const sensor_msgs::msg::PointField& rhs) {
  return lhs.name == rhs.name && lhs.datatype == rhs.datatype && lhs.count == rhs.count;
}

/**
 * @brief Check whether a field descriptor fits within a point record.
 *
 * @param field Field descriptor to validate.
 * @param point_step Size in bytes of one point record.
 * @return True when the datatype and count are valid and the field is in bounds.
 */
inline bool validField(const sensor_msgs::msg::PointField& field, uint32_t point_step) {
  const auto size = fieldDatatypeSize(field.datatype);
  return size > 0 && field.count > 0 && static_cast<std::size_t>(field.offset) + size * field.count <= point_step;
}

/**
 * @brief Find a field descriptor by name in a point cloud.
 *
 * @param cloud Point cloud whose fields are searched.
 * @param name Field name to find.
 * @return Pointer to the matching descriptor, or nullptr when it is absent.
 */
inline const sensor_msgs::msg::PointField* findField(const sensor_msgs::msg::PointCloud2& cloud, const std::string& name) {
  const auto it =
      std::find_if(cloud.fields.begin(), cloud.fields.end(), [&name](const auto& field) { return field.name == name; });
  return it == cloud.fields.end() ? nullptr : &*it;
}

/**
 * @brief Check whether a field is a valid scalar FLOAT32 XYZ component.
 *
 * @param field Field descriptor to validate; may be nullptr.
 * @param point_step Size in bytes of one point record.
 * @return True when the field is a valid in-bounds scalar FLOAT32 value.
 */
inline bool validXyzField(const sensor_msgs::msg::PointField* field, uint32_t point_step) {
  return field != nullptr && field->datatype == sensor_msgs::msg::PointField::FLOAT32 && field->count == 1 &&
         validField(*field, point_step);
}

/**
 * @brief Build a common output layout and per-input copy plan for a cloud batch.
 *
 * Invalid inputs are retained in the result with a rejection reason. Depending
 * on the requested fields and input compatibility, the result either preserves
 * a homogeneous input layout or builds a packed layout for compatible fields.
 *
 * @param clouds Point clouds to validate and combine.
 * @param requested_fields Optional ordered list of fields to include.
 * @param time_field_name Name of the per-point time-offset field.
 * @return Output layout and copy plan for every input cloud.
 */
inline BatchLayout buildBatchLayout(const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>& clouds,
                                    const std::vector<std::string>& requested_fields,
                                    const std::string& time_field_name) {
  BatchLayout result;
  result.inputs.resize(clouds.size());

  std::vector<std::size_t> valid_indices;
  for (std::size_t index = 0; index < clouds.size(); ++index) {
    const auto& cloud = clouds[index];
    auto& input = result.inputs[index];
    if (!cloud) {
      input.rejection_reason = "null cloud";
      continue;
    }
    const auto* x = findField(*cloud, "x");
    const auto* y = findField(*cloud, "y");
    const auto* z = findField(*cloud, "z");
    if (!validXyzField(x, cloud->point_step) || !validXyzField(y, cloud->point_step) || !validXyzField(z, cloud->point_step)) {
      input.rejection_reason = "missing or invalid FLOAT32 x/y/z field";
      continue;
    }
    input.valid = true;
    input.x_offset = x->offset;
    input.y_offset = y->offset;
    input.z_offset = z->offset;
    if (const auto* time = findField(*cloud, time_field_name); time != nullptr &&
                                                               time->datatype == sensor_msgs::msg::PointField::UINT32 &&
                                                               time->count == 1 && validField(*time, cloud->point_step)) {
      input.time_offset = static_cast<int>(time->offset);
    }
    valid_indices.push_back(index);
  }
  if (valid_indices.empty()) return result;

  const auto& first = *clouds[valid_indices.front()];
  result.is_bigendian = first.is_bigendian;
  result.homogeneous_fast_path = std::all_of(valid_indices.begin(), valid_indices.end(), [&](std::size_t index) {
    const auto& cloud = *clouds[index];
    return cloud.point_step == first.point_step && cloud.fields == first.fields && cloud.is_bigendian == first.is_bigendian;
  });

  std::vector<sensor_msgs::msg::PointField> selected;
  if (!requested_fields.empty()) {
    std::unordered_set<std::string> emitted;
    for (const auto& name : requested_fields) {
      if (!emitted.insert(name).second) continue;
      const sensor_msgs::msg::PointField* descriptor = nullptr;
      for (const auto index : valid_indices) {
        const auto* candidate = findField(*clouds[index], name);
        if (candidate != nullptr && validField(*candidate, clouds[index]->point_step)) {
          descriptor = candidate;
          break;
        }
      }
      if (descriptor != nullptr) selected.push_back(*descriptor);
    }
  } else if (result.homogeneous_fast_path) {
    selected = first.fields;
  } else {
    std::unordered_map<std::string, sensor_msgs::msg::PointField> descriptors;
    std::unordered_set<std::string> conflicts;
    std::vector<std::string> order;
    for (const auto index : valid_indices) {
      for (const auto& field : clouds[index]->fields) {
        if (!validField(field, clouds[index]->point_step)) continue;
        const auto [it, inserted] = descriptors.emplace(field.name, field);
        if (inserted) {
          order.push_back(field.name);
        } else if (!sameType(it->second, field)) {
          conflicts.insert(field.name);
        }
      }
    }
    for (const auto& name : order) {
      if (conflicts.count(name) != 0U) {
        result.conflicting_fields.push_back(name);
      } else {
        selected.push_back(descriptors.at(name));
      }
    }
  }

  // XYZ is part of every valid output even if a configured optional-field list
  // omitted it. This keeps the PointCloud2 usable and guarantees transformed XYZ.
  for (const char* name : {"x", "y", "z"}) {
    if (std::none_of(selected.begin(), selected.end(), [name](const auto& field) { return field.name == name; })) {
      selected.push_back(*findField(first, name));
    }
  }

  if (requested_fields.empty() && result.homogeneous_fast_path) {
    result.fields = selected;
    result.point_step = first.point_step;
  } else {
    for (auto field : selected) {
      field.offset = result.point_step;
      result.point_step += static_cast<uint32_t>(fieldDatatypeSize(field.datatype) * field.count);
      result.fields.push_back(std::move(field));
    }
  }
  for (const auto& field : result.fields) {
    if (field.name == "x") result.x_offset = field.offset;
    if (field.name == "y") result.y_offset = field.offset;
    if (field.name == "z") result.z_offset = field.offset;
  }

  for (const auto index : valid_indices) {
    const auto& cloud = *clouds[index];
    auto& input = result.inputs[index];
    input.whole_point_copy = requested_fields.empty() && result.homogeneous_fast_path;
    if (input.whole_point_copy) continue;
    for (const auto& destination : result.fields) {
      const auto* source = findField(cloud, destination.name);
      if (source != nullptr && sameType(*source, destination) && validField(*source, cloud.point_step)) {
        input.copies.push_back({source->offset, destination.offset, fieldDatatypeSize(source->datatype) * source->count});
      } else {
        input.zero_filled_fields.push_back(destination.name);
      }
    }
  }
  return result;
}

}  // namespace point_cloud_fusion::detail
