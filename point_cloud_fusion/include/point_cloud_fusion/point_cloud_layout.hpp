// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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

struct FieldFill {
  uint32_t destination_offset{0};
  std::vector<uint8_t> bytes;
};

struct FieldConversion {
  uint32_t source_offset{0};
  uint32_t destination_offset{0};
  uint8_t source_datatype{0};
  uint8_t destination_datatype{0};
};

struct InputLayout {
  bool valid{false};
  uint32_t x_offset{0};
  uint32_t y_offset{0};
  uint32_t z_offset{0};
  int time_offset{-1};
  bool whole_point_copy{false};
  std::vector<FieldCopy> copies;
  std::vector<FieldConversion> conversions;
  std::vector<FieldFill> field_fills;
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
  bool is_dense{true};
  std::string fill_error;
  std::vector<std::string> unused_fill_fields;
  std::vector<InputLayout> inputs;
  std::vector<std::string> conflicting_fields;
};

using MissingFieldFillValues = std::unordered_map<std::string, double>;

/** Parse field=value entries; values may be finite numbers or NaN. */
inline MissingFieldFillValues parseMissingFieldFillValues(const std::vector<std::string>& entries) {
  const auto trim = [](const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? std::string{} : value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
  };
  MissingFieldFillValues result;
  for (const auto& entry : entries) {
    const auto separator = entry.find('=');
    if (separator == std::string::npos) {
      throw std::invalid_argument("Expected field=value in missing_field_fill_values: " + entry);
    }
    const auto name = trim(entry.substr(0, separator));
    const auto text = trim(entry.substr(separator + 1));
    if (name.empty() || text.empty()) throw std::invalid_argument("Empty field or value in missing_field_fill_values: " + entry);
    double value{};
    try {
      std::size_t consumed = 0;
      value = std::stod(text, &consumed);
      if (consumed != text.size() || std::isinf(value)) throw std::invalid_argument("invalid value");
    } catch (const std::exception&) {
      throw std::invalid_argument("Fill value for '" + name + "' must be a finite number or nan");
    }
    if (!result.emplace(name, value).second) {
      throw std::invalid_argument("Duplicate missing-field fill policy for '" + name + "'");
    }
  }
  return result;
}

/** Validate and encode one element of a field fill in host byte order. */
template <typename T>
inline void encodeFillValue(double value, uint8_t* destination) {
  T typed{};
  if constexpr (std::is_integral_v<T>) {
    if (!std::isfinite(value) || std::trunc(value) != value || value < static_cast<double>(std::numeric_limits<T>::lowest()) ||
        value > static_cast<double>(std::numeric_limits<T>::max())) {
      throw std::invalid_argument("fill must be an in-range integer (NaN is only supported for FLOAT32/FLOAT64)");
    }
    typed = static_cast<T>(value);
  } else {
    if (std::isinf(value) || (std::isfinite(value) && std::abs(value) > std::numeric_limits<T>::max())) {
      throw std::invalid_argument("fill exceeds the floating-point datatype range");
    }
    typed = std::isnan(value) ? std::numeric_limits<T>::quiet_NaN() : static_cast<T>(value);
  }
  std::memcpy(destination, &typed, sizeof(typed));
}

/** Build an endian-correct fill, broadcasting the value to every array element. */
inline FieldFill makeFieldFill(const sensor_msgs::msg::PointField& field, double value, bool is_bigendian) {
  FieldFill fill;
  fill.destination_offset = field.offset;
  const auto element_size = fieldDatatypeSize(field.datatype);
  if (element_size == 0 || field.count == 0) throw std::invalid_argument("invalid fill datatype or element count");
  fill.bytes.resize(element_size * field.count);
  using F = sensor_msgs::msg::PointField;
  switch (field.datatype) {
    case F::INT8:
      encodeFillValue<int8_t>(value, fill.bytes.data());
      break;
    case F::UINT8:
      encodeFillValue<uint8_t>(value, fill.bytes.data());
      break;
    case F::INT16:
      encodeFillValue<int16_t>(value, fill.bytes.data());
      break;
    case F::UINT16:
      encodeFillValue<uint16_t>(value, fill.bytes.data());
      break;
    case F::INT32:
      encodeFillValue<int32_t>(value, fill.bytes.data());
      break;
    case F::UINT32:
      encodeFillValue<uint32_t>(value, fill.bytes.data());
      break;
    case F::FLOAT32:
      encodeFillValue<float>(value, fill.bytes.data());
      break;
    case F::FLOAT64:
      encodeFillValue<double>(value, fill.bytes.data());
      break;
    default:
      throw std::invalid_argument("unsupported fill datatype");
  }
  const uint16_t endian_check = 1;
  uint8_t first_byte{};
  std::memcpy(&first_byte, &endian_check, sizeof(first_byte));
  const bool host_bigendian = first_byte == 0;
  if (is_bigendian != host_bigendian) {
    std::reverse(fill.bytes.begin(), std::next(fill.bytes.begin(), static_cast<std::ptrdiff_t>(element_size)));
  }
  for (uint32_t element = 1; element < field.count; ++element) {
    std::copy_n(fill.bytes.cbegin(), element_size,
                std::next(fill.bytes.begin(), static_cast<std::ptrdiff_t>(element * element_size)));
  }
  return fill;
}

/** Apply configured fills after zero-initializing a destination point. */
inline void fillMissingFields(uint8_t* destination, const InputLayout& input) {
  for (const auto& fill : input.field_fills) {
    std::copy(fill.bytes.begin(), fill.bytes.end(), std::next(destination, fill.destination_offset));
  }
}

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
 * @return True when the datatype and count are valid and the field is in
 * bounds.
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

/** Return whether a configured output-field list publishes a field. */
inline bool fieldIsPublished(const std::vector<std::string>& requested_fields, const std::string& name) {
  return requested_fields.empty() || std::find(requested_fields.begin(), requested_fields.end(), name) != requested_fields.end();
}

/**
 * Build the field list used internally while processing a batch.
 *
 * An empty requested list already retains every field. For an explicit list,
 * append the point-time field when motion compensation needs it internally.
 */
inline std::vector<std::string> processingFields(const std::vector<std::string>& requested_fields,
                                                 const std::string& time_field_name,
                                                 bool motion_compensation) {
  if (requested_fields.empty() || !motion_compensation || fieldIsPublished(requested_fields, time_field_name)) {
    return requested_fields;
  }
  auto result = requested_fields;
  result.push_back(time_field_name);
  return result;
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
 * @brief Build a common output layout and per-input copy plan for a cloud
 * batch.
 *
 * Invalid inputs are retained in the result with a rejection reason. Depending
 * on the requested fields and input compatibility, the result either preserves
 * a homogeneous input layout or builds a packed layout for compatible fields.
 *
 * @param clouds Point clouds to validate and combine.
 * @param requested_fields Optional ordered list of fields to include.
 * @param time_field_name Name of the per-point time-offset field.
 * @param fill_values Overrides; missing floats default to NaN, integers to zero.
 * @param calibration_field Optional leader descriptor retained across scalar datatype differences.
 * @return Output layout and copy plan for every input cloud.
 */
inline BatchLayout buildBatchLayout(const std::vector<sensor_msgs::msg::PointCloud2::ConstSharedPtr>& clouds,
                                    const std::vector<std::string>& requested_fields,
                                    const std::string& time_field_name,
                                    const MissingFieldFillValues& fill_values = {},
                                    const sensor_msgs::msg::PointField* calibration_field = nullptr) {
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
    result.is_dense = result.is_dense && cloud->is_dense;
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
      if (descriptor != nullptr) {
        selected.push_back(calibration_field && name == calibration_field->name ? *calibration_field : *descriptor);
      }
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
      if (calibration_field && name == calibration_field->name) {
        selected.push_back(*calibration_field);
      } else if (conflicts.count(name) != 0U) {
        result.conflicting_fields.push_back(name);
      } else {
        selected.push_back(descriptors.at(name));
      }
    }
  }

  // XYZ is part of every valid output even if a configured optional-field list
  // omitted it. This keeps the PointCloud2 usable and guarantees transformed
  // XYZ.
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

  std::unordered_set<std::string> used_fill_fields;
  for (const auto index : valid_indices) {
    const auto& cloud = *clouds[index];
    auto& input = result.inputs[index];
    input.whole_point_copy = requested_fields.empty() && result.homogeneous_fast_path;
    if (input.whole_point_copy) continue;
    for (const auto& destination : result.fields) {
      const auto* source = findField(cloud, destination.name);
      if (source != nullptr && sameType(*source, destination) && validField(*source, cloud.point_step)) {
        input.copies.push_back({source->offset, destination.offset, fieldDatatypeSize(source->datatype) * source->count});
      } else if (calibration_field && destination.name == calibration_field->name && source != nullptr &&
                 source->count == 1 && validField(*source, cloud.point_step)) {
        input.conversions.push_back({source->offset, destination.offset, source->datatype, destination.datatype});
      } else if (const auto fill = fill_values.find(destination.name);
                 fill != fill_values.end() || destination.datatype == sensor_msgs::msg::PointField::FLOAT32 ||
                 destination.datatype == sensor_msgs::msg::PointField::FLOAT64) {
        const double value = fill != fill_values.end() ? fill->second : std::numeric_limits<double>::quiet_NaN();
        try {
          input.field_fills.push_back(makeFieldFill(destination, value, result.is_bigendian));
        } catch (const std::invalid_argument& error) {
          result.fill_error = "Cannot fill missing field '" + destination.name + "': " + error.what();
          return result;
        }
        if (fill != fill_values.end()) used_fill_fields.insert(destination.name);
        if (std::isnan(value)) result.is_dense = false;
      } else {
        input.zero_filled_fields.push_back(destination.name);
      }
    }
  }
  for (const auto& fill : fill_values) {
    if (used_fill_fields.count(fill.first) == 0U) result.unused_fill_fields.push_back(fill.first);
  }
  std::sort(result.unused_fill_fields.begin(), result.unused_fill_fields.end());
  return result;
}

}  // namespace point_cloud_fusion::detail
