// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <point_cloud_fusion/feature_calibration.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <sstream>
#include <type_traits>

#include <point_cloud_fusion/point_cloud_layout.hpp>

namespace point_cloud_fusion {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGoldenRatioConjugate = 0.61803398874989484820;

template <typename T>
T loadScalar(const uint8_t* data, bool bigendian) {
  T value{};
  std::array<uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), data, sizeof(T));
  if (bigendian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

template <typename T>
void storeScalar(uint8_t* data, bool bigendian, T value) {
  std::array<uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  if (bigendian) std::reverse(bytes.begin(), bytes.end());
  std::memcpy(data, bytes.data(), sizeof(T));
}

template <typename Byte>
Byte* byteOffset(Byte* data, std::size_t offset) {
  return std::next(data, static_cast<std::ptrdiff_t>(offset));
}

template <typename T>
T saturate(float value) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(value);
  } else {
    const float rounded = std::round(value);
    return static_cast<T>(std::clamp(rounded, static_cast<float>(std::numeric_limits<T>::lowest()),
                                     static_cast<float>(std::numeric_limits<T>::max())));
  }
}

float percentile(const std::vector<float>& sorted, double fraction) {
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const float alpha = static_cast<float>(position - static_cast<double>(lower));
  return sorted[lower] + alpha * (sorted[upper] - sorted[lower]);
}

}  // namespace

float FeatureMapping::apply(float value) const {
  if (source.size() < 2 || source.size() != target.size() || !std::isfinite(value)) return value;
  if (value <= source.front()) return target.front();
  if (value >= source.back()) return target.back();
  const auto upper = std::upper_bound(source.begin(), source.end(), value);
  const std::size_t index = static_cast<std::size_t>(upper - source.begin());
  const float denominator = source[index] - source[index - 1];
  if (denominator <= 0.0F) return target[index];
  const float alpha = (value - source[index - 1]) / denominator;
  return target[index - 1] + alpha * (target[index] - target[index - 1]);
}

bool readScalarField(const uint8_t* data, uint8_t datatype, bool bigendian, float& value) {
  using Field = sensor_msgs::msg::PointField;
  switch (datatype) {
    case Field::INT8:
      value = loadScalar<int8_t>(data, bigendian);
      break;
    case Field::UINT8:
      value = loadScalar<uint8_t>(data, bigendian);
      break;
    case Field::INT16:
      value = loadScalar<int16_t>(data, bigendian);
      break;
    case Field::UINT16:
      value = loadScalar<uint16_t>(data, bigendian);
      break;
    case Field::INT32:
      value = static_cast<float>(loadScalar<int32_t>(data, bigendian));
      break;
    case Field::UINT32:
      value = static_cast<float>(loadScalar<uint32_t>(data, bigendian));
      break;
    case Field::FLOAT32:
      value = loadScalar<float>(data, bigendian);
      break;
    case Field::FLOAT64:
      value = static_cast<float>(loadScalar<double>(data, bigendian));
      break;
    default:
      return false;
  }
  return std::isfinite(value);
}

bool writeScalarField(uint8_t* data, uint8_t datatype, bool bigendian, float value) {
  using Field = sensor_msgs::msg::PointField;
  switch (datatype) {
    case Field::INT8:
      storeScalar(data, bigendian, saturate<int8_t>(value));
      break;
    case Field::UINT8:
      storeScalar(data, bigendian, saturate<uint8_t>(value));
      break;
    case Field::INT16:
      storeScalar(data, bigendian, saturate<int16_t>(value));
      break;
    case Field::UINT16:
      storeScalar(data, bigendian, saturate<uint16_t>(value));
      break;
    case Field::INT32:
      storeScalar(data, bigendian, saturate<int32_t>(value));
      break;
    case Field::UINT32:
      storeScalar(data, bigendian, saturate<uint32_t>(value));
      break;
    case Field::FLOAT32:
      storeScalar(data, bigendian, value);
      break;
    case Field::FLOAT64:
      storeScalar(data, bigendian, static_cast<double>(value));
      break;
    default:
      return false;
  }
  return true;
}

DistributionFeatureCalibrator::DistributionFeatureCalibrator(Config config, LogCallback log_callback)
    : config_(std::move(config)),
      log_callback_(std::move(log_callback)),
      sampled_inputs_(config_.input_count, false),
      sample_sequences_(config_.input_count, 0),
      samples_(config_.input_count),
      mapping_announced_(config_.input_count, false),
      mappings_(config_.input_count) {
  for (const auto index : config_.leaders) {
    if (index < sampled_inputs_.size()) sampled_inputs_[index] = true;
  }
  for (const auto index : config_.followers) {
    if (index < sampled_inputs_.size()) sampled_inputs_[index] = true;
  }
  worker_ = std::thread(&DistributionFeatureCalibrator::run, this);
}

DistributionFeatureCalibrator::~DistributionFeatureCalibrator() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  queue_condition_.notify_one();
  if (worker_.joinable()) worker_.join();
}

void DistributionFeatureCalibrator::enqueue(const std::vector<PointCloudMsg::ConstSharedPtr>& clouds) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  pending_ = clouds;  // One latest-only slot prevents calibration back-pressure.
  queue_condition_.notify_one();
}

std::vector<std::shared_ptr<const FeatureMapping>> DistributionFeatureCalibrator::mappings() const {
  std::lock_guard<std::mutex> lock(mapping_mutex_);
  return mappings_;
}

std::shared_ptr<const FeatureMapping> DistributionFeatureCalibrator::buildMapping(std::vector<float> source,
                                                                                  std::vector<float> target,
                                                                                  std::size_t quantile_count) {
  if (source.size() < 2 || target.size() < 2 || quantile_count < 1) return {};
  std::sort(source.begin(), source.end());
  std::sort(target.begin(), target.end());
  auto mapping = std::make_shared<FeatureMapping>();
  mapping->source.reserve(quantile_count + 1);
  mapping->target.reserve(quantile_count + 1);
  for (std::size_t i = 0; i <= quantile_count; ++i) {
    const double position = static_cast<double>(i) / static_cast<double>(quantile_count);
    // Concentrate knots near both tails so one extreme value cannot dominate
    // the entire final interval.
    const double fraction = 0.5 * (1.0 - std::cos(kPi * position));
    const float source_value = percentile(source, fraction);
    const float target_value = percentile(target, fraction);
    if (!mapping->source.empty() && source_value <= mapping->source.back()) {
      mapping->target.back() = target_value;
      continue;
    }
    mapping->source.push_back(source_value);
    mapping->target.push_back(target_value);
  }
  return mapping->source.size() >= 2 ? mapping : std::shared_ptr<const FeatureMapping>{};
}

void DistributionFeatureCalibrator::run() {
  auto last_update = std::chrono::steady_clock::now();
  while (true) {
    std::vector<PointCloudMsg::ConstSharedPtr> clouds;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_condition_.wait(lock, [this] { return stop_ || !pending_.empty(); });
      if (stop_) return;
      clouds.swap(pending_);
    }
    process(clouds);
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last_update).count() >= config_.update_interval_sec) {
      updateMappings();
      last_update = now;
    }
  }
}

void DistributionFeatureCalibrator::process(const std::vector<PointCloudMsg::ConstSharedPtr>& clouds) {
  for (std::size_t index = 0; index < std::min(clouds.size(), samples_.size()); ++index) {
    if (!sampled_inputs_[index]) continue;
    const auto& cloud = clouds[index];
    if (!cloud || cloud->point_step == 0) continue;
    const auto* field = detail::findField(*cloud, config_.field);
    const auto* x_field = detail::findField(*cloud, "x");
    const auto* y_field = detail::findField(*cloud, "y");
    const auto* z_field = detail::findField(*cloud, "z");
    if (field == nullptr || field->count != 1 || !detail::validField(*field, cloud->point_step)) continue;
    if (!detail::validXyzField(x_field, cloud->point_step) || !detail::validXyzField(y_field, cloud->point_step) ||
        !detail::validXyzField(z_field, cloud->point_step)) {
      continue;
    }
    const std::size_t points =
        std::min(static_cast<std::size_t>(cloud->width) * cloud->height, cloud->data.size() / cloud->point_step);
    const std::size_t count = std::min(points, config_.samples_per_cloud);
    if (count == 0) continue;
    const double stride = static_cast<double>(points) / static_cast<double>(count);
    // Rotate the sample inside each stratum on every cloud. Fixed offsets can
    // alias an organized scan pattern and repeatedly sample the same beams.
    const double phase = std::fmod(static_cast<double>(sample_sequences_[index]++) * kGoldenRatioConjugate, 1.0);
    auto& window = samples_[index];
    for (std::size_t sample = 0; sample < count; ++sample) {
      const std::size_t point = std::min(static_cast<std::size_t>((static_cast<double>(sample) + phase) * stride), points - 1);
      const uint8_t* point_data = byteOffset(cloud->data.data(), point * cloud->point_step);
      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      if (!readScalarField(byteOffset(point_data, x_field->offset), x_field->datatype, cloud->is_bigendian, x) ||
          !readScalarField(byteOffset(point_data, y_field->offset), y_field->datatype, cloud->is_bigendian, y) ||
          !readScalarField(byteOffset(point_data, z_field->offset), z_field->datatype, cloud->is_bigendian, z)) {
        continue;
      }
      float value = 0.0F;
      if (readScalarField(byteOffset(point_data, field->offset), field->datatype, cloud->is_bigendian, value)) {
        window.push_back(value);
      }
    }
    while (window.size() > config_.window_samples) window.pop_front();
  }
}

void DistributionFeatureCalibrator::updateMappings() {
  std::vector<float> leaders;
  for (const std::size_t index : config_.leaders) {
    if (index < samples_.size()) leaders.insert(leaders.end(), samples_[index].begin(), samples_[index].end());
  }
  if (leaders.size() < config_.minimum_samples) return;

  auto updated = mappings();
  bool changed = false;
  std::vector<std::string> activations;
  for (const std::size_t index : config_.followers) {
    if (index >= samples_.size() || samples_[index].size() < config_.minimum_samples) continue;
    std::vector<float> follower(samples_[index].begin(), samples_[index].end());
    auto mapping = buildMapping(std::move(follower), leaders, config_.quantile_count);
    if (mapping) {
      if (!mapping_announced_[index]) {
        std::ostringstream activation;
        activation << "input " << index << " [" << mapping->source.front() << ", " << mapping->source.back() << "] -> ["
                   << mapping->target.front() << ", " << mapping->target.back() << "]";
        activations.push_back(activation.str());
        mapping_announced_[index] = true;
      }
      updated[index] = std::move(mapping);
      changed = true;
    }
  }
  if (!changed) return;
  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    mappings_ = std::move(updated);
  }
  if (log_callback_ && !activations.empty()) {
    std::ostringstream message;
    message << "Activated distribution calibration for field '" << config_.field << "': ";
    for (std::size_t i = 0; i < activations.size(); ++i) {
      if (i != 0) message << "; ";
      message << activations[i];
    }
    log_callback_(message.str());
  }
}

}  // namespace point_cloud_fusion
