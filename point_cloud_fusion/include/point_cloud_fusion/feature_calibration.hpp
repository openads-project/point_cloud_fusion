// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace point_cloud_fusion {

struct FeatureMapping {
  std::vector<float> source;
  std::vector<float> target;

  /**
   * @brief Map a source value to the calibrated target distribution.
   * @param value Source feature value.
   * @return Piecewise-linearly mapped target value.
   */
  float apply(float value) const;
};

/**
 * @brief Read a PointField scalar and convert it to floating point.
 * @param data Address of the scalar value.
 * @param datatype sensor_msgs::msg::PointField datatype identifier.
 * @param bigendian Whether the source cloud uses big-endian byte order.
 * @param value Destination for the converted value.
 * @return True when the datatype is supported.
 */
bool readScalarField(const uint8_t* data, uint8_t datatype, bool bigendian, float& value);

/**
 * @brief Convert and write a floating-point value to a PointField scalar.
 * @param data Address of the destination scalar.
 * @param datatype sensor_msgs::msg::PointField datatype identifier.
 * @param bigendian Whether the destination cloud uses big-endian byte order.
 * @param value Value to convert and write.
 * @return True when the datatype is supported.
 */
bool writeScalarField(uint8_t* data, uint8_t datatype, bool bigendian, float value);

class DistributionFeatureCalibrator {
 public:
  using PointCloudMsg = sensor_msgs::msg::PointCloud2;

  struct Config {
    std::string field;
    std::vector<std::size_t> leaders;
    std::vector<std::size_t> followers;
    std::size_t input_count{0};
    std::size_t samples_per_cloud{2048};
    std::size_t window_samples{100000};
    std::size_t minimum_samples{20000};
    std::size_t quantile_count{64};
    double update_interval_sec{3.0};
  };

  using LogCallback = std::function<void(const std::string&)>;

  /**
   * @brief Start a continuous asynchronous distribution calibrator.
   * @param config Calibration inputs, sampling limits, and update cadence.
   * @param log_callback Optional callback for calibration status messages.
   */
  explicit DistributionFeatureCalibrator(Config config, LogCallback log_callback = {});

  /** @brief Stop the worker and release its resources. */
  ~DistributionFeatureCalibrator();

  /** @brief Distribution calibrators cannot be copied. */
  DistributionFeatureCalibrator(const DistributionFeatureCalibrator&) = delete;

  /** @brief Distribution calibrators cannot be copy-assigned. */
  DistributionFeatureCalibrator& operator=(const DistributionFeatureCalibrator&) = delete;

  /** @brief Distribution calibrators cannot be moved. */
  DistributionFeatureCalibrator(DistributionFeatureCalibrator&&) = delete;

  /** @brief Distribution calibrators cannot be move-assigned. */
  DistributionFeatureCalibrator& operator=(DistributionFeatureCalibrator&&) = delete;

  /**
   * @brief Queue the newest synchronized cloud batch for background sampling.
   * @param clouds Input clouds in configured input-topic order.
   */
  void enqueue(const std::vector<PointCloudMsg::ConstSharedPtr>& clouds);

  /**
   * @brief Get an immutable snapshot of the current per-input mappings.
   * @return Mapping for each input, with null entries for uncalibrated inputs.
   */
  std::vector<std::shared_ptr<const FeatureMapping>> mappings() const;

  /**
   * @brief Build a quantile mapping between sampled distributions.
   * @param source Samples from one follower input.
   * @param target Pooled samples from the leading inputs.
   * @param quantile_count Number of piecewise-linear quantile intervals.
   * @return Immutable mapping, or null when either distribution is unusable.
   */
  static std::shared_ptr<const FeatureMapping> buildMapping(std::vector<float> source,
                                                            std::vector<float> target,
                                                            std::size_t quantile_count);

 private:
  /** @brief Consume queued batches until shutdown is requested. */
  void run();

  /**
   * @brief Sample one synchronized batch into the rolling distributions.
   * @param clouds Input clouds in configured input-topic order.
   */
  void process(const std::vector<PointCloudMsg::ConstSharedPtr>& clouds);

  /** @brief Recompute follower mappings from the current sample windows. */
  void updateMappings();

  Config config_;
  LogCallback log_callback_;
  std::vector<bool> sampled_inputs_;
  std::vector<std::uint64_t> sample_sequences_;
  std::vector<std::deque<float>> samples_;
  std::vector<bool> mapping_announced_;

  mutable std::mutex mapping_mutex_;
  std::vector<std::shared_ptr<const FeatureMapping>> mappings_;

  std::mutex queue_mutex_;
  std::condition_variable queue_condition_;
  std::vector<PointCloudMsg::ConstSharedPtr> pending_;
  bool stop_{false};
  std::thread worker_;
};

}  // namespace point_cloud_fusion
