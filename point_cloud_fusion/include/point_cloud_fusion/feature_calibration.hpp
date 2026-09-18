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

  float apply(float value) const;
};

bool readScalarField(const uint8_t* data, uint8_t datatype, bool bigendian, float& value);
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

  explicit DistributionFeatureCalibrator(Config config, LogCallback log_callback = {});
  ~DistributionFeatureCalibrator();

  DistributionFeatureCalibrator(const DistributionFeatureCalibrator&) = delete;
  DistributionFeatureCalibrator& operator=(const DistributionFeatureCalibrator&) = delete;
  DistributionFeatureCalibrator(DistributionFeatureCalibrator&&) = delete;
  DistributionFeatureCalibrator& operator=(DistributionFeatureCalibrator&&) = delete;

  void enqueue(const std::vector<PointCloudMsg::ConstSharedPtr>& clouds);
  std::vector<std::shared_ptr<const FeatureMapping>> mappings() const;

  static std::shared_ptr<const FeatureMapping> buildMapping(std::vector<float> source,
                                                            std::vector<float> target,
                                                            std::size_t quantile_count);

 private:
  void run();
  void process(const std::vector<PointCloudMsg::ConstSharedPtr>& clouds);
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
