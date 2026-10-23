// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace point_cloud_fusion {

/**
 * @brief Collect fresh, single-use inputs without waiting indefinitely for peers.
 *
 * The first accepted input opens a collection window. Replacements never extend
 * its deadline. While rate limited, each topic retains its newest input; no
 * fusion work is performed until take() returns a batch. The newest buffered
 * stamp anchors the batch, so its full timestamp spread is bounded.
 *
 * Times and stamps must use the same clock. A backwards clock jump resets all
 * state, including duplicate detection, for bag replay. This class has no ROS
 * dependencies and must be accessed serially by its owner.
 *
 * @tparam MessagePtr Nullable pointer type holding an input message.
 */
template <typename MessagePtr>
class PartialBatchCollector {
 public:
  using Time = std::chrono::nanoseconds;

  /**
   * @brief Timestamp, collection, freshness, and publication limits.
   */
  struct Options {
    Time max_spread;
    Time collection_timeout;
    Time max_age;
    Time publish_interval;
  };

  /**
   * @brief Input message with its configured topic index and measurement stamp.
   */
  struct Sample {
    std::size_t topic_index;
    Time stamp;
    MessagePtr message;
  };

  /**
   * @brief Construct a collector with one pending slot per input topic.
   *
   * @param topic_count Number of configured input topics.
   * @param options Timing limits for collection and publication.
   * @throws std::invalid_argument If the topic count is zero or timing limits are invalid.
   */
  PartialBatchCollector(std::size_t topic_count, Options options)
      : options_(options), pending_(topic_count), last_received_stamp_(topic_count) {
    if (topic_count == 0 || options.max_spread < Time::zero() || options.collection_timeout < Time::zero() ||
        options.max_age <= Time::zero() || options.max_age < options.collection_timeout ||
        options.publish_interval <= Time::zero()) {
      throw std::invalid_argument("Invalid partial batch collector options");
    }
  }

  /**
   * @brief Buffer an input without extending the collection deadline.
   *
   * @param topic_index Index of the configured input topic.
   * @param stamp Measurement timestamp in the node clock's time base.
   * @param now Current node-clock time.
   * @param message Input message to retain until replacement or batch selection.
   * @return True if accepted; false for null, stale, duplicate, or reordered inputs.
   * @throws std::out_of_range If the topic index is not configured.
   */
  bool push(std::size_t topic_index, Time stamp, Time now, MessagePtr message) {
    observeTime(now);
    expire(now);
    if (topic_index >= pending_.size()) {
      throw std::out_of_range("Invalid partial batch input index");
    }
    if (!message || stamp < Time::zero() || !fresh(stamp, now) ||
        (last_received_stamp_[topic_index] && stamp <= *last_received_stamp_[topic_index]) ||
        (last_emitted_stamp_ && stamp <= *last_emitted_stamp_)) {
      return false;
    }
    last_received_stamp_[topic_index] = stamp;
    pending_[topic_index] = Sample{topic_index, stamp, std::move(message)};
    if (!window_start_) {
      window_start_ = now;
    }
    return true;
  }

  /**
   * @brief Determine when the pending batch becomes eligible for selection.
   *
   * @return Collection/rate-limit deadline, or std::nullopt if no input is pending.
   */
  std::optional<Time> deadline() const {
    if (!window_start_) {
      return std::nullopt;
    }
    return std::max(*window_start_ + options_.collection_timeout, next_publish_time_);
  }

  /**
   * @brief Finalize the newest compatible subset when its deadline is reached.
   *
   * Results preserve configured topic order. All pending inputs are released:
   * selected inputs are consumed once, and older incompatible inputs are dropped.
   * Even a failed downstream fusion does not retry an old observation.
   *
   * @param now Current node-clock time.
   * @return Selected inputs, or an empty vector if none are fresh or the deadline has not elapsed.
   */
  std::vector<Sample> take(Time now) {
    observeTime(now);
    expire(now);
    const auto due = deadline();
    if (!due || now < *due) {
      return {};
    }

    Time newest = Time::zero();
    for (const auto& sample : pending_) {
      if (sample) {
        newest = std::max(newest, sample->stamp);
      }
    }

    std::vector<Sample> batch;
    batch.reserve(pending_.size());
    for (auto& sample : pending_) {
      if (sample && newest - sample->stamp <= options_.max_spread) {
        batch.push_back(std::move(*sample));
      }
      sample.reset();
    }
    window_start_.reset();
    last_emitted_stamp_ = newest;
    next_publish_time_ = now + options_.publish_interval;
    return batch;
  }

  /**
   * @brief Start the rate-limit interval at actual publication completion.
   *
   * @param now Node-clock time when publication completed.
   */
  void published(Time now) {
    observeTime(now);
    next_publish_time_ = now + options_.publish_interval;
  }

 private:
  /**
   * @brief Check measurement age and future timestamp offset against the freshness limit.
   *
   * @param stamp Measurement timestamp in the node clock's time base.
   * @param now Current node-clock time.
   * @return True if the timestamp is within the configured freshness limit.
   */
  bool fresh(Time stamp, Time now) const {
    // Reject both delayed observations and implausibly far-future timestamps.
    return stamp <= now ? now - stamp <= options_.max_age : stamp - now <= options_.max_age;
  }

  /**
   * @brief Release expired inputs and close the collection window when none remain.
   *
   * @param now Current node-clock time.
   */
  void expire(Time now) {
    bool any_pending = false;
    for (auto& sample : pending_) {
      if (sample && !fresh(sample->stamp, now)) {
        sample.reset();
      }
      any_pending = any_pending || sample.has_value();
    }
    if (!any_pending) {
      window_start_.reset();
    }
  }

  /**
   * @brief Reset pending inputs and timestamp history after a backwards clock jump.
   *
   * @param now Current node-clock time to compare with the previous observation.
   */
  void observeTime(Time now) {
    if (last_now_ && now < *last_now_) {
      for (auto& sample : pending_) {
        sample.reset();
      }
      std::fill(last_received_stamp_.begin(), last_received_stamp_.end(), std::nullopt);
      window_start_.reset();
      last_emitted_stamp_.reset();
      next_publish_time_ = Time::zero();
    }
    last_now_ = now;
  }

  Options options_;
  std::vector<std::optional<Sample>> pending_;
  std::vector<std::optional<Time>> last_received_stamp_;
  std::optional<Time> window_start_;
  std::optional<Time> last_emitted_stamp_;
  std::optional<Time> last_now_;
  Time next_publish_time_{Time::zero()};
};

}  // namespace point_cloud_fusion
