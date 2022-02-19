/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file memory_profiler.h
 * \brief memory profiler
 */
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "device.h"

#define PROFILE_MEMORY(DEVICE, TAG)                                     \
  {                                                                     \
    if (mnm::memory_profiler::MemoryProfiler::Get()->IsProfiling()) {   \
      mnm::memory_profiler::MemoryProfiler::Get()->Record(DEVICE, TAG); \
    }                                                                   \
  }

namespace mnm {
namespace memory_profiler {

using FloatPair = std::pair<float, float>;

/*! \brief A memory trace unit that records the memory usage of a device at the moement. */
struct MemoryTrace {
  float used = 0;
  float allocated = 0;
  std::string tag = "";

  MemoryTrace(){};

  MemoryTrace(float used, float allocated, std::string tag)
      : used(used), allocated(allocated), tag(tag) {
  }

  bool operator>(const MemoryTrace& other) const {
    return used > other.used;
  }

  bool operator<(const MemoryTrace& other) const {
    return !(this->operator>(other));
  }
};

/*! \brief Memory stats of a device. */
struct MemoryStat {
  /*! \brief A sequence of memory traces. */
  std::vector<MemoryTrace> traces;
  /*! \brief The trace index that achieves the peak memory usage. */
  int max_trace_idx = 0;
  /*! \brief The number of triggered garbage collections. */
  int num_gc = 0;
};

/*! \brief The memory profiler for all devices. */
class MemoryProfiler {
 public:
  static MemoryProfiler* Get();
  ~MemoryProfiler();

  void SetProfile(bool profile) {
    is_profiling_ = profile;
  }

  bool IsProfiling() {
    return is_profiling_;
  }

  /*!
   * \brief Record the current used and allocated memory for the given device and tag.
   * \param device The device to record.
   * \param tag The tag of this record.
   */
  void Record(const Device& device, const std::string& tag);

  /*! \brief Reset all memory stats. */
  void Reset();

  /*!
   * \brief Get max memory info.
   * \param device The device to get the memory stats.
   * \return The memory stats of the device, including max used, max allocated,
   * the max trace index and number of triggerd GCs.
   */
  Map<String, FloatImm> GetMaxMemoryInfo(const Device& device);

  /*!
   * \brief Get the memory trace of the given device.
   * \param device The device to get the memory trace.
   * \return The memory trace table of the device in a pretty string.
   */
  std::string GetMemoryTrace(const Device& device);

 private:
  /*! \brief Mapping from device string to memory stats. */
  std::unordered_map<std::string, MemoryStat> memory_stats_;
  /*! \brief Whether the profiling is enabled. */
  bool is_profiling_ = false;
};
}  // namespace memory_profiler
}  // namespace mnm
