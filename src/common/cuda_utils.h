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
 * \file src/common/cuda_utils.h
 * \brief Utilities for cuda
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mnm/device.h"

#define CUDA_CALL(func)                                           \
  do {                                                            \
    cudaError_t e = (func);                                       \
    CHECK(e == cudaSuccess) << "CUDA: " << cudaGetErrorString(e); \
  } while (false)

template <typename T, int value,
          typename std::enable_if<std::is_same<T, __half>::value, int>::type = 0>
inline const void* const_typed_addr() {
  float tmp = static_cast<float>(value);
  static const T a = static_cast<T>(tmp);
  return static_cast<const void*>(&a);
}

template <typename T, int value,
          typename std::enable_if<!std::is_same<T, __half>::value, int>::type = 0>
inline const void* const_typed_addr() {
  static const T a = static_cast<T>(value);
  return static_cast<const void*>(&a);
}

template <int value>
inline const void* const_addr(cudaDataType_t dt) {
  switch (dt) {
    case CUDA_R_8I:
      return const_typed_addr<int8_t, value>();
    case CUDA_R_8U:
      return const_typed_addr<uint8_t, value>();
    case CUDA_R_16F:
      return const_typed_addr<__half, value>();
    case CUDA_R_32F:
      return const_typed_addr<float, value>();
    case CUDA_R_64F:
      return const_typed_addr<double, value>();
    default:
      LOG(FATAL) << "Not supported data type!";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

template <typename T>
inline std::shared_ptr<void> shared_typed_addr(float value) {
  return std::make_shared<T>(value);
}

inline std::shared_ptr<void> shared_addr(cudaDataType_t dt, float value) {
  switch (dt) {
    case CUDA_R_8I:
      return shared_typed_addr<int8_t>(value);
    case CUDA_R_8U:
      return shared_typed_addr<uint8_t>(value);
    case CUDA_R_16F:
      return shared_typed_addr<__half>(value);
    case CUDA_R_32F:
      return shared_typed_addr<float>(value);
    case CUDA_R_64F:
      return shared_typed_addr<double>(value);
    default:
      LOG(FATAL) << "Not supported data type!";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

namespace mnm {

template <>
inline DType::operator cudaDataType_t() const {
  switch (code) {
    case kDLInt:
      if (bits == 8) return CUDA_R_8I;
      break;
    case kDLUInt:
      if (bits == 8) return CUDA_R_8U;
      break;
    case kDLFloat:
      if (bits == 16) return CUDA_R_16F;
      if (bits == 32) return CUDA_R_32F;
      if (bits == 64) return CUDA_R_64F;
    default:
      LOG(FATAL) << "NotImplementedError: " << c_str();
  }
  throw;
}

}  // namespace mnm
