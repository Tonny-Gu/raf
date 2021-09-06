/*!
 * Copyright (c) 2019 by Contributors
 * \file stream_pool.h
 * \brief Stream pool API
 */
#pragma once
#include <memory>
#include <string>
#include "./device.h"
#include "./enum_base.h"

namespace mnm {
namespace stream_pool {

class Tag final {
 public:
  explicit Tag(const std::string& data) : data(data) {
    index = GetTagIndex_(data);
  }

 public:
  std::string data;
  int index;

 private:
  static int GetTagIndex_(const std::string& tag);
};

enum StreamTag {
  kUnknown = 0,
  kCudaCompute = 1,
  kMemCpyCpuToCuda = 2,
  kMemCpyCudaToCpu = 3,
  kMemCpyCudaToCuda = 4,
  kCudaCommunicate = 5,
  kReserved1 = 6,
  kReserved2 = 7,
  kReserved3 = 8,
  kReserved4 = 9,
  kReserved5 = 10,
  kReserved6 = 11,
  kReserved7 = 12,
  kReserved8 = 13,
  kReserved9 = 14,
  kReserved10 = 15,
};

class StreamTagEnum : public EnumBase<StreamTagEnum, 16, int32_t, StreamTag> {
 public:
  ENUM_DEF_HEADER(StreamTagEnum, 0, plain);
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 0, Unknown, kUnknown, "Unknown");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 1, CudaCompute, kCudaCompute, "Cuda compute");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 2, MemCpyCpuToCuda, kMemCpyCpuToCuda,
                           "Memcopy from CPU to CUDA");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 3, MemCpyCudaToCpu, kMemCpyCudaToCpu,
                           "Memcopy from CUDA to CPU");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 4, MemCudaToCuda, kMemCpyCudaToCuda,
                           "Memcopy from CUDA to CUDA");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 5, CudaCommunicate, kCudaCommunicate,
                           "Communicate between Cuda devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 6, Reserved1, kReserved1, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 7, Reserved2, kReserved2, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 8, Reserved3, kReserved3, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 9, Reserved4, kReserved4, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 10, Reserved5, kReserved5, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 11, Reserved6, kReserved6, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 12, Reserved7, kReserved7, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 13, Reserved8, kReserved8, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 14, Reserved9, kReserved9, "Reserved for other devices");
  ENUM_DEF_ENTRY_WITH_NAME(StreamTagEnum, 15, Reserved10, kReserved10,
                           "Reserved for other devices");
};

class Stream;

class Stream final {
 public:
  class Impl;
  friend Impl;

 public:
  Stream() = default;

  explicit Stream(Impl* impl);

  ~Stream();

  void* data() const;

  static std::shared_ptr<Stream> Get(const Device& dev, int tag_idx, int index);

  void Wait() const;

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace stream_pool
}  // namespace mnm
