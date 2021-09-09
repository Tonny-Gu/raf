/*!
 * Copyright (c) 2019 by Contributors
 * \file device_api.h
 * \brief Unified low-level API for heterogeneous devices
 */
#pragma once
#include <memory>
#include "./device.h"

namespace mnm {
namespace device_api {

// TODO(@junrushao1994): To pass flags to stream/event/..., do we add thread_local flags?
class DeviceAPI {
 public:
  virtual ~DeviceAPI() = default;

  /*!
   * Allocate a chuck of memory.
   * \param nbytes The size of memory in bytes to allocate.
   * \param alignment The alignment size.
   * \return The allocated memory.
   */
  virtual void* AllocMemory(int64_t nbytes, int64_t alignment) = 0;

  /*!
   * Allocate a chuck of memory asynchronously.
   * \param nbytes The size of memory in bytes to allocate.
   * \param stream The stream to place the allocation on.
   * \param alignment The alignment size.
   * \return The allocated memory.
   */
  virtual void* AllocMemoryAsync(int64_t nbytes, void* stream, int64_t alignment) = 0;

  /*!
   * Free the allocated memory.
   * \param ptr The allocated memory to be freed.
   */
  virtual void FreeMemory(void* ptr) = 0;

  /*!
   * Free the memory asynchronously.
   * \param ptr The allocated memory to be freed.
   * \param stream The stream to place the free operation on.
   */
  virtual void FreeMemoryAsync(void* ptr, void* stream) = 0;

  // If the device API itself has a memory pool, this API is used to query
  // the current pool status (used memory, allocated memory) in bytes.
  /*!
   * Query the memory pool size of the underlying memory pool of this device, if applicable.
   * \return <used, allocated> 'used' is the number of bytes that has been allocated to the user,
   * and the 'allocated' is the number of bytes that has been allocated from the system.
   */
  virtual std::pair<int64_t, int64_t> GetPoolSize() {
    return std::make_pair(0, 0);
  };

  // Set the device for memory allocation. This API is for GPU only,
  // CPU should never call this API
  virtual void SetDevice(const int device_id) = 0;

  /*!
   * Create a stream on given device.
   * \param dev The device to create the stream.
   * \return The created stream.
   */
  virtual void* CreateStream(const Device& dev) = 0;

  /*!
   * Free a stream.
   * \param dev The device to free the stream.
   * \param stream The stream.
   */
  virtual void FreeStream(const Device& dev, void* stream) = 0;

  /*!
   * Create an event on given device.
   * \param dev The device to create the event.
   * \param flags The flags of the event. The value depends on the underlying device. For cuda
   * device, refers to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html for
   * the available flags.
   * \return The created event.
   */
  virtual void* CreateEvent(const Device& dev, uint32_t flags = 0) = 0;

  /*!
   * Free an event.
   * \param dev The device of the event.
   * \param event The event.
   */
  virtual void FreeEvent(const Device& dev, void* event) = 0;

  /*!
   * Place an event on given stream. It would record the pending workloads on that stream.
   * \param dev The device of the event and stream.
   * \param event The event to record workloads.
   * \param stream The stream to be recorded.
   */
  virtual void EventRecordOnStream(const Device& dev, void* event, void* stream) = 0;

  /*!
   * Let a stream wait for an event. This call is asynchronous. All workloads issued to given
   * stream would be executed after the workloads recorded by the event.
   * \param dev The device of the event and stream.
   * \param stream The stream to wait for the event.
   * \param event The event to be waited for.
   */
  virtual void StreamWaitEvent(const Device& dev, void* stream, void* event) = 0;

  // will call the device api of `next_ctx` to wait for `prev`
  // therefore, we should the assumption that `after.device == device_api.device`
  virtual void SyncStream(const Device& prev_dev, void* prev, void* next) = 0;

  // Granularity of synchronization
  /*!
   * Synchronize the device. It would block the host thread until all pending workloads on the given
   * device finished.
   * \param dev The device to wait.
   */
  virtual void WaitDevice(const Device& dev) = 0;

  /*!
   * Synchronize the stream. It would block the host thread until all pending workloads on the given
   * stream finished.
   * \param dev The device of the stream.
   * \param stream The stream to wait.
   */
  virtual void WaitStream(const Device& dev, void* stream) = 0;

 public:
  /*!
   * The the device api of given device type
   * \param device_type The device type.
   * \return The device api.
   */
  static std::shared_ptr<DeviceAPI> Get(DevType device_type);
};

}  // namespace device_api
}  // namespace mnm
