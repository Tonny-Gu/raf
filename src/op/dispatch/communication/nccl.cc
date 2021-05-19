/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/dispatch/communication/nccl.cc
 * \brief Communication operators implmentated by NCCL
 */
#include <vector>
#include "mnm/op_utils.h"
#include "mnm/dist_context.h"
#include "../../schema/communication.h"
#include "./communication_utils.h"
#include <chrono>
#include <thread>

namespace mnm {
namespace op {
namespace communication {
namespace nccl {
using namespace distributed;
using namespace distributed::communicator;
using common::shape_utils::BytesCompactTensor;
using stream_pool::StreamTagEnum;

class NCCLAllReduce : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  DType dtype;

  explicit NCCLAllReduce(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::AllreduceArgs>();
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto& tv = args->x;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() == 1) return;
    RequestWorkspace(&fused_data, cv->device, total_size);
  }

 public:
  ~NCCLAllReduce() {
    // Nothing
  }
  void Execute(const CallValues& cv) {
    // // We can use sleep to test communication scheduling locally.
    // using namespace std::this_thread;
    // using namespace std::chrono;
    // sleep_until(system_clock::now() + nanoseconds(200));

    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto args = cv->args.as<mnm::op::schema::AllreduceArgs>();
    // Fuse Tensor
    auto& tv = args->x;
    size_t dtype_size = 0;
    if (tv.size() == 1) {
      DLTensor* x = tv[0];
      DLTensor* out = cv->out;
      dtype_size = sizeof(x->dtype);
      NCCL_CALL(ncclAllReduce(x->data, out->data, total_size / dtype_size, dtype, ncclSum,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      return;
    }
    size_t offset = 0;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                      (cudaStream_t)stream);
      offset += tuple_sizes[i];
      dtype_size = sizeof(x->dtype);
    }

    // Allreduce
    NCCL_CALL(ncclAllReduce(fused_data, fused_data, total_size / dtype_size, dtype, ncclSum,
                            (ncclComm_t)nccl_comm, (cudaStream_t)stream));
    // UnFuse Tensor
    value::TupleValue out = tvm::runtime::Downcast<value::TupleValue>(cv->out);
    auto& of = out->fields;
    for (int i = of.size() - 1; i >= 0; --i) {
      DLTensor* x = of[i];
      offset -= tuple_sizes[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                      (cudaStream_t)stream);
    }
  }
  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    // TODO(@icemelon9): add implementation for this
    LOG(FATAL) << "Not implemented";
  }
  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllReduce(cv);
  }
};
MNM_OP_DISPATCH("mnm.op._allreduce", NCCLAllReduce::make, DevType::kCUDA(), "nccl_communication");

class NCCLAllGather : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  explicit NCCLAllGather(const CallValues& cv) {
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
  }

 public:
  ~NCCLAllGather() {
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::AllgatherArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    DLTensor* x = inputs[0];
    DLTensor* out = output;
    int64_t size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      size *= x->shape[i];
    }
    NCCL_CALL(ncclAllGather(x->data, out->data, size, DType(x->dtype), (ncclComm_t)nccl_comm,
                            (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllGather(cv);
  }
};

MNM_OP_DISPATCH("mnm.op._allgather", NCCLAllGather::make, DevType::kCUDA(), "nccl_communication");

class NCCLReduceScatter : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  void* in_buffer;
  size_t size_in_bytes;
  size_t size;

  explicit NCCLReduceScatter(const CallValues& cv) {
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    const DLTensor* out = cv->out;
    size_in_bytes = BytesCompactTensor(*out);
    size = size_in_bytes / (out->dtype.bits / 8);
    RequestWorkspace(&in_buffer, cv->device, size_in_bytes * DistContext::Global()->size);
  }

 public:
  ~NCCLReduceScatter() {
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::ReduceScatterArgs>();
    Execute(std::vector<value::Value>(args->x.begin(), args->x.end()), cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    size_t offset = 0;
    DLTensor* out = output;
    DType dtype;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const DLTensor* x = inputs[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(in_buffer) + size_in_bytes * i;
      cudaMemcpyAsync(buffer_data_at_offset, x->data, size_in_bytes, cudaMemcpyDeviceToDevice,
                      (cudaStream_t)stream);
      dtype = x->dtype;
    }
    NCCL_CALL(ncclReduceScatter(in_buffer, out->data, size, dtype, ncclSum, (ncclComm_t)nccl_comm,
                                (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLReduceScatter(cv);
  }
};

MNM_OP_DISPATCH("mnm.op._reduce_scatter", NCCLReduceScatter::make, DevType::kCUDA(),
                "nccl_communication");

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace mnm
