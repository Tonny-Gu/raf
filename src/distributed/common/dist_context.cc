/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/context.cc
 * \brief Context of Distributed Settings.
 */
#include "raf/registry.h"
#include "raf/dist_context.h"

namespace raf {
namespace distributed {

using communicator::Communicator;
using communicator::CommunicatorManager;

DistContext DistContext::make() {
  ir::ObjectPtr<DistContextObj> n = ir::make_object<DistContextObj>();
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  n->root_rank = comm->GetRootRank();
  n->rank = comm->GetRank();
  n->size = comm->GetSize();
  n->local_rank = comm->GetLocalRank();
  n->local_size = comm->GetLocalSize();
  auto dev_type = comm->type == "NCCL" ? DevType::kCUDA() : DevType::kCPU();
  // TODO: it is improper to assume we only have two types of devices supporting MPI-like library
  for (int i = 0; i < n->size; ++i) {
    auto device = Device::make(dev_type, i);
    n->dist_devices.push_back(device);
    if (i == n->rank)
      n->local_device = device;
  }
  return DistContext(n);
}

DistContext DistContext::Global() {
  static DistContext inst = DistContext::make();
  return inst;
}

void EnableDataParallel(bool enable) {
  DistContext::Global()->enable_data_parallel = enable;
}

void ZeroOpt(int opt_level) {
  DistContext::Global()->zero_opt_level = opt_level;
}

void SetGlobalRank(int rank) {
  DistContext::Global()->rank = rank;
}

void SetGlobalSize(int size) {
  DistContext::Global()->size = size;
}

void AutoDPProfilingStartIter(int auto_dp_profiling_start_iter) {
  DistContext::Global()->auto_dp_profiling_start_iter = auto_dp_profiling_start_iter;
}

void AutoDPProfilingEndIter(int auto_dp_profiling_end_iter) {
  DistContext::Global()->auto_dp_profiling_end_iter = auto_dp_profiling_end_iter;
}

RAF_REGISTER_GLOBAL("raf.distributed._make.DistContext").set_body_typed(DistContext::make);
RAF_REGISTER_GLOBAL("raf.distributed.Global").set_body_typed(DistContext::Global);
RAF_REGISTER_GLOBAL("raf.distributed.EnableDataParallel").set_body_typed(EnableDataParallel);
RAF_REGISTER_GLOBAL("raf.distributed.ZeroOpt").set_body_typed(ZeroOpt);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalRank").set_body_typed(SetGlobalRank);
RAF_REGISTER_GLOBAL("raf.distributed.SetGlobalSize").set_body_typed(SetGlobalSize);
RAF_REGISTER_GLOBAL("raf.distributed.AutoDPProfilingStartIter")
    .set_body_typed(AutoDPProfilingStartIter);
RAF_REGISTER_GLOBAL("raf.distributed.AutoDPProfilingEndIter")
    .set_body_typed(AutoDPProfilingEndIter);

RAF_REGISTER_OBJECT_REFLECT(DistContextObj);

}  // namespace distributed
}  // namespace raf
