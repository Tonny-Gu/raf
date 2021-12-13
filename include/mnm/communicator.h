/*!
 * Copyright (c) 2019 by Contributors
 * \file communicator.h
 * \brief Communication resources.
 */
#pragma once
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <memory>
#include "dmlc/logging.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "./connector.h"

namespace mnm {
namespace distributed {
namespace communicator {

using connector::Connector;
using connector::ConnectorManager;
using registry::GetPackedFunc;

class Communicator {
 public:
  Communicator() {
  }
  virtual ~Communicator() {
  }
  int GetLocalSize() {
    return connector_->local_size;
  }
  int GetLocalRank() {
    return connector_->local_rank;
  }
  int GetSize() {
    return connector_->size;
  }
  int GetRank() {
    return connector_->rank;
  }
  int GetRootRank() {
    return root_rank;
  }
  bool IsRoot() {
    return GetRank() == GetRootRank();
  }
  virtual void* GetCommHandle() = 0;

 protected:
  virtual void Init() = 0;
  virtual void Finalize() = 0;
  void GetConnector(const std::string& name = "mpi") {
    connector_.reset(ConnectorManager::Get()->GetConnector(name));
  }

 public:
  std::string type;
  int root_rank = 0;
  std::shared_ptr<Connector> connector_;
};

class CommunicatorManager {
 public:
  // TODO: support multiple communicators.
  CommunicatorManager() {
    comm_world_ = nullptr;
  }
  static CommunicatorManager* Get() {
    static CommunicatorManager* instance = new CommunicatorManager();
    return instance;
  }

  Communicator* GetCommunicator(const std::string& name = "") {
    CHECK_LT(name.size(), 128) << "There is no such communicator: " << name;
    thread_local char maker_name[128];

    std::string default_name = "nccl";
    snprintf(maker_name, sizeof(maker_name), "mnm.distributed.communicator._make.%s",
             default_name.c_str());
    const registry::PackedFunc* pf = registry::Registry::Get(maker_name);
    if (pf == nullptr) default_name = "void";

    if (comm_world_ == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (comm_world_ == nullptr) {
        // ok, it is truly a nullptr
        if (name == "") {
          snprintf(maker_name, sizeof(maker_name), "mnm.distributed.communicator._make.%s",
                   default_name.c_str());
        } else {
          if (name != "void") CHECK_EQ(name, "nccl") << "Unsupported communicator: " << name;
          snprintf(maker_name, sizeof(maker_name), "mnm.distributed.communicator._make.%s",
                   name.c_str());
        }
        void* ret = GetPackedFunc(maker_name)();
        comm_world_.reset(static_cast<Communicator*>(ret));
        return comm_world_.get();
      }
    }
    // otherwise this is not nullptr
    CHECK_EQ(name, "") << "You have already initialized a communicator [" << comm_world_->type
                       << "], and currently we do not support multiple communicators";
    return comm_world_.get();
  }

  void Remove() {
    std::lock_guard<std::mutex> lock(mutex_);
    comm_world_ = nullptr;
  }

 public:
  std::map<std::vector<int64_t>, std::shared_ptr<Communicator>> comm_;
  std::shared_ptr<Communicator> comm_world_;
  std::mutex mutex_;
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
