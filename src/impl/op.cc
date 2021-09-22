/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/op.cc
 * \brief MNM operator interface underlying implementation
 */
#include <tvm/runtime/device_api.h>
#include "dmlc/registry.h"
#include "mnm/executor.h"
#include "mnm/ir.h"
#include "mnm/op.h"
#include "mnm/dialect.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "mnm/device_api.h"
#include "../requests.h"
#include "../op/schema/list_args.h"

#ifdef MNM_USE_CUDA
#include "../op/dialect/cudnn/cudnn_utils.h"
#include "../op/dialect/cublas/cublas_utils.h"
#endif

namespace dmlc {
DMLC_REGISTRY_ENABLE(::mnm::op::OpEnvMaker);
}  // namespace dmlc

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using executor::Executor;
using requests::Requests;

CallValues CallValues::make(value::Value callee, ir::Attrs args) {
  ObjectPtr<CallValuesNode> n = make_object<CallValuesNode>();
  n->callee = std::move(callee);
  n->args = std::move(args);
  return CallValues(n);
}

// Implementation: OpEnv

class OpEnv::Impl : public Requests {
 public:
  executor::Executor* executor = nullptr;
};

OpEnv::OpEnv() : impl(new OpEnv::Impl()) {
}

OpEnv::~OpEnv() {
  if (impl->executor != nullptr) {
    impl->executor->OnDestruct(this);
  }
}

void OpEnv::RequestWorkspace(void** dest, const Device& dev, int64_t nbytes) {
  int index = impl->workspace.size();
  impl->workspace.push_back({dest, dev, nbytes, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestWorkspace(impl.get(), index);
  }
}

void OpEnv::RequestStream(void** dest, const Device& dev, int tag_idx) {
  int index = impl->stream.size();
  impl->stream.push_back({dest, dev, tag_idx, index, nullptr});
  if (impl->executor != nullptr) {
    impl->executor->RequestStream(impl.get(), index);
  }
}

void OpEnv::RequestDistributed(void** dest) {
  int index = impl->distributed.size();
  impl->distributed.push_back({dest});
  if (impl->executor != nullptr) {
    impl->executor->RequestDistributed(impl.get(), index);
  }
}

void OpEnv::BindExecutor(Executor* executor) {
  CHECK(impl->executor != nullptr);
  impl->executor = executor;
  executor->OnBind(this);
}

std::shared_ptr<Requests> OpEnv::GetRequests() const {
  return this->impl;
}

void OpEnv::SetStreamForAllBackends(Device device, void* stream) {
#ifdef MNM_USE_CUDA
  tvm::runtime::DeviceAPI::Get(device)->SetStream(device, stream);
  mnm::op::cudnn::SetStream(static_cast<cudaStream_t>(stream));
  mnm::op::cublas::SetStream(static_cast<cudaStream_t>(stream));
#endif
}

// Implementation: OpEnvMaker

OpEnvMaker& OpEnvMaker::set_name(const std::string& name) {
  this->name = name;
  return *this;
}

OpEnvMaker& OpEnvMaker::set_func(FMakeOpEnv func) {
  func_ = func;
  return *this;
}

OpEnvMaker::TRegistry* OpEnvMaker::Registry() {
  return TRegistry::Get();
}

const OpEnvMaker* OpEnvMaker::Get(const std::string& op_name) {
  return TRegistry::Get()->Find(op_name);
}

std::shared_ptr<OpEnv> OpEnvMaker::Make(const std::string& op_name, const CallValues& call) {
  auto maker = OpEnvMaker::Get(op_name);
  CHECK(maker) << "Cannot find an OpEnvMaker registered to " << op_name;
  auto env = (*maker)(call);
  return std::shared_ptr<OpEnv>(env);
}

// Implementation : helper functions

std::shared_ptr<OpEnv> DispatchSingleOp(const CallValues& call) {
  Op op = Downcast<OpValue>(call->callee)->op;
  std::string skip_dialect = "";
  if (IsDialectOp(op)) {
    // dialect op, directly call the OpEnvMaker registered to it
    auto env = OpEnvMaker::Make(op->name, call);
    if (env != nullptr) {
      DLOG(INFO) << "Dispatch to " << op->name;
      return env;
    }
    // failed to generate OpEnv, lift back to base op and try other dialects
    skip_dialect = GetDialect(op);
    auto base_op = GetBaseOp(op);
    base_op->op_type = op->op_type;
    op = base_op;
  }
  // Iterate over all dialect ops based on plevel.
  auto dialect_list = OpDialect::GetDispatchList(op, call->device.device_type());
  for (const auto& entry : dialect_list) {
    if (entry.dialect == skip_dialect) {
      continue;
    }
    auto dialect_op = Op::Get(entry.dialect_op);
    dialect_op->op_type = op->op_type;
    if (auto env = OpEnvMaker::Make(dialect_op->name, call)) {
      DLOG(INFO) << "Dispatch to " << dialect_op->name;
      return env;
    }
  }
  LOG(FATAL) << "Cannot find a valid dispatch for op " << op->name;
  return nullptr;
}

std::shared_ptr<OpEnv> DispatchFusedOp(const CallValues& call) {
  auto clo = Downcast<ClosureValue>(call->callee);
  auto func = clo->func;
  ICHECK(func->HasNonzeroAttr(attr::kPrimitive))
      << "Encountered a non-primitive function when dispatching a call";
  auto dialect = func->GetAttr<String>(attr::kDialect);
  ICHECK(dialect.defined()) << "Fused function doesn't have dialect attribute: "
                            << ir::AsText(func);
  std::ostringstream os;
  os << "mnm.op." << dialect.value() << "._fused_op";
  return OpEnvMaker::Make(os.str(), call);
}

std::shared_ptr<OpEnv> Dispatch(const CallValues& call) {
  if (call->callee.as<value::OpValueObj>()) {
    return DispatchSingleOp(call);
  } else if (call->callee.as<value::ClosureValueObj>()) {
    return DispatchFusedOp(call);
  }
  LOG(FATAL) << "call->op type " << call->callee->GetTypeKey() << " unsupported";
  return nullptr;
}

Attrs MakeListArgs(const Array<Value>& values) {
  auto attrs = make_object<schema::ListArgs>();
  attrs->args = values;
  return Attrs(attrs);
}

Array<Value> GetListArgs(const Attrs& attrs) {
  return attrs.as<schema::ListArgs>()->args;
}

std::string GetUniqueName(std::string name) {
  static std::unordered_map<std::string, int> name_map;
  for (size_t i = 0; i < name.length(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  while (true) {
    auto it = name_map.find(name);
    if (it == name_map.end()) {
      name_map[name] = 1;
      return name;
    } else {
      std::ostringstream os;
      os << name << "_" << it->second;
      ++(it->second);
      name = os.str();
    }
  }
  return name;
}

std::string TruncateName(std::string name) {
  constexpr static size_t kMaxFuncNameLength = 80;
  if (name.size() > kMaxFuncNameLength) {
    std::stringstream truncated_name;
    truncated_name << name.substr(0, kMaxFuncNameLength);
    truncated_name << "_" << std::hash<std::string>{}(name) << "_";
    name = truncated_name.str();
  }
  return name;
}

Op GetOp(const std::string& op_name) {
  return Op::Get(op_name);
}

MNM_REGISTER_GLOBAL("mnm.op.GetOp").set_body_typed(GetOp);

}  // namespace op
}  // namespace mnm
