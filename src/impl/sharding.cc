/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/sharding.cc
 * \brief MNM Sharding System underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/sharding.h"

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::value;

ShardSpec ShardSpec::make(bool immutable, bool replicated,
                          Array<IntValue> assigned_devices,
                          Array<IntValue> num_devices_on_dim,
                          Array<IntValue> num_replicas_on_dim) {
  ObjectPtr<ShardSpecObj> n = make_object<ShardSpecObj>();
  n->immutable = immutable;
  n->replicated = replicated;
  n->assigned_devices = std::move(assigned_devices);
  n->num_devices_on_dim = std::move(num_devices_on_dim);
  n->num_replicas_on_dim = std::move(num_replicas_on_dim);
  return ShardSpec(n);
}

MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
MNM_REGISTER_OBJECT_REFLECT(ShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* n = static_cast<const ShardSpecObj*>(ref.get());
      p->stream << "ShardSpec(" \
        << (n->replicated ? "Replicated " : "")
        << (n->immutable ? "Immutable " : "")
        << ")";
      // TODO: print out allocation table, sample:
      //   dim0: shard0@cuda(0), s1@cuda(1),
      //   dim1: shard2@cuda(2),cuda(3), s3@cuda(4),cuda(5)
    });

TVM_REGISTER_NODE_TYPE(ShardOpAttrs);

}  // namespace sharding
}  // namespace mnm