/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/sharding.cc
 * \brief MNM Sharding System underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/sharding.h"
#include <string>

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::value;

ReplicatedSpec ReplicatedSpec::make(bool immutable) {
  auto n = make_object<ReplicatedSpecObj>();
  n->immutable = immutable;
  return ReplicatedSpec(n);
}

ShardSpec ShardSpec::make(bool immutable,
                          Array<Device> assigned_devices,
                          Array<Integer> num_devices_on_dim,
                          Array<Integer> num_replicas_on_dim) {
  auto n = make_object<ShardSpecObj>();
  n->immutable = immutable;
  n->assigned_devices = std::move(assigned_devices);
  n->num_devices_on_dim = std::move(num_devices_on_dim);
  n->num_replicas_on_dim = std::move(num_replicas_on_dim);
  return ShardSpec(n);
}

TupleShardSpec TupleShardSpec::make(bool immutable,
                                    Array<BaseShardSpec> tuple_elem) {
  auto n = make_object<TupleShardSpecObj>();
  n->immutable = immutable;
  n->tuple_elem = tuple_elem;
  return TupleShardSpec(n);
}

MNM_REGISTER_GLOBAL("mnm.sharding._make.ReplicatedSpec").set_body_typed(ReplicatedSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.TupleShardSpec").set_body_typed(TupleShardSpec::make);

MNM_REGISTER_OBJECT_NO_REFLECT(BaseShardSpecObj);
MNM_REGISTER_OBJECT_REFLECT(ReplicatedSpecObj);
MNM_REGISTER_OBJECT_REFLECT(ShardSpecObj);
MNM_REGISTER_OBJECT_REFLECT(TupleShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ReplicatedSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ReplicatedSpec>(ref);
      p->stream << "ReplicatedSpec("
                << (r->immutable ? "Immut" : "")
                << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ShardSpec>(ref);
      p->stream << "ShardSpec("
                << (r->immutable ? "Immut " : "");
      r.printAllocTable(p->stream);
      p->stream << "\b)";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<TupleShardSpec>(ref);
      p->stream << "TupleShardSpec(" 
                << (r->immutable ? "Immut" : "")
                << ")";
    });

TVM_REGISTER_NODE_TYPE(ShardOpAttrs);

}  // namespace sharding
}  // namespace mnm
