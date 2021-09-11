/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/sharding.cc
 * \brief MNM Sharding System underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/sharding.h"
#include "mnm/dist_context.h"
#include <string>

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::distributed;

ReplicatedSpec ReplicatedSpec::make(bool immutable) {
  auto n = make_object<ReplicatedSpecObj>();
  n->immutable = immutable;
  return ReplicatedSpec(n);
}

ShardSpec ShardSpec::make(bool immutable,
                          Array<Device> assigned_devices,
                          Array<Integer> num_devices_on_dim,
                          Array<Integer> num_replicas_on_dim) {
  CHECK_EQ(num_devices_on_dim.size(), num_replicas_on_dim.size());
  auto n = make_object<ShardSpecObj>();
  auto ndim = num_devices_on_dim.size();
  auto shard_dim = std::vector<Integer>(ndim);
  auto shard_idx = std::vector<Integer>(ndim);

  int64_t shard_id = -1;
  for (int64_t i = 0; i < assigned_devices.size(); ++i) {
    if (DistContext::Global()->local_device.same_as(assigned_devices[i])) {
      shard_id = i;
      break;
    }
  }
  for (int64_t i = ndim - 1; i >= 0; --i) {
    auto num_devices = num_devices_on_dim[i]->value;
    auto num_replicas = num_replicas_on_dim[i]->value;
    shard_dim[i] = num_devices / num_replicas;
    shard_idx[i] = (shard_id % num_devices) / num_replicas;
    shard_id /= num_devices;
  }

  n->immutable = immutable;
  n->assigned_devices = std::move(assigned_devices);
  n->num_devices_on_dim = std::move(num_devices_on_dim);
  n->num_replicas_on_dim = std::move(num_replicas_on_dim);
  n->_shard_dim = Array<Integer>(shard_dim.begin(), shard_dim.end());
  n->_shard_idx = Array<Integer>(shard_idx.begin(), shard_idx.end());

  return ShardSpec(n);
}

TupleShardSpec TupleShardSpec::make(bool immutable,
                                    Array<BaseShardSpec> tuple_elem) {
  auto n = make_object<TupleShardSpecObj>();
  n->immutable = immutable;
  n->tuple_elem = tuple_elem;
  return TupleShardSpec(n);
}

Attrs ShardOpAttrs::make(BaseShardSpec shard_in, BaseShardSpec shard_out) {
  auto attrs = make_object<ShardOpAttrs>();
  attrs->shard_in = std::move(shard_in);
  attrs->shard_out = std::move(shard_out);
  return Attrs(attrs);
}

static thread_local bool print_brief_alloc_table = false;

MNM_REGISTER_GLOBAL("mnm.sharding._make.ReplicatedSpec").set_body_typed(ReplicatedSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.TupleShardSpec").set_body_typed(TupleShardSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardOpAttrs").set_body_typed(ShardOpAttrs::make);
MNM_REGISTER_GLOBAL("mnm.sharding.TogglePrintMode").set_body_typed([&]() {
  print_brief_alloc_table = !print_brief_alloc_table;
});

MNM_REGISTER_OBJECT_NO_REFLECT(BaseShardSpecObj);
MNM_REGISTER_OBJECT_REFLECT(ReplicatedSpecObj);
MNM_REGISTER_OBJECT_REFLECT(ShardSpecObj);
MNM_REGISTER_OBJECT_REFLECT(TupleShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;

void PrintAllocTable(const ObjectRef& ref, ReprPrinter* p) {
  size_t dev_idx = 0;
  const auto obj = Downcast<ShardSpec>(ref);
  const auto num_dim = obj->num_devices_on_dim.size();
  static thread_local size_t *indices = new size_t[num_dim];
  std::function<void(int)> _print_alloc_table;
  _print_alloc_table = [&](int depth) {
    if (depth == num_dim) {
      p->stream << (dev_idx != 0 ? " [" : "[");
      for (size_t i = 0; i < num_dim; ++i) {
        auto num_devices = obj->num_devices_on_dim[i]->value;
        auto num_replicas = obj->num_replicas_on_dim[i]->value;
        auto index = std::to_string(indices[i] / num_replicas);
        p->stream << (num_devices == 1 ? ":" : index)
                  << (i != num_dim - 1 ? ", " : "");
      }
      auto dev_info = obj->assigned_devices[dev_idx++].c_str();
      p->stream << "]@" << dev_info;
    } else {
      auto num_devices = obj->num_devices_on_dim[depth]->value;
      for (size_t i = 0; i < num_devices; ++i) {
        indices[depth] = i;
        _print_alloc_table(depth + 1);
      }
    }
  };
  _print_alloc_table(0);
}

void PrintBriefAllocTable(const ObjectRef& ref, ReprPrinter* p) {
  const auto obj = Downcast<ShardSpec>(ref);
  const auto num_dim = obj->num_devices_on_dim.size();
  p->stream << "[";
  for (size_t i = 0; i < num_dim; ++i) {
    auto num_devices = obj->num_devices_on_dim[i]->value;
    auto num_replicas = obj->num_replicas_on_dim[i]->value;
    p->stream << (num_devices == 1 ? ":" : std::to_string(num_devices))
              << (num_replicas == 1 ? "" : "/" + std::to_string(num_replicas))
              << (i != num_dim - 1 ? ", " : "");
  }
  p->stream << "]";
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ReplicatedSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ReplicatedSpec>(ref);
      p->stream << "ReplicatedSpec"
                << (r->immutable ? "(Immut)" : "");
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ShardSpec>(ref);
      p->stream << "ShardSpec("
                << (r->immutable ? "Immut " : "");
      print_brief_alloc_table ? 
        PrintBriefAllocTable(ref, p) : PrintAllocTable(ref, p);
      p->stream << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<TupleShardSpec>(ref);
      p->stream << "TupleShardSpec(" 
                << (r->immutable ? "(Immut)" : "");
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardOpAttrs>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* n = static_cast<const ShardOpAttrs*>(ref.get());
      p->stream << "ShardOpAttrs("
                << "in=" << n->shard_in
                << " out=" << n->shard_out
                << ")";
    });

TVM_REGISTER_NODE_TYPE(ShardOpAttrs);

}  // namespace sharding
}  // namespace mnm
