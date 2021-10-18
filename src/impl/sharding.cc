/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/sharding.cc
 * \brief MNM Sharding System underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include "mnm/ir.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/registry.h"
#include "mnm/sharding.h"
#include "mnm/dist_context.h"
#include "../op/schema/ufunc.h"
#include "../op/schema/sharding.h"
#include <string>

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::op::schema;
using namespace mnm::value;
using namespace mnm::distributed;

ReplicatedSpec ReplicatedSpec::make(bool immutable) {
  auto n = make_object<ReplicatedSpecObj>();
  n->immutable = immutable;
  return ReplicatedSpec(n);
}

ShardSpec ShardSpec::make(bool immutable,
                          Array<Device> devices_in_grid,
                          Array<Integer> partition_shape,
                          Array<Integer> subgroup_sizes) {
  auto ndim = partition_shape.size();
  CHECK_EQ(ndim, subgroup_sizes.size());
  auto n = make_object<ShardSpecObj>();
  auto subgroup_idx = std::vector<Integer>(ndim);
  auto grid_shape = std::vector<Integer>(ndim);

  int64_t device_rank = -1;
  for (int64_t i = 0; i < devices_in_grid.size(); ++i) {
    if (DistContext::Global()->local_device.same_as(devices_in_grid[i])) {
      device_rank = i;
      break;
    }
  } // perhaps it is improper to calculate runtime data here

  for (int64_t i = ndim - 1; i >= 0; --i) {
    grid_shape[i] = partition_shape[i]->value / subgroup_sizes[i]->value;
    subgroup_idx[i] = device_rank % grid_shape[i]->value;
    device_rank /= grid_shape[i]->value;
  }

  n->immutable = immutable;
  n->devices_in_grid = std::move(devices_in_grid);
  n->grid_shape = Array<Integer>(grid_shape.begin(), grid_shape.end());
  n->subgroup_sizes = std::move(subgroup_sizes);
  n->subgroup_idx = (device_rank != -1) ? Array<Integer>(subgroup_idx.begin(), subgroup_idx.end()) : 
                                          NullValue<Array<Integer>>();
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

void GetSliceRange(const CallValues& call) {
  const auto* args = call->args.as<GetSliceRangeArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  auto spec = Downcast<ShardSpec>(args->spec);
  auto ndim = x->ndim;
  CHECK_EQ(x->ndim, spec->grid_shape.size());
  std::vector<Value> begin(ndim);
  std::vector<Value> end(ndim);
  if (spec->subgroup_idx.defined()) {
    for (int i = 0; i < ndim; ++i) {
      auto idx = spec->subgroup_idx[i]->value;
      auto num = spec->grid_shape[i]->value;
      CHECK_EQ(x->shape[i] % num, 0) << "Currently automaic padding is unsupported.";
      begin[i] = ScalarValue::make((x->shape[i] / num) * idx);
      end[i] = ScalarValue::make((x->shape[i] / num) * (idx + 1) - 1);
    }
    call->out = TupleValue::make({
      TupleValue::make(begin),
      TupleValue::make(end)
    });
  } else {
    call->out = ir::NullValue<Value>();
  }
  call->callee = ir::NullValue<OpValue>();
}

MNM_OP_DECLARE("mnm.op._get_slice_range", GetSliceRange);

MNM_REGISTER_GLOBAL("mnm.sharding._make.ReplicatedSpec").set_body_typed(ReplicatedSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.TupleShardSpec").set_body_typed(TupleShardSpec::make);
MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardOpAttrs").set_body_typed(ShardOpAttrs::make);

MNM_REGISTER_OBJECT_NO_REFLECT(BaseShardSpecObj);
MNM_REGISTER_OBJECT_REFLECT(ReplicatedSpecObj);
MNM_REGISTER_OBJECT_REFLECT(ShardSpecObj);
MNM_REGISTER_OBJECT_REFLECT(TupleShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;

void PrintAllocTable(const ObjectRef& ref, ReprPrinter* p) {
  /*size_t dev_idx = 0;
  const auto obj = Downcast<ShardSpec>(ref);
  const auto num_dim = obj->grid_shape.size();
  static thread_local size_t *indices = new size_t[num_dim];
  std::function<void(int)> _print_alloc_table;
  _print_alloc_table = [&](int depth) {
    if (depth == num_dim) {
      p->stream << (dev_idx != 0 ? " [" : "[");
      for (size_t i = 0; i < num_dim; ++i) {
        auto num_devices = obj->grid_shape[i]->value;
        auto index = std::to_string(indices[i]);
        p->stream << (num_devices == 1 ? ":" : index)
                  << (i != num_dim - 1 ? ", " : "");
      }
      auto dev_info = obj->devices_in_grid[dev_idx++].c_str();
      p->stream << "]@" << dev_info;
    } else {
      auto subgroup_num = obj->grid_shape[depth]->value;
      for (size_t i = 0; i < subgroup_num; ++i) {
        indices[depth] = i;
        _print_alloc_table(depth + 1);
      }
    }
  };
  _print_alloc_table(0);*/
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
      auto ndim = r->grid_shape.size();
      p->stream << "ShardSpec("
                << (r->immutable ? "Immut " : "")
                << "[";
      for (size_t i = 0; i < ndim; ++i) {
        auto subgroup_num = r->grid_shape[i]->value;
        auto subgroup_size = r->subgroup_sizes[i]->value;
        p->stream << (subgroup_num == 1 ? ":" : std::to_string(subgroup_num))
                  << (subgroup_size == 1 ? "" : "(x" + std::to_string(subgroup_size) + ")")
                  << (i != ndim - 1 ? ", " : "");
      }
      p->stream << "])";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<TupleShardSpec>(ref);
      p->stream << "TupleShardSpec" 
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
