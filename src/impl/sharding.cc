/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/sharding.cc
 * \brief RAF Sharding System underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include "raf/ir.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/type.h"
#include "raf/registry.h"
#include "raf/sharding.h"
#include "raf/dist_context.h"
#include "../op/ty/utils.h"
#include "../op/schema/ufunc.h"
#include "../op/schema/sharding.h"
#include "../op/dialect/tvm/tvm_utils.h"
#include "../op/dialect/tvm/tvm_attrs.h"
#include <string>

namespace raf {
namespace sharding {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::op::schema;
using namespace raf::value;
using namespace raf::distributed;

ReplicatedSpec ReplicatedSpec::make(bool immutable) {
  auto n = make_object<ReplicatedSpecObj>();
  n->immutable = immutable;
  return ReplicatedSpec(n);
}

ShardSpec ShardSpec::make(bool immutable, Array<Integer> ranks, Array<Integer> phy_shape,
                          Array<Integer> replicas) {
  auto ndim = phy_shape.size();
  CHECK_EQ(ndim, replicas.size());
  auto n = make_object<ShardSpecObj>();
  auto phy_index = std::vector<Integer>(ndim);
  auto logic_index = std::vector<Integer>(ndim);
  auto logic_shape = std::vector<Integer>(ndim);

  int64_t rank_idx = -1;
  for (int64_t i = 0; i < ranks.size(); ++i) {
    if (DistContext::Global()->rank == ranks[i]->value) {
      rank_idx = i;
      break;
    }
  }

  for (int64_t i = ndim - 1; i >= 0; --i) {
    logic_shape[i] = phy_shape[i]->value / replicas[i]->value;
    phy_index[i] = rank_idx % phy_shape[i]->value;
    logic_index[i] = phy_index[i]->value / replicas[i]->value;
    rank_idx /= phy_shape[i]->value;
  }

  n->immutable = immutable;
  n->ranks = std::move(ranks);
  n->replicas = std::move(replicas);
  n->phy_shape = std::move(phy_shape);
  n->logic_shape = Array<Integer>(logic_shape);
  if (rank_idx == -1) {
    n->phy_index = NullValue<Array<Integer>>();
    n->logic_index = NullValue<Array<Integer>>();
  } else {
    n->phy_index = Array<Integer>(phy_index);
    n->logic_index = Array<Integer>(logic_index);
  }

  return ShardSpec(n);
}

TupleShardSpec TupleShardSpec::make(bool immutable, Array<BaseShardSpec> tuple_elem) {
  auto n = make_object<TupleShardSpecObj>();
  n->immutable = immutable;
  n->tuple_elem = tuple_elem;
  return TupleShardSpec(n);
}

Attrs ShardOpCallAttrs::make(BaseShardSpec shard_in, BaseShardSpec shard_out) {
  auto attrs = make_object<ShardOpCallAttrs>();
  attrs->shard_in = std::move(shard_in);
  attrs->shard_out = std::move(shard_out);
  return Attrs(attrs);
}

void Reshard_R2S(const CallValues& call) {
  const auto* args = call->args.as<ShardUnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  auto spec = Downcast<ShardSpec>(args->spec);
  if (spec->logic_index.defined()) {
    for (int64_t i = 0; i < x->ndim; ++i) {
      auto shard_dim_size = spec->logic_shape[i]->value;
      CHECK_EQ(x->shape[i] % shard_dim_size, 0) << "Currently automaic padding is unsupported.";
      shape[i] /= shard_dim_size;
    }
    call->out = TensorValue::Assemble(/*dev=*/x->device,
                                      /*dtype=*/x->dtype,
                                      /*shape=*/shape);
  } else {
    // idle when this local machine doesn't involve
    call->out = ir::NullValue<Value>();
    call->callee = ir::NullValue<OpValue>();
  }
  call->device = x->device;
}

RAF_OP_DECLARE("raf.op._reshard_r2s", Reshard_R2S);

Type Reshard_R2S_Infer(const CallValues& call) {
  const auto* args = call->args.as<ShardUnaryArgs>();
  CHECK(args != nullptr);
  auto spec = Downcast<ShardSpec>(args->spec);
  auto data = Downcast<TensorType>(GetType(args->x));
  Array<PrimExpr> dshape = data->shape;
  size_t ndim = dshape.size();
  std::vector<PrimExpr> oshape(ndim);
  CHECK(spec.defined());
  CHECK(spec->logic_index.defined());
  for (int64_t i = 0; i < ndim; ++i) {
    auto shard_dim_size = spec->logic_shape[i]->value;
    auto dim_size = Downcast<IntImm>(dshape[i])->value;
    CHECK_EQ(dim_size % shard_dim_size, 0) << "Currently automaic padding is unsupported.";
    oshape[i] = Integer(dim_size / shard_dim_size);
  }
  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op._reshard_r2s", "Reshard_R2S", Reshard_R2S_Infer);

RAF_REGISTER_GLOBAL("raf.sharding._make.ReplicatedSpec").set_body_typed(ReplicatedSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.TupleShardSpec").set_body_typed(TupleShardSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.ShardOpCallAttrs").set_body_typed(ShardOpCallAttrs::make);

RAF_REGISTER_OBJECT_NO_REFLECT(BaseShardSpecObj);
RAF_REGISTER_OBJECT_REFLECT(ReplicatedSpecObj);
RAF_REGISTER_OBJECT_REFLECT(ShardSpecObj);
RAF_REGISTER_OBJECT_REFLECT(TupleShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;

void PrintAllocTable(const ObjectRef& ref, ReprPrinter* p) {
  /*size_t dev_idx = 0;
  const auto obj = Downcast<ShardSpec>(ref);
  const auto num_dim = obj->phy_shape.size();
  static thread_local size_t *indices = new size_t[num_dim];
  std::function<void(int)> _print_alloc_table;
  _print_alloc_table = [&](int depth) {
    if (depth == num_dim) {
      p->stream << (dev_idx != 0 ? " [" : "[");
      for (size_t i = 0; i < num_dim; ++i) {
        auto num_devices = obj->phy_shape[i]->value;
        auto index = std::to_string(indices[i]);
        p->stream << (num_devices == 1 ? ":" : index)
                  << (i != num_dim - 1 ? ", " : "");
      }
      auto dev_info = obj->ranks[dev_idx++].c_str();
      p->stream << "]@" << dev_info;
    } else {
      auto subgroup_num = obj->phy_shape[depth]->value;
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
      p->stream << "ReplicatedSpec" << (r->immutable ? "(Immut)" : "");
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ShardSpec>(ref);
      auto ndim = r->logic_shape.size();
      p->stream << "ShardSpec(" << (r->immutable ? "Immut " : "") << "[";
      for (size_t i = 0; i < ndim; ++i) {
        auto shard_dim_size = r->logic_shape[i]->value;
        auto subgroup_size = r->replicas[i]->value;
        p->stream << (shard_dim_size == 1 ? ":" : std::to_string(shard_dim_size))
                  << (subgroup_size == 1 ? "" : "(x" + std::to_string(subgroup_size) + ")")
                  << (i != ndim - 1 ? ", " : "");
      }
      p->stream << "])";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<TupleShardSpec>(ref);
      p->stream << "TupleShardSpec" << (r->immutable ? "(Immut)" : "");
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardOpCallAttrs>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* n = static_cast<const ShardOpCallAttrs*>(ref.get());
      p->stream << "ShardOpCallAttrs("
                << "in=" << n->shard_in << " out=" << n->shard_out << ")";
    });

TVM_REGISTER_NODE_TYPE(ShardOpCallAttrs);

}  // namespace sharding
}  // namespace raf

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
using namespace raf::sharding;

std::vector<Value> ReshardSchema2Args(const ShardUnaryArgs* args) {
  return {args->x};
}

std::vector<std::string> ReshardSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs ReshardSchema2Attrs(const ShardUnaryArgs* args) {
  auto attrs = make_object<StridedSliceAttrs>();
  auto spec = Downcast<ShardSpec>(args->spec);
  const DLTensor* x = args->x;
  std::vector<Integer> begin(x->ndim);
  std::vector<Integer> end(x->ndim);
  CHECK(spec->logic_index.defined());
  for (int i = 0; i < x->ndim; ++i) {
    auto idx = spec->logic_index[i]->value;
    auto size = spec->logic_shape[i]->value;
    begin[i] = Integer((x->shape[i] / size) * idx);
    end[i] = Integer((x->shape[i] / size) * (idx + 1));
  }
  attrs->begin = Array<Integer>(begin);
  attrs->end = Array<Integer>(end);
  return Attrs(attrs);
}

HashKey ReshardHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ShardUnaryArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  auto spec = Downcast<ShardSpec>(args->spec);
  for (auto array : {spec->ranks, spec->phy_shape, spec->replicas}) {
    for (auto i : array) {
      key << i->value;
    }
  }

  return key;
}

RAF_TVM(_reshard_r2s, Reshard_R2S, ShardUnaryArgs, ReshardSchema2Args, ReshardSchemaArgNames,
        ReshardSchema2Attrs, ReshardHasher, kInjective);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf