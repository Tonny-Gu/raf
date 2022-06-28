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
#include "raf/communicator.h"
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
using namespace raf::distributed::communicator;

int64_t ShardSpec::GetRankIdx(Array<Integer> ranks) {
  for (int64_t i = 0; i < ranks.size(); ++i) {
    if (GetGlobalCommunicator()->rank == ranks[i]->value) {
      return i;
    }
  }
  return -1;
}

ShardSpec ShardSpec::make(Array<Integer> ranks, Array<Integer> phy_shape, Array<Integer> subgroup_shape, bool mutable_) {
  CHECK_EQ(phy_shape.size(), subgroup_shape.size());
  auto ndim = phy_shape.size();
  auto subgroup_index = std::vector<Integer>(ndim);
  auto phy_index = std::vector<Integer>(ndim);
  auto logic_index = std::vector<Integer>(ndim);
  auto logic_shape = std::vector<Integer>(ndim);
  auto rank_idx = ShardSpec::GetRankIdx(ranks);
  int64_t nshard = 1, ngroup = 1;
  
  auto t1 = rank_idx;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    phy_index[i] = t1 % phy_shape[i]->value;
    t1 /= phy_shape[i]->value;

    logic_shape[i] = phy_shape[i]->value / subgroup_shape[i]->value;
    logic_index[i] = phy_index[i]->value / subgroup_shape[i]->value;
    nshard *= logic_shape[i]->value;

    subgroup_index[i] = phy_index[i]->value % subgroup_shape[i]->value;
    ngroup *= subgroup_shape[i]->value;
  }

  auto spec = make_object<ShardSpecObj>();
  spec->mutable_ = mutable_;
  spec->ndim_ = ndim;
  spec->nshard_ = nshard;
  spec->ngroup_ = ngroup;
  spec->ranks = std::move(ranks);
  spec->subgroup_shape = std::move(subgroup_shape);
  spec->phy_shape = std::move(phy_shape);
  spec->logic_shape = Array<Integer>(logic_shape);
  if (rank_idx == -1) {
    spec->subgroup_index_ = NullValue<Array<Integer>>();
    spec->phy_index_ = NullValue<Array<Integer>>();
    spec->logic_index_ = NullValue<Array<Integer>>();
  } else {
    spec->subgroup_index_ = Array<Integer>(subgroup_index);
    spec->phy_index_ = Array<Integer>(phy_index);
    spec->logic_index_ = Array<Integer>(logic_index);
  }

  return ShardSpec(spec);
}

Attrs ShardOpCallAttrs::make(Array<BaseShardSpec> in, Array<BaseShardSpec> out) {
  auto attrs = make_object<ShardOpCallAttrs>();
  attrs->in = std::move(in);
  attrs->out = std::move(out);
  return Attrs(attrs);
}

void Reshard_R2S(const CallValues& call) {
  const auto* args = call->args.as<ShardUnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  auto spec = Downcast<ShardSpec>(args->spec);
  if (spec->logic_index_.defined()) {
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
  CHECK(spec->logic_index_.defined());
  for (int64_t i = 0; i < ndim; ++i) {
    auto shard_dim_size = spec->logic_shape[i]->value;
    auto dim_size = Downcast<IntImm>(dshape[i])->value;
    CHECK_EQ(dim_size % shard_dim_size, 0) << "Currently automaic padding is unsupported.";
    oshape[i] = Integer(dim_size / shard_dim_size);
  }
  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op._reshard_r2s", "Reshard_R2S", Reshard_R2S_Infer);

RAF_REGISTER_GLOBAL("raf.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.UnsetShardSpec").set_body_typed(UnsetShardSpec::make);
RAF_REGISTER_GLOBAL("raf.sharding._make.ShardOpCallAttrs").set_body_typed(ShardOpCallAttrs::make);

RAF_REGISTER_OBJECT_NO_REFLECT(BaseShardSpecObj);
RAF_REGISTER_OBJECT_REFLECT(ShardSpecObj);
RAF_REGISTER_OBJECT_REFLECT(UnsetShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;

std::string PrintAllocTable(const ObjectRef& ref) {
  size_t dev_idx = 0;
  const auto spec = Downcast<ShardSpec>(ref);
  const auto ndim = spec->ndim_;

  std::stringstream ss;

  auto subgroup_index = std::vector<Integer>(ndim);
  auto phy_index = std::vector<Integer>(ndim);
  auto logic_index = std::vector<Integer>(ndim);

  ss << "| Rank | Physical Index | Logic Index | Subgroup Index |" << std::endl;

  for (int64_t rank_idx = 0; rank_idx < spec->ranks.size(); ++rank_idx) {
    auto t1 = rank_idx;
    for (int64_t i = ndim - 1; i >= 0; --i) {
      phy_index[i] = t1 % spec->phy_shape[i]->value;
      t1 /= spec->phy_shape[i]->value;
      logic_index[i] = phy_index[i]->value / spec->subgroup_shape[i]->value;
      subgroup_index[i] = phy_index[i]->value % spec->subgroup_shape[i]->value;
    }
    ss << "| " << spec->ranks[rank_idx]->value << " | ";
    for (auto arr : {phy_index, logic_index, subgroup_index}) {
      ss << "(";
      for (auto e : arr) {
        ss << e << ", ";
      }
      ss.seekp(-2, std::ios_base::end);
      ss << ") | ";
    }
    ss << std::endl;
  }

  return ss.str();
}

RAF_REGISTER_GLOBAL("raf.sharding.PrintAllocTable").set_body_typed(PrintAllocTable);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<ShardSpec>(ref);
      auto ndim = r->ndim_;
      if (r->nshard_ == 1) {
        p->stream << "ShardSpec(Mirrored)";
      } else {
        p->stream << "ShardSpec(" << "[";
        for (size_t i = 0; i < ndim; ++i) {
          auto nshard_on_dim = r->logic_shape[i]->value;
          auto ngroup_on_dim = r->subgroup_shape[i]->value;
          p->stream << (nshard_on_dim == 1 ? ":" : std::to_string(nshard_on_dim))
                    << (ngroup_on_dim == 1 ? "" : "(x" + std::to_string(ngroup_on_dim) + ")")
                    << (i != ndim - 1 ? ", " : "");
        }
        p->stream << "])";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UnsetShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto r = Downcast<UnsetShardSpec>(ref);
      p->stream << "UnsetShardSpec()";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardOpCallAttrs>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* n = static_cast<const ShardOpCallAttrs*>(ref.get());
      p->stream << "ShardOpCallAttrs("
                << "in=" << n->in << ", out=" << n->out << ")";
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
  CHECK(spec->logic_index_.defined());
  for (int i = 0; i < x->ndim; ++i) {
    auto idx = spec->logic_index_[i]->value;
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
  for (auto array : {spec->ranks, spec->logic_shape}) {
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