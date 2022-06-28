/*!
 * Copyright (c) 2021 by Contributors
 * \file sharding.h
 * \brief RAF Sharding System
 */
#pragma once
#include "./value.h"
#include <sstream>

namespace raf {
namespace sharding {

using namespace raf::ir;
using namespace raf::value;

class BaseShardSpecObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.sharding.BaseShardSpec";
  RAF_BASE_OBJECT(BaseShardSpecObj, ValueObj);
};

class BaseShardSpec : public Value {
 public:
  RAF_OBJECT_REF(BaseShardSpec, Value, BaseShardSpecObj);
};

class ShardSpecObj final : public BaseShardSpecObj {
 public:
  bool mutable_;
  int64_t ndim_;
  int64_t nshard_;
  int64_t ngroup_;
  Array<Integer> ranks;
  Array<Integer> logic_shape;
  Array<Integer> logic_index_;
  Array<Integer> phy_shape;
  Array<Integer> phy_index_;
  Array<Integer> subgroup_shape;
  Array<Integer> subgroup_index_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mutable", &mutable_);
    v->Visit("ndim", &ndim_);
    v->Visit("nshard", &nshard_);
    v->Visit("ngroup", &ngroup_);
    v->Visit("ranks", &ranks);
    v->Visit("logic_shape", &logic_shape);
    v->Visit("logic_index", &logic_index_);
    v->Visit("phy_shape", &logic_shape);
    v->Visit("phy_index", &logic_index_);
    v->Visit("subgroup_shape", &subgroup_shape);
    v->Visit("subgroup_index", &subgroup_index_);
  }

  static constexpr const char* _type_key = "raf.sharding.ShardSpec";
  RAF_FINAL_OBJECT(ShardSpecObj, BaseShardSpecObj);
};

class ShardSpec final : public BaseShardSpec {
 public:
  static ShardSpec make(Array<Integer> ranks, Array<Integer> phy_shape, Array<Integer> subgroup_shape, bool mutable_);
  static int64_t GetRankIdx(Array<Integer> ranks);
  RAF_OBJECT_REF(ShardSpec, BaseShardSpec, ShardSpecObj);
};

class UnsetShardSpecObj final : public BaseShardSpecObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "raf.sharding.UnsetShardSpec";
  RAF_FINAL_OBJECT(UnsetShardSpecObj, BaseShardSpecObj);
};

class UnsetShardSpec final : public BaseShardSpec {
 public:
  static UnsetShardSpec make() {
    auto n = make_object<UnsetShardSpecObj>();
    return UnsetShardSpec(n);
  };
  RAF_OBJECT_REF(UnsetShardSpec, BaseShardSpec, BaseShardSpecObj);
};

struct ShardOpCallAttrs : public tvm::AttrsNode<ShardOpCallAttrs> {
  static Attrs make(Array<BaseShardSpec> in, Array<BaseShardSpec> out);
  Array<BaseShardSpec> in, out;
  TVM_DECLARE_ATTRS(ShardOpCallAttrs, "raf.attrs.ShardOpCallAttrs") {
    TVM_ATTR_FIELD(in)
        .set_default(NullValue<Array<BaseShardSpec>>())
        .describe("Sharding Specifications of inputs");
    TVM_ATTR_FIELD(out)
        .set_default(NullValue<Array<BaseShardSpec>>())
        .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace raf
