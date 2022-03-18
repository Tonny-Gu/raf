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

/* BaseShardSpec */
class BaseShardSpecObj : public ValueObj {
 public:
  bool immutable;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
  }
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.sharding.BaseShardSpec";
  RAF_BASE_OBJECT(BaseShardSpecObj, ValueObj);
};

class BaseShardSpec : public Value {
 public:
  RAF_OBJECT_REF(BaseShardSpec, Value, BaseShardSpecObj);
};

/* ReplicatedSpec */
class ReplicatedSpecObj final : public BaseShardSpecObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
  }
  static constexpr const char* _type_key = "raf.sharding.ReplicatedSpec";
  RAF_FINAL_OBJECT(ReplicatedSpecObj, BaseShardSpecObj);
};

class ReplicatedSpec final : public BaseShardSpec {
 public:
  static ReplicatedSpec make(bool immutable);
  RAF_OBJECT_REF(ReplicatedSpec, BaseShardSpec, ReplicatedSpecObj);
};

/* ShardSpec */
class ShardSpecObj final : public BaseShardSpecObj {
 public:
  Array<Integer> ranks;
  Array<Integer> replicas;
  Array<Integer> logic_shape;
  Array<Integer> logic_index;
  Array<Integer> phy_shape;
  Array<Integer> phy_index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("ranks", &ranks);
    v->Visit("replicas", &replicas);
    v->Visit("logic_shape", &logic_shape);
    v->Visit("logic_index", &logic_index);
    v->Visit("phy_shape", &phy_shape);
    v->Visit("phy_index", &phy_index);
  }

  static constexpr const char* _type_key = "raf.sharding.ShardSpec";
  RAF_FINAL_OBJECT(ShardSpecObj, BaseShardSpecObj);
};

class ShardSpec final : public BaseShardSpec {
 public:
  static ShardSpec make(bool immutable, Array<Integer> ranks, Array<Integer> partition_shape,
                        Array<Integer> replicas);
  RAF_OBJECT_REF(ShardSpec, BaseShardSpec, ShardSpecObj);
};

/* TupleShardSpec */
class TupleShardSpecObj final : public BaseShardSpecObj {
 public:
  Array<BaseShardSpec> tuple_elem;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("tuple_elem", &tuple_elem);
  }
  static constexpr const char* _type_key = "raf.sharding.TupleShardSpec";
  RAF_FINAL_OBJECT(TupleShardSpecObj, BaseShardSpecObj);
};

class TupleShardSpec final : public BaseShardSpec {
 public:
  static TupleShardSpec make(bool immutable, Array<BaseShardSpec> tuple_elem);
  RAF_OBJECT_REF(TupleShardSpec, BaseShardSpec, TupleShardSpecObj);
};

struct ShardOpCallAttrs : public tvm::AttrsNode<ShardOpCallAttrs> {
  static Attrs make(BaseShardSpec shard_in, BaseShardSpec shard_out);
  BaseShardSpec shard_in, shard_out;
  TVM_DECLARE_ATTRS(ShardOpCallAttrs, "raf.attrs.ShardOpCallAttrs") {
    TVM_ATTR_FIELD(shard_in)
        .set_default(NullValue<BaseShardSpec>())
        .describe("Sharding Specifications of inputs");
    TVM_ATTR_FIELD(shard_out)
        .set_default(NullValue<BaseShardSpec>())
        .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace raf
