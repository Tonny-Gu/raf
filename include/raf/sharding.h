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

/* BaseSpecValue */
class BaseSpecValueObj : public ValueObj {
 public:
  bool immutable;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
  }
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.sharding.BaseSpecValue";
  RAF_BASE_OBJECT(BaseSpecValueObj, ValueObj);
};

class BaseSpecValue : public Value {
 public:
  RAF_OBJECT_REF(BaseSpecValue, Value, BaseSpecValueObj);
};

/* ReplicatedSpecValue */
class ReplicatedSpecValueObj final : public BaseSpecValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
  }
  static constexpr const char* _type_key = "raf.sharding.ReplicatedSpecValue";
  RAF_FINAL_OBJECT(ReplicatedSpecValueObj, BaseSpecValueObj);
};

class ReplicatedSpecValue final : public BaseSpecValue {
 public:
  static ReplicatedSpecValue make(bool immutable);
  RAF_OBJECT_REF(ReplicatedSpecValue, BaseSpecValue, ReplicatedSpecValueObj);
};

/* ShardSpecValue */
class ShardSpecValueObj final : public BaseSpecValueObj {
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

  static constexpr const char* _type_key = "raf.sharding.ShardSpecValue";
  RAF_FINAL_OBJECT(ShardSpecValueObj, BaseSpecValueObj);
};

class ShardSpecValue final : public BaseSpecValue {
 public:
  static ShardSpecValue make(bool immutable, Array<Integer> ranks, Array<Integer> partition_shape,
                        Array<Integer> replicas);
  RAF_OBJECT_REF(ShardSpecValue, BaseSpecValue, ShardSpecValueObj);
};

/* TupleSpecValue */
class TupleSpecValueObj final : public BaseSpecValueObj {
 public:
  Array<BaseSpecValue> tuple_elem;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("tuple_elem", &tuple_elem);
  }
  static constexpr const char* _type_key = "raf.sharding.TupleSpecValue";
  RAF_FINAL_OBJECT(TupleSpecValueObj, BaseSpecValueObj);
};

class TupleSpecValue final : public BaseSpecValue {
 public:
  static TupleSpecValue make(bool immutable, Array<BaseSpecValue> tuple_elem);
  RAF_OBJECT_REF(TupleSpecValue, BaseSpecValue, TupleSpecValueObj);
};

struct ShardOpCallAttrs : public tvm::AttrsNode<ShardOpCallAttrs> {
  static Attrs make(BaseSpecValue shard_in, BaseSpecValue shard_out);
  BaseSpecValue shard_in, shard_out;
  TVM_DECLARE_ATTRS(ShardOpCallAttrs, "raf.attrs.ShardOpCallAttrs") {
    TVM_ATTR_FIELD(shard_in)
        .set_default(NullValue<BaseSpecValue>())
        .describe("Sharding Specifications of inputs");
    TVM_ATTR_FIELD(shard_out)
        .set_default(NullValue<BaseSpecValue>())
        .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace raf
