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

class BaseSpecObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.sharding.BaseSpec";
  RAF_BASE_OBJECT(BaseSpecObj, ValueObj);
};

class BaseSpec : public Value {
 public:
  RAF_OBJECT_REF(BaseSpec, Value, BaseSpecObj);
};

class BaseShardSpecObj : public BaseSpecObj {
 public:
  Array<Integer> ranks;
  Integer ndim;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ranks", &ranks);
    v->Visit("ndim", &ndim);
  }

  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.sharding.BaseShardSpec";
  RAF_BASE_OBJECT(BaseShardSpecObj, BaseSpecObj);
};

class BaseShardSpec : public BaseSpec {
 public:
  RAF_OBJECT_REF(BaseShardSpec, BaseSpec, BaseShardSpecObj);
};

class TupleSpecObj final : public BaseSpecObj {
 public:
  Array<BaseShardSpec> tuple_elem;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple_elem", &tuple_elem);
  }
  static constexpr const char* _type_key = "raf.sharding.TupleSpec";
  RAF_FINAL_OBJECT(TupleSpecObj, BaseSpecObj);
};

class TupleSpec final : public BaseSpec {
 public:
  static TupleSpec make(Array<BaseShardSpec> tuple_elem) {
    auto n = make_object<TupleSpecObj>();
    n->tuple_elem = tuple_elem;
    return TupleSpec(n);
  }
  RAF_OBJECT_REF(TupleSpec, BaseSpec, TupleSpecObj);
};

class MirroredSpecObj final : public BaseShardSpecObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ranks", &ranks);
    v->Visit("ndim", &ndim);
  }
  static constexpr const char* _type_key = "raf.sharding.MirroredSpec";
  RAF_FINAL_OBJECT(MirroredSpecObj, BaseShardSpecObj);
};

class MirroredSpec final : public BaseShardSpec {
 public:
  static MirroredSpec make(Array<Integer> ranks, Integer ndim) {
    auto n = make_object<MirroredSpecObj>();
    n->ranks = ranks;
    n->ndim = ndim;
    return MirroredSpec(n);
  }

  RAF_OBJECT_REF(MirroredSpec, BaseShardSpec, MirroredSpecObj);
};

class AnyShardSpecObj final : public BaseShardSpecObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ranks", &ranks);
    v->Visit("ndim", &ndim);
  }
  static constexpr const char* _type_key = "raf.sharding.AnyShardSpec";
  RAF_FINAL_OBJECT(AnyShardSpecObj, BaseShardSpecObj);
};

class AnyShardSpec final : public BaseShardSpec {
 public:
  static AnyShardSpec make(Array<Integer> ranks, Integer ndim) {
    auto n = make_object<AnyShardSpecObj>();
    n->ranks = ranks;
    n->ndim = ndim;
    return AnyShardSpec(n);
  }

  RAF_OBJECT_REF(AnyShardSpec, BaseShardSpec, AnyShardSpecObj);
};

class ShardSpecObj : public BaseShardSpecObj {
 public:
  Array<Integer> logic_shape;
  Array<Integer> logic_index_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ranks", &ranks);
    v->Visit("ndim", &ndim);
    v->Visit("logic_shape", &logic_shape);
    v->Visit("logic_index", &logic_index_);
  }

  static constexpr const char* _type_key = "raf.sharding.ShardSpec";
  RAF_FINAL_OBJECT(ShardSpecObj, BaseShardSpecObj);
};

class ShardSpec : public BaseShardSpec {
 public:
  static ShardSpec make(Array<Integer> ranks, Array<Integer> shape);
  static int64_t GetRankIdx(Array<Integer> ranks);
  RAF_OBJECT_REF(ShardSpec, BaseShardSpec, ShardSpecObj);
};

class ReplicaSpecObj final : public ShardSpecObj {
 public:
  Array<Integer> replicas;
  Array<Integer> phy_shape;
  Array<Integer> phy_index_;

 void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ranks", &ranks);
    v->Visit("ndim", &ndim);
    v->Visit("replicas", &replicas);
    v->Visit("logic_shape", &logic_shape);
    v->Visit("logic_index", &logic_index_);
    v->Visit("phy_shape", &phy_shape);
    v->Visit("phy_index", &phy_index_);
  }
};

class ReplicaSpec : public ShardSpec {
 public:
  static ReplicaSpec make(Array<Integer> ranks, Array<Integer> replicas, Array<Integer> phy_shape);
  RAF_OBJECT_REF(ReplicaSpec, ShardSpec, ReplicaSpecObj);
};

struct ShardOpCallAttrs : public tvm::AttrsNode<ShardOpCallAttrs> {
  static Attrs make(BaseSpec shard_in, BaseSpec shard_out);
  BaseSpec shard_in, shard_out;
  TVM_DECLARE_ATTRS(ShardOpCallAttrs, "raf.attrs.ShardOpCallAttrs") {
    TVM_ATTR_FIELD(shard_in)
        .set_default(NullValue<BaseSpec>())
        .describe("Sharding Specifications of inputs");
    TVM_ATTR_FIELD(shard_out)
        .set_default(NullValue<BaseSpec>())
        .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace raf
