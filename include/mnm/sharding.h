/*!
 * Copyright (c) 2021 by Contributors
 * \file sharding.h
 * \brief MNM Sharding System
 */
#pragma once
#include "./value.h"
#include <sstream>

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::value;

/* BaseShardSpec */
class BaseShardSpecObj : public Object {
 public:
  bool immutable;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
  }
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mnm.sharding.BaseShardSpec";
  MNM_BASE_OBJECT(BaseShardSpecObj, Object);
};

class BaseShardSpec : public ObjectRef {
 public:
  MNM_OBJECT_REF(BaseShardSpec, ObjectRef, BaseShardSpecObj);
};

/* ReplicatedSpec */
class ReplicatedSpecObj final : public BaseShardSpecObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
  }
  static constexpr const char* _type_key = "mnm.sharding.ReplicatedSpec";
  MNM_FINAL_OBJECT(ReplicatedSpecObj, BaseShardSpecObj);
};

class ReplicatedSpec final : public BaseShardSpec {
 public:
  static ReplicatedSpec make(bool immutable);
  MNM_OBJECT_REF(ReplicatedSpec, BaseShardSpec, ReplicatedSpecObj);
};

/* ShardSpec */
class ShardSpecObj final : public BaseShardSpecObj {
 public:
  Array<Device> devices_in_grid;
  Array<Integer> grid_shape;
  Array<Integer> subgroup_sizes;
  Array<Integer> subgroup_idx;
  
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("devices_in_grid", &devices_in_grid);
    v->Visit("grid_shape", &grid_shape);
    v->Visit("subgroup_sizes", &subgroup_sizes);
    v->Visit("subgroup_idx", &subgroup_idx);
  }

  static constexpr const char* _type_key = "mnm.sharding.ShardSpec";
  MNM_FINAL_OBJECT(ShardSpecObj, BaseShardSpecObj);
};

class ShardSpec final : public BaseShardSpec {
 public:
  static ShardSpec make(bool immutable,
                        Array<Device> devices_in_grid,
                        Array<Integer> partition_shape,
                        Array<Integer> subgroup_sizes);
  MNM_OBJECT_REF(ShardSpec, BaseShardSpec, ShardSpecObj);
};

/* TupleShardSpec */
class TupleShardSpecObj final : public BaseShardSpecObj {
 public:
  Array<BaseShardSpec> tuple_elem;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("tuple_elem", &tuple_elem);
  }
  static constexpr const char* _type_key = "mnm.sharding.TupleShardSpec";
  MNM_FINAL_OBJECT(TupleShardSpecObj, BaseShardSpecObj);
};

class TupleShardSpec final : public BaseShardSpec {
 public:
  static TupleShardSpec make(bool immutable,
                              Array<BaseShardSpec> tuple_elem);
  MNM_OBJECT_REF(TupleShardSpec, BaseShardSpec, TupleShardSpecObj);
};

struct ShardOpAttrs : public tvm::AttrsNode<ShardOpAttrs> {
  static Attrs make(BaseShardSpec shard_in, BaseShardSpec shard_out);
  BaseShardSpec shard_in, shard_out;
  TVM_DECLARE_ATTRS(ShardOpAttrs, "mnm.attrs.ShardOpAttrs") {
    TVM_ATTR_FIELD(shard_in).set_default(NullValue<BaseShardSpec>())
                             .describe("Sharding Specifications of inputs");
    TVM_ATTR_FIELD(shard_out).set_default(NullValue<BaseShardSpec>())
                             .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace mnm
