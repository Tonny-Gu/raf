/*!
 * Copyright (c) 2021 by Contributors
 * \file sharding.h
 * \brief MNM Sharding System
 */
#pragma once
#include "./value.h"

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::value;

/* Sharding Specifications */
class ShardSpecObj : public Object {
 public:
  bool immutable;
  bool replicated;
  // TODO: replace with new Device object. PR: https://github.com/meta-project/meta/pull/725
  Array<Integer> assigned_devices;
  Array<Integer> num_devices_on_dim;
  Array<Integer> num_replicas_on_dim;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("replicated", &replicated);
    v->Visit("assigned_devices", &assigned_devices);
    v->Visit("num_devices_on_dim", &num_devices_on_dim);
    v->Visit("num_replicas_on_dim", &num_replicas_on_dim);
  }
  static constexpr const uint32_t _type_index = tvm::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mnm.sharding.ShardSpec";
  MNM_FINAL_OBJECT(ShardSpecObj, Object);
};

class ShardSpec : public ObjectRef {
 public:
  static ShardSpec make(bool immutable, bool replicated,
                        Array<Integer> assigned_devices,
                        Array<Integer> num_devices_on_dim,
                        Array<Integer> num_replicas_on_dim);
  MNM_OBJECT_REF(ShardSpec, ObjectRef, ShardSpecObj);
};

struct ShardOpAttrs : public tvm::AttrsNode<ShardOpAttrs> {
  Array<ShardSpec> shard_out;
  TVM_DECLARE_ATTRS(ShardOpAttrs, "mnm.attrs.ShardOpAttrs") {
    TVM_ATTR_FIELD(shard_out).set_default(NullValue<Array<ShardSpec> >())
                             .describe("Sharding Specifications of outputs");
  }
};

}  // namespace sharding
}  // namespace mnm