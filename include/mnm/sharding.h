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
  Array<Device> assigned_devices;
  Array<Integer> num_devices_on_dim;
  Array<Integer> num_replicas_on_dim;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("immutable", &immutable);
    v->Visit("assigned_devices", &assigned_devices);
    v->Visit("num_devices_on_dim", &num_devices_on_dim);
    v->Visit("num_replicas_on_dim", &num_replicas_on_dim);
  }

  static constexpr const char* _type_key = "mnm.sharding.ShardSpec";
  MNM_FINAL_OBJECT(ShardSpecObj, BaseShardSpecObj);
};

class ShardSpec final : public BaseShardSpec {
 public:
  static ShardSpec make(bool immutable,
                        Array<Device> assigned_devices,
                        Array<Integer> num_devices_on_dim,
                        Array<Integer> num_replicas_on_dim);
  
  void printAllocTable(std::ostream& out = std::cout) const {
    size_t dev_idx = 0;
    const auto obj = this->operator->();
    const auto num_dim = obj->num_devices_on_dim.size();
    static thread_local size_t *indices = new size_t[num_dim];
    std::function<void(int)> _print_alloc_table;
    _print_alloc_table = [&](int depth) {
      if (depth == num_dim) {
        out << "[";
        for (size_t i = 0; i < num_dim; ++i) {
          auto num_devices = obj->num_devices_on_dim[i]->value;
          auto num_replicas = obj->num_replicas_on_dim[i]->value;
          if (num_devices == 1) {
            out << ":, ";
          } else {
            auto index = indices[i] / num_replicas;
            out << index << ", ";
          }
        }
        auto dev_info = obj->assigned_devices[dev_idx++].c_str();
        out << "\b\b]@" << dev_info << " ";
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
