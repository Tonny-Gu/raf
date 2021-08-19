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

/* Sharding Specifications */
class ShardSpecObj : public Object {
 public:
  bool immutable;
  bool replicated;
  
  Array<Device> assigned_devices;
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
                        Array<Device> assigned_devices,
                        Array<Integer> num_devices_on_dim,
                        Array<Integer> num_replicas_on_dim);
  
  const void print_alloc_table(std::ostream& ostream = std::cout) const {
    size_t dev_idx = 0;
    const auto obj = this->operator->();
    const auto num_dim = obj->num_devices_on_dim.size();
    static thread_local size_t *indices = new size_t[num_dim];
    std::function<void(int)> _print_alloc_table;
    _print_alloc_table = [&](int depth) {
      if (depth == num_dim) {
        ostream << "[";
        for (size_t i = 0; i < num_dim; ++i) {
          auto num_devices = obj->num_devices_on_dim[i]->value;
          auto num_replicas = obj->num_replicas_on_dim.defined() ?
                              obj->num_replicas_on_dim[i]->value : 1;
          if (num_devices == 1) {
            ostream << ":, ";
          } else {
            auto index = indices[i] / num_replicas;
            ostream << index << ", ";
          }
        }
        auto dev_info = obj->assigned_devices[dev_idx++].c_str();
        ostream << "\b\b]@" << dev_info << " ";
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

  const char* c_str() const {
    static thread_local char buf[2048];
    std::stringstream sstream;
    const auto obj = this->operator->();
    sstream.clear();
    sstream << (obj->immutable ? "Immutable " : "");
    if (obj->replicated) {
      sstream << "Replicated ";
    } else {
      print_alloc_table(sstream);
      sstream << "\b";
    }
    strncpy(buf, sstream.str().c_str(), 2048);
    return buf;
  }

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