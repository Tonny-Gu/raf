/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/sharding.cc
 * \brief MNM Sharding System underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/sharding.h"
#include <string>

namespace mnm {
namespace sharding {

using namespace mnm::ir;
using namespace mnm::value;

ShardSpec ShardSpec::make(bool immutable, bool replicated,
                          Array<Integer> assigned_devices,
                          Array<Integer> num_devices_on_dim,
                          Array<Integer> num_replicas_on_dim) {
  ObjectPtr<ShardSpecObj> n = make_object<ShardSpecObj>();
  n->immutable = immutable;
  n->replicated = replicated;
  n->assigned_devices = std::move(assigned_devices);
  n->num_devices_on_dim = std::move(num_devices_on_dim);
  n->num_replicas_on_dim = std::move(num_replicas_on_dim);
  return ShardSpec(n);
}

MNM_REGISTER_GLOBAL("mnm.sharding._make.ShardSpec").set_body_typed(ShardSpec::make);
MNM_REGISTER_OBJECT_REFLECT(ShardSpecObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShardSpecObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* n = static_cast<const ShardSpecObj*>(ref.get());
      p->stream << "ShardSpec("
        << (n->immutable ? "Immutable " : "");
      if (n->replicated) {
        p->stream << "Replicated)";
      } else {
        auto num_dim = n->num_devices_on_dim.size();
        static thread_local size_t *indices = new size_t[num_dim];
        size_t dev_idx = 0;
        std::function<void(int)> report_alloc;
        report_alloc = [&](int dim) {
          if (dim == num_dim) {
            p->stream << "\n    [";
            for (size_t i = 0; i < num_dim; ++i) {
              auto num_devices = n->num_devices_on_dim[i]->value;
              auto num_replicas = n->num_replicas_on_dim.defined() ?
                                  n->num_replicas_on_dim[i]->value : 1;
              if (num_devices == 1) {
                p->stream << ":, ";
              } else {
                auto index = indices[i] / num_replicas;
                p->stream << index << ":" << index + 1 << ", ";
              }
            }
            auto dev_id = n->assigned_devices[dev_idx++]->value;
            p->stream << "\b\b]@cpu(" << dev_id << ")";
            return;
          }
          auto num_devices = n->num_devices_on_dim[dim]->value;
          for (size_t i = 0; i < num_devices; ++i) {
            indices[dim] = i;
            report_alloc(dim + 1);
          }
        };
        report_alloc(0);
        p->stream << ")";
      }
    });

TVM_REGISTER_NODE_TYPE(ShardOpAttrs);

}  // namespace sharding
}  // namespace mnm