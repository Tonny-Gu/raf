/*!
 * Copyright (c) 2021 by Contributors
 * \file  init_shardspec.cc
 * \brief Gradient operator input selection pass
 */
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/sharding.h"
#include <string>
#include <vector>

namespace mnm {
namespace pass {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;
using namespace mnm::sharding;

namespace shard_pass {

class ShardOpAttrsSetter : public ExprMutator {
 public:
  explicit ShardOpAttrsSetter(const Map<Expr, Attrs>& attrs_map) :
    _attrs_map(attrs_map) {}
  
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    static auto default_spec = ReplicatedSpec::make(false);
    static auto default_attrs = ShardOpAttrs::make(BaseShardSpec(default_spec),
                                                   BaseShardSpec(default_spec));
    if (callee->IsInstance<OpNode>()) {
      auto ref = GetRef<Expr>(node);
      if (_attrs_map.count(ref)) {
        auto attrs = _attrs_map[ref];
        return Call(node->op, node->args, Attrs(attrs));
      } else {
        return Call(node->op, node->args, Attrs(default_attrs));
      }
    }
    return ExprMutator::VisitExpr_(node);
  }
 private:
  const Map<Expr, Attrs>& _attrs_map;
};

class ShardOpExpander : public ExprMutator {
 public:
  explicit ShardOpExpander(const Map<Expr, Expr>& func_map) :
    _func_map(func_map) {}
  
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    if (callee->IsInstance<OpNode>()) {
      auto ref = GetRef<Expr>(node);
      if (_func_map.count(ref)) {
        return _func_map[ref];
      }
    }
    return ExprMutator::VisitExpr_(node);
  }
 private:
  const Map<Expr, Expr>& _func_map;
};

}  // namespace sharding

Pass SetShardOpAttrs(const Map<Expr, Attrs>& attrs_map) {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::SetShardOpAttrs";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpAttrsSetter(attrs_map);
            auto func =
                tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "SetShardOpAttrs", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.SetShardOpAttrs").set_body_typed(SetShardOpAttrs);

Pass ExpandShardOp(const Map<Expr, Expr>& func_map) {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::ExpandShardOp";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpExpander(func_map);
            auto func =
                tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "ExpandShardOp", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.ExpandShardOp").set_body_typed(ExpandShardOp);

}  // namespace pass
}  // namespace mnm
