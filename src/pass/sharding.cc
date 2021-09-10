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
      auto new_expr = Call(node->op, node->args, Attrs(_attrs_map.count(ref) ? _attrs_map[ref] : default_attrs));
      return ExprMutator::VisitExpr_(new_expr.as<CallNode>());
    }
    return ExprMutator::VisitExpr_(node);
  }
 private:
  const Map<Expr, Attrs>& _attrs_map;
};

class ShardOpCallExpander : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& op = node->op;
    const Attrs& attrs = node->attrs;
    const auto *f = tvm::runtime::Registry::Get("mnm.sharding._match_expansion_pattern");
    if (attrs.defined() && op->IsInstance<OpNode>() && attrs->IsInstance<ShardOpAttrs>()) {
      auto call = GetRef<Call>(node);
      Expr new_expr = (*f)(call->op, call->args, call->attrs);
      return ExprMutator::VisitExpr(new_expr); // nested conversion
    }
    return ExprMutator::VisitExpr_(node);
  }
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

Pass ExpandShardOpCall() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::ExpandShardOpCall";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpCallExpander();
            auto func =
                tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "ExpandShardOpCall", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.ExpandShardOpCall").set_body_typed(ExpandShardOpCall);

}  // namespace pass
}  // namespace mnm
