/*!
 * Copyright (c) 2021 by Contributors
 * \file  init_shardspec.cc
 * \brief Gradient operator input selection pass
 */
#include <sstream>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/sharding.h"
#include <string>
#include <vector>

namespace raf {
namespace pass {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::sharding;

namespace shard_pass {

class ShardOpCallAttrsSetter : public ExprMutator {
 public:
  explicit ShardOpCallAttrsSetter(const Map<Expr, Attrs>& attrs_map) : _attrs_map(attrs_map) {
  }

  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    // static auto default_spec = ReplicatedSpec::make(false);
    // static auto default_attrs =
    //     ShardOpCallAttrs::make(BaseShardSpec(default_spec), BaseShardSpec(default_spec));
    if (callee->IsInstance<OpNode>()) {
      auto ref = GetRef<Expr>(node);
      if (_attrs_map.count(ref)) {
        auto new_expr = Call(node->op, node->args, Attrs(_attrs_map[ref]));
        return ExprMutator::VisitExpr_(new_expr.as<CallNode>());
      }
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
    const auto* f = tvm::runtime::Registry::Get("raf.sharding._match_expansion_pattern");
    if (attrs.defined() && op->IsInstance<OpNode>() && attrs->IsInstance<ShardOpCallAttrs>()) {
      auto call = GetRef<Call>(node);
      Expr new_expr = (*f)(call);
      return call.same_as(new_expr) ? new_expr : ExprMutator::VisitExpr(new_expr);
    }
    return ExprMutator::VisitExpr_(node);
  }
};

}  // namespace shard_pass

Pass SetShardOpCallAttrs(const Map<Expr, Attrs>& attrs_map) {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::SetShardOpCallAttrs";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpCallAttrsSetter(attrs_map);
            auto func = tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "SetShardOpCallAttrs", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.SetShardOpCallAttrs").set_body_typed(SetShardOpCallAttrs);

Pass ExpandShardOpCall() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::ExpandShardOpCall";
        IRModule updated_mod = IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<FunctionNode>()) {
            auto setter = shard_pass::ShardOpCallExpander();
            auto func = tvm::runtime::Downcast<Function>(setter.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "ExpandShardOpCall", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.ExpandShardOpCall").set_body_typed(ExpandShardOpCall);

}  // namespace pass
}  // namespace raf
