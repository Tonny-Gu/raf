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
namespace sharding {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;
using namespace mnm::sharding;

class ShardAttrsInstaller : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* node) override {
    const Expr& callee = node->op;
    static auto default_spec = ReplicatedSpec::make(false);
    static auto default_attrs = make_object<ShardOpAttrs>();
    default_attrs->shard_in = default_spec;
    default_attrs->shard_out = default_spec;
    if (callee->IsInstance<OpNode>()) {
      return Call(node->op, node->args, Attrs(default_attrs));
    }
  }
};

}  // namespace sharding

Pass InitShardOpAttrs() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::InitShardOpAttrs";
        ir::IRModule updated_mod = ir::IRModule(mod->functions);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<ir::FunctionNode>()) {
            auto func =
                tvm::runtime::Downcast<ir::Function>(sharding::ShardAttrsInstaller().VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "InitShardOpAttrs", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.InitShardOpAttrs").set_body_typed(InitShardOpAttrs);

}  // namespace pass
}  // namespace mnm
