/*!
 * Copyright (c) 2021 by Contributors
 * \file to_dataflow.cc
 * \brief Convert A-normal form to dataflow graph.
 */
#include <unordered_map>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "mnm/pass.h"
#include "support/arena.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/op_attr_types.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace to_dataflow_graph {

using namespace mnm::ir;
using namespace mnm::op;
using namespace tvm::support;

class DFGConverter : public MixedModeMutator {
 public:
  Expr VisitExpr_(const LetNode* ln) final {
    Expr body = GetRef<Let>(ln);
    std::vector<std::pair<Var, Expr>> scopes;
    // Iteratively visit let nodes to avoid stack overflow.
    while (body->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(body);
      auto new_value = VisitExpr(let->value);
      if (new_value->IsInstance<RefCreateNode>() || new_value->IsInstance<RefReadNode>() ||
          new_value->IsInstance<RefWriteNode>()) {
        // Keep the Let for ref-related nodes as the order affects the correctness
        // auto new_body = VisitExpr(let->body);
        scopes.emplace_back(let->var, new_value);
        body = let->body;
      } else {
        let_map_.emplace(let->var.get(), new_value);
        scopes.emplace_back(Var(), Expr());
        body = let->body;
      }
    }
    Expr new_body = VisitExpr(body);
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
      if (it->first.defined()) {
        new_body = Let(it->first, it->second, new_body);
      }
    }
    return new_body;
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = let_map_.find(var);
    if (it != let_map_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

 private:
  std::unordered_map<const VarNode*, Expr> let_map_;
};
}  // namespace to_dataflow_graph

ir::Expr ToDataflowGraph(ir::Expr expr) {
  return to_dataflow_graph::DFGConverter().Mutate(expr);
}

// TODO - Cleanup when pass manager is introduced.
ir::Module ToDataflowGraph(ir::Module mod) {
  ir::Module updated_mod = ir::Module::make(mod->functions);
  std::vector<std::pair<ir::GlobalVar, ir::Function>> updated_funcs;
  auto converter = to_dataflow_graph::DFGConverter();

  for (auto kv : updated_mod->functions) {
    if (kv.second.as<ir::FunctionNode>()) {
      auto expr = converter.Mutate(kv.second);
      auto func = tvm::runtime::Downcast<ir::Function>(expr);
      updated_funcs.emplace_back(kv.first, func);
    }
  }

  for (const auto& it : updated_funcs) {
    updated_mod->Add(it.first, it.second, true);
  }
  return updated_mod;
}

MNM_REGISTER_GLOBAL("mnm.pass_.ToDataflowGraph").set_body_typed([](ir::Module mod) {
  return ToDataflowGraph(mod);
});

}  // namespace pass
}  // namespace mnm