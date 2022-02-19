/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file extract_binding.cc
 * \brief Extracting a relay body from frontend defined binding
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/binding.h"
#include "mnm/pass.h"

namespace mnm {
namespace pass {
namespace extract_binding {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::binding;

class Extractor final : public ExprVisitor {
 public:
  explicit Extractor(const Array<Var>& ignores) {
    for (const Var& var : ignores) {
      this->ignore.insert(var.get());
    }
  }

  void VisitExpr_(const VarNode* var) final {
    LOG(FATAL) << "Should not be here";
  }

  void VisitExpr_(const CallNode* node) final {
    EnqueueVar(node->op);
    for (const Expr& expr : node->args) {
      EnqueueVar(expr);
    }
  }

  void VisitExpr_(const TupleNode* node) final {
    for (const Expr& expr : node->fields) {
      EnqueueVar(expr);
    }
  }

  void VisitExpr_(const TupleGetItemNode* node) final {
    EnqueueVar(node->tuple);
  }

  void VisitExpr_(const IfNode* node) final {
    EnqueueVar(node->cond);
    EnqueueVar(node->true_branch);
    EnqueueVar(node->false_branch);
  }

  void VisitExpr_(const FunctionNode* node) final {
    for (const Var& var : FreeVars(GetRef<Function>(node))) {
      EnqueueVar(var);
    }
  }

  void EnqueueVar(const Expr& expr) {
    if (expr->IsInstance<ConstantNode>() || expr->IsInstance<OpNode>()) {
      return;
    }
    if (phase == 0) {
      if (const VarNode* var = expr.as<VarNode>()) {
        if (++in_degree[var] == 1) {
          queue.push_back(var);
        }
      } else {
        LOG(FATAL) << "Every intermediate result should be bound to a relay.Var";
      }
      return;
    }
    if (phase == 1) {
      if (const VarNode* var = expr.as<VarNode>()) {
        if (--in_degree[var] == 0) {
          queue.push_back(var);
        }
      } else {
        LOG(FATAL) << "Every intermediate result should be bound to a relay.Var";
      }
      return;
    }
    LOG(FATAL) << "Shouldn't be here";
    throw;
  }

  std::vector<const VarNode*> queue;
  std::unordered_map<const VarNode*, int> in_degree;
  std::unordered_map<const VarNode*, const ExprNode*> bindings;
  std::unordered_set<const VarNode*> ignore;
  int phase{};

  Expr Run(const Var& var) {
    // Calculate the in_degree of each var
    // Basically in_degree means how many times the var is used in other exprs
    phase = 0;
    EnqueueVar(var);
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      queue.pop_back();
      if (ignore.find(var) != ignore.end()) {
        continue;
      }
      const auto& binding = LookupBinding(var);
      CHECK(binding.defined()) << "Unbinded variable " << GetRef<Var>(var);
      if (const auto* sym = binding.as<SymbolBindingObj>()) {
        const Expr& expr = sym->expr;
        bindings[var] = expr.operator->();
        if (expr.defined()) {
          ExprVisitor::VisitExpr(expr);
        }
      } else if (binding->IsInstance<NDArrayBindingObj>()) {
        bindings[var] = {};
        continue;
      }
    }
    // Do topo-sort based on in_degree
    // If the in_degree of a var decreases to 0,
    // it means the var can be bound without being malformed
    phase = 1;
    Expr body = var;
    queue.clear();
    this->visit_counter_.clear();
    EnqueueVar(var);
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      const ExprNode* expr_node = bindings[var];
      queue.pop_back();
      if (expr_node == nullptr) {
        continue;
      }
      if (!expr_node->IsInstance<ConstantNode>()) {
        body = Let(GetRef<Var>(var), GetRef<Expr>(expr_node), body);
      }
      ExprVisitor::VisitExpr(GetRef<Expr>(expr_node));
    }
    return body;
  }
};

Expr ExtractBinding(const Var& var, const Array<Var>& ignore) {
  return Extractor(ignore).Run(var);
}

MNM_REGISTER_GLOBAL("mnm.pass_.ExtractBinding").set_body_typed(ExtractBinding);
}  // namespace extract_binding
}  // namespace pass
}  // namespace mnm
