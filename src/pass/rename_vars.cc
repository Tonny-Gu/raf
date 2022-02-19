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

namespace mnm {
namespace pass {
namespace rename_vars {

using namespace mnm::ir;

struct RenameVarsMutator : public ExprMutator {
 public:
  explicit RenameVarsMutator(const Map<String, Var>& named_vars) {
    for (const auto& iter : named_vars) {
      const auto* var = iter.second.as<ExtendedVarNode>();
      var_map_.Set(iter.second,
                   mnm::ir::MakeVar(iter.first, iter.second->type_annotation, var->may_share));
    }
  }

  Expr VisitExpr_(const VarNode* node) final {
    return var_map_.at(GetRef<Var>(node));
  }

  Expr VisitExpr_(const LetNode* node) final {
    auto pre_visit = [this](const LetNode* node) {
      const Var& var = node->var;
      CHECK_EQ(var_map_.count(var), 0) << "IR is malformed: cannot bind var twice";
      const auto* vn = var.as<ExtendedVarNode>();
      Var may_share = vn->may_share;
      Var new_var =
          mnm::ir::MakeVar("a" + std::to_string(++num_bound_var_), var->type_annotation,
                           may_share.defined() ? Downcast<Var>(var_map_.at(may_share)) : may_share);
      var_map_.Set(var, new_var);
      this->Mutate(node->value);
    };
    auto post_visit = [this](const LetNode* node) {
      Var var = Downcast<Var>(node->var);
      Expr value = this->Mutate(node->value);
      Expr body = this->Mutate(node->body);

      auto expr = GetRef<Expr>(node);
      if (var.same_as(node->var) && value.same_as(node->value) && body.same_as(node->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(Downcast<Var>(var_map_[var]), value, body);
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

 private:
  /*! \brief The counter of bound variables. */
  int num_bound_var_ = 0;
  /*! \brief Map from original var to the renamed var. */
  Map<Var, Expr> var_map_;
};

Expr RenameVars(Expr expr, Map<String, Var> named_vars) {
  return RenameVarsMutator(named_vars).Mutate(expr);
}

MNM_REGISTER_GLOBAL("mnm.pass_.RenameVars").set_body_typed(RenameVars);
}  // namespace rename_vars
}  // namespace pass
}  // namespace mnm
