/*!
 * Copyright (c) 2021 by Contributors
 * \file src/pass/inline_closure.cc
 * \brief Inline closure invoke
 */
#include <unordered_map>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/binding.h"
#include "./common.h"
#include "./let_list.h"

namespace mnm {
namespace pass {
namespace inline_closure {

/*! \brief This pass inlines non-recursive closure invokes.
 * An IRModule with mutiple functions can be simplified into a single function by this pass.
 * Thus runtime that does not support closure can benefit from it.
 *
 * For example, before inlining, the IRModule contains 3 functions, which are @main, @fwd and
 * @lifted_name542...:
 *
 * def @main(%x, %x1, %dy) {
 *   let %v = @fwd;
 *   let %v1 = %v(%x, %x1);
 *   let %v2 = %v1.0;
 *   let %v3 = %v1.1;
 *   let %v4 = %v3(%dy);
 *   let %v5 = %v4.0;
 *   let %v6 = %v4.1;
 *   let %v7 = (%v5, %v6);
 *   let %v8 = (%v2, %v7);
 *   %v8
 * }
 *
 * def @fwd(%x2, %y) {
 *   let %a1 = mnm.op.add(%x2, %y, -114514, -114514)
 *   let %adjoint_closure = @lifted_name5429879841773454120(%x2, %y);
 *   let %ret = (%a1, %adjoint_closure);
 *   %ret
 * }
 *
 * def @lifted_name5429879841773454120(%x3, %y1, Closure=1) {
 *   fn (%dy1) {
 *     let %x_2 = mnm.op.sum(%dy1, -114514, -114514, -114514);
 *     let %x_5 = mnm.op.sum(%dy1, -114514, -114514, -114514);
 *     let %x_6 = (%x_2, %x_5);
 *     %x_6
 *   }
 * }
 *
 * After inlining, it is simplified into a single function:
 *
 * def @main(%x, %x1, %dy) {
 *   let %x_1 = mnm.op.add(%x, %x1, -114514, -114514);
 *   let %x_4 = mnm.op.sum(%dy, -114514, -114514, -114514);
 *   let %x_5 = mnm.op.sum(%dy, -114514, -114514, -114514);
 *   let %x_7 = (%x_4, %x_5);
 *   let %x_8 = (%x_1, %x_7);
 *   %x_8
 * }
 *
 * Note: this pass assumes LambdaLift has been run beforehand.
 */

using namespace mnm::ir;
using namespace mnm::op;

class ClosureInliner : public MixedModeMutator {
 public:
  ClosureInliner(IRModule mod) : mod_(mod) {
    for (const auto& kv : mod->functions) {
      if (kv.second.as<FunctionNode>()) {
        func_map_[kv.first] = Downcast<Function>(kv.second);
      }
    }
  }

  Expr Rewrite_(const TupleGetItemNode* pre, const Expr& post) override {
    auto it = tuple_map_.find(Downcast<TupleGetItem>(post)->tuple);
    if (it != tuple_map_.end()) {
      Tuple tuple = it->second;
      return VisitExpr(tuple->fields[pre->index]);
    }
    return post;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    auto call = Downcast<Call>(post);
    Array<Expr> updated_args = Downcast<Call>(post)->args;
    if (pre->checked_type().as<FuncTypeNode>()) {
      // Partial function application
      Function func = Downcast<Function>(func_map_.at(call->op));
      Map<Var, Expr> args_map;
      CHECK_EQ(pre->args.size(), func->params.size());
      for (size_t i = 0; i < pre->args.size(); ++i) {
        args_map.Set(func->params[i], updated_args[i]);
      }
      func_map_[let_var_] = Downcast<Function>(InferType(Substitute(func->body, args_map)));
    } else if (!pre->op.as<OpNode>()) {
      // Function application
      Expr func_var = call->op;
      Function func = func_map_.at(func_var);
      return Inline(func, updated_args);
    }
    return post;
  }

  Expr Inline(const Function& func, const Array<Expr>& args) {
    std::unique_ptr<ExplicitLetList> ell = ExplicitLetList::make(func->body);
    const std::vector<Var>& vars = ell->vars;
    const std::vector<Expr>& exprs = ell->exprs;
    size_t n = vars.size();
    Var tmp = let_var_;
    CHECK_EQ(vars.size(), exprs.size());
    CHECK_EQ(func->params.size(), args.size());
    for (size_t i = 0; i < func->params.size(); ++i) {
      memo_[func->params[i]] = args[i];
    }
    for (size_t i = 0; i < n; ++i) {
      let_var_ = vars[i];
      Expr expr = VisitExpr(exprs[i]);
      if (expr.as<ExtendedVarNode>()) {
        memo_[let_var_] = Downcast<Var>(expr);
      } else if (expr.defined()) {
        Var v = ll_->Push(expr);
        if (expr.as<TupleNode>()) {
          tuple_map_[v] = Downcast<Tuple>(expr);
        } else if (expr.as<GlobalVarNode>()) {
          func_map_[v] = func_map_[expr];
        }
        memo_[let_var_] = v;
      }
      if (memo_.find(let_var_) != memo_.end() && func_map_.find(let_var_) != func_map_.end()) {
        func_map_[memo_[let_var_]] = func_map_[let_var_];
      }
    }
    Expr ret = VisitExpr(ell->ret);
    let_var_ = tmp;
    return ret;
  }

  Expr operator()(const Expr& e) {
    auto func = Downcast<Function>(e);
    Function result;
    if (func->body.as<FunctionNode>()) {
      // special handling for Closure, where its body is still a Function
      // which is not standard ANF form
      result =
          Function(func->params, this->operator()(func->body), func->ret_type, func->type_params);
    } else {
      Expr inlined_body = LetList::With([&](LetList* ll) {
        ll_ = ll;
        Array<Expr> args;
        for (const auto& v : func->params) {
          args.push_back(v);
        }
        return Inline(func, args);
      });
      result = Downcast<Function>(DeadCodeElimination(
          Function(func->params, inlined_body, func->ret_type, func->type_params)));
    }
    return result;
  }

 private:
  /*! \brief The IRModule */
  IRModule mod_;
  /*! \brief Mapping of a var to a tuple. */
  std::unordered_map<Expr, Tuple, ObjectPtrHash, ObjectPtrEqual> tuple_map_;
  /*! \brief Mapping from a var to a global var. */
  std::unordered_map<Expr, Function, ObjectPtrHash, ObjectPtrEqual> func_map_;
  /*! \brief The current let variable */
  Var let_var_;
  /*! \breif The result let list */
  LetList* ll_;
};

}  // namespace inline_closure

Pass InlineClosure() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    return Downcast<Function>(inline_closure::ClosureInliner(m)(f));
  };
  return CreateMNMFunctionPass(pass_func, 1, "InlineClosure", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.InlineClosure").set_body_typed(InlineClosure);

}  // namespace pass
}  // namespace mnm