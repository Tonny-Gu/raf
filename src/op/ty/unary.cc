/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/unary.cc
 * \brief Typing relations of unary operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using schema::UnaryArgs;
using schema::UnaryDxArgs;
using schema::UnaryUfuncArgs;

Type UnaryInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  // Unary ops' outputs are identical with inputs
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.log", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.log2", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.cos", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.sin", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.sign", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.round", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.relu", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.gelu", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.tanh", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.sigmoid", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.copy", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.abs", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.ceil", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.floor", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.exp", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.erf", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.sqrt", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.rsqrt", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.atan", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.trunc", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.ndarray_size", "Identity", UnaryInfer);

Type UnaryDxInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryDxArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.defined() || args->y.defined());
  // Unary ops' outputs are identical with inputs
  if (args->x.defined()) {
    return GetType(args->x.value());
  } else {
    return GetType(args->y.value());
  }
}

MNM_OP_TYPE("mnm.op.relu_dx", "IdentityDx", UnaryDxInfer);
MNM_OP_TYPE("mnm.op.gelu_dx", "IdentityDx", UnaryDxInfer);
MNM_OP_TYPE("mnm.op.tanh_dx", "IdentityDx", UnaryDxInfer);
MNM_OP_TYPE("mnm.op.sigmoid_dx", "IdentityDx", UnaryDxInfer);
MNM_OP_TYPE("mnm.op.erf_dx", "IdentityDx", UnaryDxInfer);
MNM_OP_TYPE("mnm.op.sqrt_dx", "IdentityDx", UnaryDxInfer);

Type UnaryUfuncInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  // UnaryUfunc ops' outputs are identical with inputs
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.negative", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.logical_not", "Identity", UnaryInfer);

Type UnaryShapeInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<tvm::PrimExpr> shape;
  shape.push_back(ir::Integer(x->shape.size()));
  return TensorType(shape, tvm::runtime::DataType::UInt(32));
}

MNM_OP_TYPE("mnm.op.shape", "Shape", UnaryShapeInfer);
MNM_OP_TYPE("mnm.op.zeros_like", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.ones_like", "Identity", UnaryInfer);

Type NumelInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  ICHECK(args != nullptr);
  ICHECK(args->x.defined());
  return TensorType({}, tvm::runtime::DataType::Int(32));
}

MNM_OP_TYPE("mnm.op.numel", "Numel", NumelInfer);

Type ShapeAsTensorInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<tvm::PrimExpr> shape;
  shape.push_back(ir::Integer(x->shape.size()));
  return TensorType(shape, tvm::runtime::DataType::Int(32));
}

MNM_OP_TYPE("mnm.op.shape_as_tensor", "ShapeAsTensor", ShapeAsTensorInfer);

}  // namespace op
}  // namespace mnm
