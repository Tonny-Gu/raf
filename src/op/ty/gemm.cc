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
 * \file src/op/ty/gemm.cc
 * \brief Typing of gemm operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using schema::BinaryArgs;

template <bool transpose_a, bool transpose_b>
Type MatmulInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x1));
  TensorType y = Downcast<TensorType>(GetType(args->x2));
  CHECK(x->shape.size() == 2 && y->shape.size() == 2);
  PrimExpr n1 = x->shape[0];
  PrimExpr m1 = x->shape[1];
  PrimExpr n2 = y->shape[0];
  PrimExpr m2 = y->shape[1];
  if (transpose_a) {
    std::swap(n1, m1);
  }
  if (transpose_b) {
    std::swap(n2, m2);
  }
  CHECK(TypeCheckCompare(m1, n2, std::equal_to<int>()))
      << "Matmul: shapes of x and y is inconsistent, "
      << " x shape=" << x->shape << ", y shape=" << y->shape;
  Array<tvm::PrimExpr> oshape = {n1, m2};
  return TensorType(oshape, x->dtype);
}

template <bool transpose_a, bool transpose_b>
Type BatchMatmulInfer(const CallValues& value) {
  const auto* args = value->args.as<BinaryArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x1));
  TensorType y = Downcast<TensorType>(GetType(args->x2));
  CHECK(x->shape.size() == 3 && y->shape.size() == 3);
  PrimExpr k1 = x->shape[0];
  PrimExpr n1 = x->shape[1];
  PrimExpr m1 = x->shape[2];
  PrimExpr k2 = y->shape[0];
  PrimExpr n2 = y->shape[1];
  PrimExpr m2 = y->shape[2];
  if (transpose_a) {
    std::swap(n1, m1);
  }
  if (transpose_b) {
    std::swap(n2, m2);
  }
  int64_t k1_v = k1.as<IntImmNode>()->value;
  int64_t k2_v = k2.as<IntImmNode>()->value;
  CHECK(k1_v == k2_v || k1_v == 1 || k2_v == 1)
      << "Incompatible broadcast type " << x << " and " << y;
  PrimExpr k = (k1_v > k2_v) ? k1 : k2;
  Array<tvm::PrimExpr> oshape = {k, n1, m2};
  return TensorType(oshape, x->dtype);
}

MNM_OP_TYPE("mnm.op.matmul", "Matmul", (MatmulInfer<false, false>));
MNM_OP_TYPE("mnm.op.matmul_nt", "MatmulNT", (MatmulInfer<false, true>));
MNM_OP_TYPE("mnm.op.matmul_tn", "MatmulTN", (MatmulInfer<true, false>));
MNM_OP_TYPE("mnm.op.matmul_tt", "MatmulTT", (MatmulInfer<true, true>));
MNM_OP_TYPE("mnm.op.dense", "DenseInfer", (MatmulInfer<false, true>));
MNM_OP_TYPE("mnm.op.batch_matmul", "BatchMatmulNN", (BatchMatmulInfer<false, false>));
MNM_OP_TYPE("mnm.op.batch_matmul_nt", "BatchMatmulNT", (BatchMatmulInfer<false, true>));
MNM_OP_TYPE("mnm.op.batch_matmul_tn", "BatchMatmulTN", (BatchMatmulInfer<true, false>));
MNM_OP_TYPE("mnm.op.batch_matmul_tt", "BatchMatmulTT", (BatchMatmulInfer<true, true>));

}  // namespace op
}  // namespace mnm
