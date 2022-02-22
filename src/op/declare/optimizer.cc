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
 * \file src/op/declare/nn.cc
 * \brief Declaration of nn-specific operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/optimizer.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.sgd", [](const CallValues& call) {
  const auto* args = call->args.as<SgdArgs>();
  CHECK(args != nullptr);
  const DLTensor* x0 = args->x;
  const DLTensor* dx = args->dx;
  const DLTensor* v0 = args->v;
  CHECK_EQ(x0->ndim, dx->ndim);
  CHECK_EQ(v0->ndim, dx->ndim);
  for (int i = 0; i < x0->ndim; ++i) {
    CHECK_EQ(x0->shape[i], dx->shape[i]);
    CHECK_EQ(v0->shape[i], dx->shape[i]);
  }
  auto v1 = TensorValue::Assemble(
      /*dev=*/dx->device,
      /*dtype=*/dx->dtype,
      /*shape=*/std::vector<int64_t>(dx->shape, dx->shape + dx->ndim));
  auto x1 = TensorValue::Assemble(
      /*dev=*/dx->device,
      /*dtype=*/dx->dtype,
      /*shape=*/std::vector<int64_t>(dx->shape, dx->shape + dx->ndim));
  call->out = TupleValue::make(tvm::Array<Value>({v1, x1}));
  call->device = dx->device;
});

void LansDecl(const CallValues& call) {
  const auto* args = call->args.as<LansArgs>();
  CHECK(args != nullptr);
  CHECK(args->tensor_list.size() % 4 == 0);
  const DLTensor* x = args->tensor_list[0];
  call->device = x->device;
  int ntensors = args->tensor_list.size() / 4;
  Array<Value> output;
  for (int i = 0; i < args->tensor_list.size(); ++i) {
    output.push_back(args->tensor_list[i]);
  }
  call->out = TupleValue::make(output);
}

MNM_OP_DECLARE("mnm.op.lans", LansDecl)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TMNMInplaceUpdate>("TMNMInplaceUpdate", {{0, 0}});
}  // namespace declare
}  // namespace op
}  // namespace mnm
