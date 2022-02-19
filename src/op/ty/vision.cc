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
 * \file src/op/ty/vision.cc
 * \brief Typing of vision operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/vision.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;

Type GetValidCountsInfer(const CallValues& value) {
  const auto* args = value->args.as<GetValidCountsArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  CHECK_EQ(data->shape.size(), 3) << "ValueError: Input data should be 3-D";

  Array<Type> ret;
  Array<PrimExpr> oshape(data->shape.begin(), data->shape.begin() + 1);
  Array<PrimExpr> data_shape(data->shape);
  Array<PrimExpr> oshape_indices(data->shape.begin(), data->shape.begin() + 2);
  ret.push_back(TensorType(oshape, DataType::Int(32)));
  ret.push_back(TensorType(data_shape, data->dtype));
  ret.push_back(TensorType(oshape_indices, DataType::Int(32)));
  return TupleType(ret);
}

MNM_OP_TYPE("mnm.op.get_valid_counts", "GetValidCounts", GetValidCountsInfer);

Type NonMaxSuppressionInfer(const CallValues& value) {
  const auto* args = value->args.as<NonMaxSuppressionArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType valid_count = Downcast<TensorType>(GetType(args->valid_count));
  CHECK_EQ(data->shape.size(), 3) << "ValueError: Input data should be 3-D";
  CHECK_EQ(valid_count->shape.size(), 1) << "ValueError: Input valid count should be 1-D";

  if (args->return_indices) {
    Array<Type> ret;
    Array<PrimExpr> oshape(data->shape.begin(), data->shape.begin() + 2);
    Array<PrimExpr> count_shape({data->shape[0], 1});
    ret.push_back(TensorType(oshape, DataType::Int(32)));
    ret.push_back(TensorType(count_shape, DataType::Int(32)));
    return TupleType(ret);
  } else {
    return data;
  }
}

MNM_OP_TYPE("mnm.op.non_max_suppression", "NonMaxSuppression", NonMaxSuppressionInfer);

Type RoiAlignInfer(const CallValues& value) {
  const auto* args = value->args.as<RoiAlignArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType rois = Downcast<TensorType>(GetType(args->rois));
  CHECK_EQ(data->shape.size(), 4) << "Input data should be 4-D.";
  CHECK_EQ(rois->shape.size(), 2) << "Input rois should be 2-D.";
  // assign output type
  std::vector<PrimExpr> oshape;
  if (args->layout == "NCHW") {
    oshape.push_back(rois->shape[0]);
    oshape.push_back(data->shape[1]);
    oshape.push_back(int32_t(args->pooled_size[0]));
    oshape.push_back(int32_t(args->pooled_size[1]));
  } else {
    ICHECK_EQ(args->layout, "NHWC") << "Unexpected ROI Align layout " << args->layout;
    oshape.push_back(rois->shape[0]);
    oshape.push_back(int32_t(args->pooled_size[0]));
    oshape.push_back(int32_t(args->pooled_size[1]));
    oshape.push_back(data->shape[3]);
  }
  return TensorType(oshape, data->dtype);
}

MNM_OP_TYPE("mnm.op.roi_align", "RoiAlign", RoiAlignInfer);

Type RoiAlignDxInfer(const CallValues& value) {
  const auto* args = value->args.as<RoiAlignDxArgs>();
  CHECK(args != nullptr);
  return Downcast<TensorType>(GetType(args->data));
}

MNM_OP_TYPE("mnm.op.roi_align_dx", "RoiAlignDx", RoiAlignDxInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
