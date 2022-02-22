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
 * \file ./src/op/from_relay/nn.cc
 * \brief Operators bridged from Relay.
 */
#include "mnm/op_utils.h"
#include "tvm/relay/attrs/nn.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_GENERIC_ATTR_OP_FROM_RELAY("nn.batch_matmul", "mnm.op.batch_matmul_nt");
MNM_GENERIC_ATTR_OP_FROM_RELAY("nn.dense", "mnm.op.dense");

MNM_OP_FROM_RELAY("nn.conv2d", "mnm.op.conv2d",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<Conv2DAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));

                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but Meta currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->groups)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->kernel_layout)));
                    if (relay_attrs->out_layout != "") {
                      mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->out_layout)));
                    } else {
                      mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    }
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("nn.conv2d_transpose", "mnm.op.conv2d_transpose",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<Conv2DTransposeAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));

                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but Meta currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->output_padding)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->groups)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->kernel_layout)));
                    if (relay_attrs->out_layout != "") {
                      mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->out_layout)));
                    } else {
                      mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->data_layout)));
                    }
                    return mnm_args;
                  });

#define MNM_SOFTMAX_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME)                                      \
  MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME,                                                    \
                    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
                      Array<Expr> mnm_args = args;                                                 \
                      const auto* relay_attrs = attrs.as<SoftmaxAttrs>();                          \
                      mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));      \
                      return mnm_args;                                                             \
                    })

MNM_SOFTMAX_OP_FROM_RELAY("nn.softmax", "mnm.op.softmax");
MNM_SOFTMAX_OP_FROM_RELAY("nn.log_softmax", "mnm.op.log_softmax");

MNM_OP_FROM_RELAY("nn.bias_add", "mnm.op.bias_add",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<BiasAddAttrs>();
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("nn.max_pool2d", "mnm.op.max_pool2d",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<MaxPool2DAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->pool_size)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));
                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but Meta currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->ceil_mode)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(true)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("nn.avg_pool2d", "mnm.op.avg_pool2d",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<AvgPool2DAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->pool_size)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->strides)));
                    // Relay enforces 4-way padding to support asymmetric padding,
                    // but Meta currently only supports symmetric padding.
                    auto padding = ArrayToInt(relay_attrs->padding);
                    CHECK(padding[0] == padding[2] && padding[1] == padding[3])
                        << "Asymmetric padding for Conv2D is not supported yet";
                    padding.pop_back();
                    padding.pop_back();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(padding)));
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->dilation)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(relay_attrs->ceil_mode)));
                    mnm_args.push_back(MakeConstant(BoolValue::make(true)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
                    return mnm_args;
                  });

Array<Expr> AdaptivePoolFromRelay(const Attrs& attrs, const Array<Expr>& args,
                                  const VarValueMap& val_map) {
  Array<Expr> mnm_args = args;
  const auto* relay_attrs = attrs.as<AdaptivePool2DAttrs>();
  mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->output_size)));
  mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
  return mnm_args;
}

MNM_OP_FROM_RELAY("nn.adaptive_max_pool2d", "mnm.op.adaptive_max_pool2d", AdaptivePoolFromRelay);
MNM_OP_FROM_RELAY("nn.adaptive_avg_pool2d", "mnm.op.adaptive_avg_pool2d", AdaptivePoolFromRelay);

MNM_OP_FROM_RELAY("nn.layer_norm", "mnm.op.layer_norm",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<LayerNormAttrs>();
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->epsilon)));
                    return mnm_args;
                  });

MNM_OP_FROM_RELAY("nn.batch_norm", "mnm.op.batch_norm_train",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args;
                    const auto* relay_attrs = attrs.as<BatchNormAttrs>();
                    mnm_args.push_back(args[0]);                               // x
                    mnm_args.push_back(args[3]);                               // running_mean
                    mnm_args.push_back(args[4]);                               // running_var
                    mnm_args.push_back(args[1]);                               // w
                    mnm_args.push_back(args[2]);                               // b
                    mnm_args.push_back(MakeConstant(ScalarValue::make(0.1)));  // momentum
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->epsilon)));
                    return mnm_args;
                  });

Array<Array<Expr>> BatchNormMutationFromRelay(const Var& var, const Call& call) {
  Array<Array<Expr>> res = {
      {TryGetMayShare(call->args[1]), TupleGetItem(var, 1)},  // running_mean
      {TryGetMayShare(call->args[2]), TupleGetItem(var, 2)}   // running_var
  };
  return res;
}

MNM_OP_MUTATION_FROM_RELAY("nn.batch_norm", BatchNormMutationFromRelay);

MNM_OP_FROM_RELAY("nn.pad", "mnm.op.pad",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args;
                    mnm_args.push_back(args[0]);
                    const auto* relay_attrs = attrs.as<PadAttrs>();
                    Array<Integer> flat_pad_width;
                    for (int i = 0; i < relay_attrs->pad_width.size(); ++i) {
                      for (int j = 0; j < relay_attrs->pad_width[i].size(); ++j) {
                        flat_pad_width.push_back(relay_attrs->pad_width[i][j]);
                      }
                    }
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(flat_pad_width)));
                    const auto* konst = GetKonstFromValueMap(args[1], val_map);
                    CHECK(konst) << "'pad_value' must be a const tensor.";
                    mnm_args.push_back(MakeConstant(Constant2ScalarValue<double>(konst)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->pad_mode)));
                    return mnm_args;
                  });

// FIXME(@XIAO-XIA): Re-enable once dropout/dropout_dx can be dispatched to CuDNN.
// MNM_OP_FROM_RELAY("nn.dropout", "mnm.op._contrib_dropout",
//                   [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
//                     Array<Expr> mnm_args = args;
//                     const auto* relay_attrs = attrs.as<DropoutAttrs>();
//                     mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->rate)));
//                     return mnm_args;
//                   });
RELAY_REGISTER_OP("nn.dropout")
    .set_attr<op::FMNMFromRelay>("FMNMFromRelay", [](const Attrs& attrs, const Array<Expr>& args,
                                                     const VarValueMap& val_map) {
      LOG(WARNING) << "nn.dropout is unavailable in Meta, ignored";
      const auto* relay_attrs = attrs.as<DropoutAttrs>();

      Array<Expr> ret;
      ret.push_back(args[0]);
      ret.push_back(MakeConstant(ScalarValue::make(2)));
      return Tuple(std::move(ret));
    });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
