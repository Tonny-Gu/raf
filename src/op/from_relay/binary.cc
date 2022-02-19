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
 * \file ./src/op/from_relay/binary.cc
 * \brief Operators bridged from Relay.
 */
#include "mnm/ir.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

using namespace mnm::ir;

#define MNM_BINARY_UFUNC_ATTR_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME)                            \
  MNM_OP_FROM_RELAY(RELAY_OP_NAME, MNM_OP_NAME,                                                    \
                    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) { \
                      Array<Expr> mnm_args = args;                                                 \
                      mnm_args.push_back(MakeConstant(NullValue<Value>()));                        \
                      mnm_args.push_back(MakeConstant(NullValue<Value>()));                        \
                      return mnm_args;                                                             \
                    })

MNM_BINARY_UFUNC_ATTR_OP_FROM_RELAY("add", "mnm.op.add");
MNM_BINARY_UFUNC_ATTR_OP_FROM_RELAY("subtract", "mnm.op.subtract");
MNM_GENERIC_ATTR_OP_FROM_RELAY("divide", "mnm.op.divide");
MNM_GENERIC_ATTR_OP_FROM_RELAY("floor_divide", "mnm.op.floor_divide");
MNM_GENERIC_ATTR_OP_FROM_RELAY("multiply", "mnm.op.multiply");
MNM_GENERIC_ATTR_OP_FROM_RELAY("power", "mnm.op.power");
MNM_GENERIC_ATTR_OP_FROM_RELAY("mod", "mnm.op.mod");
MNM_GENERIC_ATTR_OP_FROM_RELAY("less", "mnm.op.less");
MNM_GENERIC_ATTR_OP_FROM_RELAY("greater", "mnm.op.greater");
MNM_GENERIC_ATTR_OP_FROM_RELAY("less_equal", "mnm.op.less_equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("greater_equal", "mnm.op.greater_equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("equal", "mnm.op.equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("not_equal", "mnm.op.not_equal");
MNM_GENERIC_ATTR_OP_FROM_RELAY("maximum", "mnm.op.maximum");
MNM_GENERIC_ATTR_OP_FROM_RELAY("minimum", "mnm.op.minimum");
MNM_GENERIC_ATTR_OP_FROM_RELAY("logical_and", "mnm.op.logical_and");

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
