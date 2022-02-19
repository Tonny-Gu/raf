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
 * \file src/impl/ir_ext.cc
 * \brief MNM extension to TVM/Relay IR.
 */
#include <printer/text_printer.h>
#include "mnm/ir_ext.h"
#include "mnm/registry.h"
#include "mnm/pass.h"
#include "mnm/value.h"

namespace mnm {
namespace ir {

using namespace mnm::value;

IRModule GlobalModule() {
  static IRModule inst = IRModule();
  return inst;
}

tvm::runtime::NDArray MakeFakeTensor() {
  static int64_t a[1] = {-114514};
  static int64_t b[1] = {1};
  Device dev = mnm::Device(mnm::DevType::kCPU(), 0);
  DType dtype = mnm::DType(mnm::DTypeCode::kInt(), 64, 1);
  DLTensor tensor;
  tensor.data = a;
  tensor.device = dev;
  tensor.dtype = dtype;
  tensor.shape = b;
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty({}, dtype, dev);
  array.CopyFrom(&tensor);
  return array;
}

bool ConstantNode::IsTensor() const {
  return value.defined() && value.as<BaseTensorValueObj>();
}

bool ConstantNode::IsScalar() const {
  return value.defined() && value.as<ScalarValueObj>();
}

ObjectPtr<ConstantNode> MakeConstantNode(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  static const auto fake_tensor = MakeFakeTensor();
  n->data = fake_tensor;
  n->value = std::move(node_ref);
  return n;
}

RelayConstant MakeConstant(ObjectRef node_ref) {
  ObjectPtr<ConstantNode> n = MakeConstantNode(node_ref);
  return RelayConstant(n);
}

RelayConstant MakeNull() {
  return MakeConstant(NullValue<Value>());
}

ObjectRef ConstantExtractValue(RelayConstant _node) {
  const ConstantNode* node = static_cast<const ConstantNode*>(_node.get());
  return node->value;
}

Var MakeVar_(Id vid, Type type_annotation, Var may_share = Var()) {
  ObjectPtr<ExtendedVarNode> n = make_object<ExtendedVarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->may_share = may_share;
  return Var(n);
}

Var MakeVar(const std::string& name_hint, Type type_annotation, Var may_share) {
  return MakeVar_(Id(name_hint), type_annotation, may_share);
}

Var GetMayShare(Expr var) {
  const auto* vn = var.as<ExtendedVarNode>();
  CHECK(vn);
  return vn->may_share;
}

Var TryGetMayShare(Expr var) {
  const auto* vn = var.as<ExtendedVarNode>();
  CHECK(vn);
  while (vn->may_share.defined()) {
    vn = vn->may_share.as<ExtendedVarNode>();
    CHECK(vn);
  }
  return GetRef<Var>(vn);
}

std::string AsText(const ObjectRef& node, bool show_meta_data) {
  auto annotate =
      tvm::runtime::TypedPackedFunc<String(ObjectRef)>([](const ObjectRef& expr) -> String {
        std::ostringstream os;
        const auto* constant = expr.as<ConstantNode>();
        if (constant) {
          if (constant->value.defined()) {
            os << constant->value;
          } else {
            os << "nullptr";
          }
        }
        if (expr.as<CallNode>() && Downcast<Expr>(expr)->checked_type_.defined()) {
          auto meta = tvm::TextMetaDataContext();
          tvm::relay::RelayTextPrinter printer(false, &meta, nullptr);
          os << " /* ty=" << printer.Print(Downcast<Expr>(expr)->checked_type()).str() << " */";
        }
        const auto* extended_var = expr.as<ExtendedVarNode>();
        if (extended_var && extended_var->may_share.defined()) {
          os << "(share: %" << extended_var->may_share->name_hint() << ")";
        }
        return String(os.str());
      });

  std::string ret = tvm::AsText(node, show_meta_data, annotate);
  size_t index = 0;
  while (true) {
    index = ret.find("-114514", index);
    if (index == std::string::npos) {
      break;
    }
    ret.replace(index, 7, "");
  }
  return ret;
}

MNM_REGISTER_GLOBAL("mnm.ir.AsText").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  ObjectRef value = args[0];
  bool show_meta_data = args.size() == 2 ? args[1] : false;
  *rv = AsText(value, show_meta_data);
});

MNM_REGISTER_GLOBAL("mnm.ir._make.Constant").set_body_typed(MakeConstant);
MNM_REGISTER_GLOBAL("mnm.ir._make.Var").set_body_typed(MakeVar);
MNM_REGISTER_GLOBAL("mnm.ir.constant.ExtractValue").set_body_typed(ConstantExtractValue);
MNM_REGISTER_GLOBAL("mnm.ir.variable.GetMayShare").set_body_typed(GetMayShare);
MNM_REGISTER_GLOBAL("mnm.ir.module.Global").set_body_typed(GlobalModule);

}  // namespace ir
}  // namespace mnm
