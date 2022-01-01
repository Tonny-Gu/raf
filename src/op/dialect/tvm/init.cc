/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dialect/tvm/init.cc
 * \brief Init operators bridged from TVM.
 */
#include "mnm/value.h"
#include "./tvm_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/init.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::tensor;
using namespace mnm::op::schema;

std::vector<Value> InitOpSchema2Args(const InitOpArgs* args) {
  return {};
}

std::vector<std::string> InitOpSchemaArgNames(const op::CallValues& call) {
  return {};
}

Attrs InitOpSchema2Attrs(const InitOpArgs* args) {
  auto attrs = make_object<InitOpAttrs>();
  std::vector<int64_t> shape_vec = GetShapeVecFromValue(args->shape);
  Array<Integer> shape;
  for (size_t i = 0; i < shape_vec.size(); ++i) {
    shape.push_back(shape_vec[i]);
  }
  attrs->shape = shape;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey InitOpHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const InitOpArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  key << shape;
  key << ir::String2DLDataType(args->dtype);
  key << args->device;
  return key;
}

MNM_TVM(zeros, Zeros, InitOpArgs, InitOpSchema2Args, InitOpSchemaArgNames, InitOpSchema2Attrs,
        InitOpHasher, kElemWise);
MNM_TVM(ones, Ones, InitOpArgs, InitOpSchema2Args, InitOpSchemaArgNames, InitOpSchema2Attrs,
        InitOpHasher, kElemWise);

std::vector<Value> OneHotSchema2Args(const OneHotArgs* args) {
  return {args->indices, args->on_value, args->off_value};
}

std::vector<std::string> OneHotSchemaArgNames(const op::CallValues& call) {
  return {"indices", "on_value", "off_value"};
}

Attrs OneHotSchema2Attrs(const OneHotArgs* args) {
  auto attrs = make_object<OneHotAttrs>();
  attrs->depth = args->depth;
  attrs->axis = args->axis;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey OneHotHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const OneHotArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->depth;
  key << args->axis;
  key << ir::String2DLDataType(args->dtype);
  key << args->device;
  return key;
}

MNM_TVM(one_hot, OneHot, OneHotArgs, OneHotSchema2Args, OneHotSchemaArgNames, OneHotSchema2Attrs,
        OneHotHasher, kOutEWiseFusable);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
