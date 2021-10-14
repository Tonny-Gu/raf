/*!
 * Copyright (c) 2019 by Contributors
 * \file value.h
 * \brief Definition of MNM values
 */
#pragma once
#include <tvm/relay/type.h>
#include <memory>
#include <vector>
#include <string>
#include "./ir.h"
#include "./tensor.h"
#include "./memory_pool.h"

namespace mnm {
namespace op {
class OpEnv;
}  // namespace op
}  // namespace mnm

// Basic values used in tensor algebra
namespace mnm {
namespace value {

/* Value type */
enum ValueType {
  kNullptr = 0,
  kIntValue = 1,
  kFloatValue = 2,
  kBoolValue = 3,
  kStringValue = 4,
  kTensorValue = 5,
  kTensorTypeValue = 6,
  kTupleValue = 7,
  kClosureValue = 8,
  kRefValue = 9,
  kOpValue = 10,
  kOpaqueValue = 11,
  kNoGradValue = 12,
  kVoidValue = 13,
};

/*!
 * \brief Convert value type key to value type.
 * \param type_key The type key.
 * \return The corresponding value type.
 */
ValueType TypeKey2ValueType(const char* type_key);

/*!
 * \brief Get the name of the value type.
 * \param type The value type.
 * \return The name.
 */
std::string ValueType2String(ValueType type);

/* Value */
class ValueObj : public ir::Object {
 public:
  mutable std::shared_ptr<op::OpEnv> op_env{nullptr};
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mnm.value.Value";
  MNM_BASE_OBJECT(ValueObj, ir::Object);
};

class Value : public ir::ObjectRef {
 public:
  operator DLTensor*() const;
  operator tensor::Tensor &() const;
  template <typename TValue,
            typename = typename std::enable_if<std::is_base_of<Value, TValue>::value>::type>
  explicit operator TValue() const {
    return ir::Downcast<TValue>(*this);
  }
  MNM_OBJECT_REF(Value, ir::ObjectRef, ValueObj);
};

// Scalar values

class IntValue;
class FloatValue;
class BoolValue;

/* ScalarValue */
class ScalarValueObj : public ValueObj {
 public:
  ir::DataType dtype;
  static constexpr const uint32_t _type_index = tvm::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mnm.value.ScalarValue";
  MNM_BASE_OBJECT(ScalarValueObj, ValueObj);
};

class ScalarValue : public Value {
 public:
  static IntValue make(int8_t value);
  static IntValue make(int16_t value);
  static IntValue make(int32_t value);
  static IntValue make(int64_t value);
  static IntValue make(uint8_t value);
  static IntValue make(uint16_t value);
  static IntValue make(uint32_t value);
  static IntValue make(uint64_t value);
  static FloatValue make(float value);
  static FloatValue make(double value);
  static BoolValue make(bool value);
  MNM_OBJECT_REF(ScalarValue, Value, ScalarValueObj);
};

/* IntValue */
class IntValueObj : public ScalarValueObj {
 public:
  int64_t value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.IntValue";
  MNM_FINAL_OBJECT(IntValueObj, ScalarValueObj);
};

class IntValue : public ScalarValue {
 public:
  static IntValue make(ir::DataType dtype, int64_t value);
  MNM_OBJECT_REF(IntValue, ScalarValue, IntValueObj);
};

/* FloatValue */
class FloatValueObj : public ScalarValueObj {
 public:
  double value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.FloatValue";
  MNM_FINAL_OBJECT(FloatValueObj, ScalarValueObj);
};

class FloatValue : public ScalarValue {
 public:
  static FloatValue make(ir::DataType dtype, double value);
  MNM_OBJECT_REF(FloatValue, ScalarValue, FloatValueObj);
};

/* BoolValue */
class BoolValueObj : public ScalarValueObj {
 public:
  bool value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.BoolValue";
  MNM_FINAL_OBJECT(BoolValueObj, ScalarValueObj);
};

class BoolValue : public ScalarValue {
 public:
  static BoolValue make(bool data);
  MNM_OBJECT_REF(BoolValue, ScalarValue, BoolValueObj);
};

/* BaseTensorValue */
class BaseTensorValueObj : public ValueObj {
 public:
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mnm.value.BaseTensorValue";
  MNM_BASE_OBJECT(BaseTensorValueObj, ValueObj);
};

class BaseTensorValue : public Value {
 public:
  MNM_OBJECT_REF(BaseTensorValue, Value, BaseTensorValueObj);
};

/* TensorValue */
class TensorValueObj final : public BaseTensorValueObj {
 public:
  mutable tensor::Tensor tensor;
  mutable std::shared_ptr<memory_pool::Memory> mem{nullptr};
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_tensor", &tensor);
  }
  static constexpr const char* _type_key = "mnm.value.TensorValue";
  MNM_FINAL_OBJECT(TensorValueObj, BaseTensorValueObj);
};

class TensorValue final : public BaseTensorValue {
 public:
  static TensorValue make(tensor::Tensor tensor,
                          std::shared_ptr<memory_pool::Memory> mem = nullptr);
  static TensorValue Assemble(const Device& dev, const DType& dtype,
                              const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides = {}, void* data = nullptr,
                              std::shared_ptr<memory_pool::Memory> mem = nullptr);
  static TensorValue Assemble(const Device& dev, const DType& dtype,
                              const ir::Array<IntValue> shape,
                              const std::vector<int64_t>& strides = {}, void* data = nullptr,
                              std::shared_ptr<memory_pool::Memory> mem = nullptr);
  TensorValue CreateView(const std::vector<int64_t>& shape = {},
                         const std::vector<int64_t>& strides = {}) const;
  MNM_OBJECT_REF(TensorValue, BaseTensorValue, TensorValueObj);
};

/* TensorTypeValue */
class TensorTypeValueObj final : public BaseTensorValueObj {
 public:
  tvm::relay::TensorType type;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_type", &type);
  }
  static constexpr const char* _type_key = "mnm.value.TensorTypeValue";
  MNM_FINAL_OBJECT(TensorTypeValueObj, BaseTensorValueObj);
};

class TensorTypeValue final : public BaseTensorValue {
 public:
  static TensorTypeValue make(tvm::relay::TensorType type);
  MNM_OBJECT_REF(TensorTypeValue, BaseTensorValue, TensorTypeValueObj);
};

/* TupleValue */
class TupleValueObj final : public ValueObj {
 public:
  ir::Array<Value> fields;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_fields", &fields);
  }
  static constexpr const char* _type_key = "mnm.value.TupleValue";
  MNM_FINAL_OBJECT(TupleValueObj, ValueObj);
};

class TupleValue final : public Value {
 public:
  static TupleValue make(ir::Array<Value> fields);
  MNM_OBJECT_REF(TupleValue, Value, TupleValueObj);
};

/* ClosureValue */
class ClosureValueObj final : public ValueObj {
 public:
  ir::Map<ir::Var, Value> env;
  ir::Function func;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_env", &env);
    v->Visit("_func", &func);
  }
  static constexpr const char* _type_key = "mnm.value.ClosureValue";
  MNM_FINAL_OBJECT(ClosureValueObj, ValueObj);
};

class ClosureValue final : public Value {
 public:
  static ClosureValue make(ir::Map<ir::Var, Value> env, ir::Function func);
  MNM_OBJECT_REF(ClosureValue, Value, ClosureValueObj);
};

/* RefValue */
class RefValueObj final : public ValueObj {
 public:
  mutable Value value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.RefValue";
  MNM_FINAL_OBJECT(RefValueObj, ValueObj);
};

class RefValue final : public Value {
 public:
  static RefValue make(Value value);
  MNM_OBJECT_REF(RefValue, Value, RefValueObj);
};

/* OpValue */
class OpValueObj final : public ValueObj {
 public:
  ir::Op op;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_op", &op);
  }
  static constexpr const char* _type_key = "mnm.value.OpValue";
  MNM_FINAL_OBJECT(OpValueObj, ValueObj);
};

class OpValue final : public Value {
 public:
  static OpValue make(ir::Op op);
  MNM_OBJECT_REF(OpValue, Value, OpValueObj);
};

/* ConstructorValue */
class ConstructorValueObj;
class ConstructorValue;

/* OpaqueValue */
class OpaqueValueObj : public ValueObj {
 public:
  mutable Value data{nullptr};
  static constexpr const char* _type_key = "mnm.value.OpaqueValue";
  MNM_FINAL_OBJECT(OpaqueValueObj, ValueObj);
};

class OpaqueValue : public Value {
 public:
  MNM_OBJECT_REF(OpaqueValue, Value, OpaqueValueObj);
};

/* StringValue */
class StringValueObj : public ValueObj {
 public:
  std::string value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.StringValue";
  MNM_FINAL_OBJECT(StringValueObj, ValueObj);
};

class StringValue : public Value {
 public:
  static StringValue make(const std::string& data);
  MNM_OBJECT_REF(StringValue, Value, StringValueObj);
};

/* Specific values */
class NoGradValueObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
  }
  static constexpr const char* _type_key = "mnm.value.NoGradValue";
  MNM_FINAL_OBJECT(NoGradValueObj, ValueObj);
};

class NoGradValue : public Value {
 public:
  static NoGradValue make();
  MNM_OBJECT_REF(NoGradValue, Value, NoGradValueObj);
};

/* Null values */
class VoidValueObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
  }
  static constexpr const char* _type_key = "mnm.value.VoidValue";
  MNM_FINAL_OBJECT(VoidValueObj, ValueObj);
};

class VoidValue : public Value {
 public:
  static VoidValue make();
  MNM_OBJECT_REF(VoidValue, Value, VoidValueObj);
};

template <class T>
T GetScalarValueData(const Value& value);

template <>
bool GetScalarValueData<bool>(const Value& value);

template <>
float GetScalarValueData<float>(const Value& value);

/*!
 * \brief Copy a value to specified device.
 * \param src Value to be copyed.
 * \param dev The pecified device.
 * \return The copyed value.
 */
Value CopyTo(Value src, const Device& dev);

/*!
 * \brief Create a dummy value according to type.
 * \param type The tensor or tuple type.
 * \param device The device to allocate memory from when creating a tensor.
 * \return The dummy value.
 */
Value CreateDummyValueFromType(const tvm::Type& type, Device device);

}  // namespace value
}  // namespace mnm
