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
 * \file src/op/regs/value2schema.h
 * \brief Converters from values to MNM operator schemas
 */
#pragma once
#include <string>
#include <vector>
#include "mnm/value.h"
#include "./regs_utils.h"

namespace mnm {
namespace op {
namespace regs {
namespace value2schema {

using mnm::ir::Array;

#define MNM_PRELUDE_ALLOW_NULL() \
  using namespace mnm::value;    \
  using namespace mnm::ir;       \
  if (!a.defined()) {            \
    return {};                   \
  }

#define MNM_PRELUDE_DISALLOW_NULL(type)                                             \
  using namespace mnm::value;                                                       \
  using namespace mnm::ir;                                                          \
  if (!a.defined()) {                                                               \
    LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\""             \
               << " is undefined (NULL), but is required to be of type " << (type); \
    throw;                                                                          \
  }

inline value::Value ArrayLike(const value::Value& a) {
  MNM_PRELUDE_ALLOW_NULL();
  if (a->IsInstance<IntValueObj>() || a->IsInstance<FloatValueObj>() ||
      a->IsInstance<BoolValueObj>() || a->IsInstance<BaseTensorValueObj>() ||
      a->IsInstance<TupleValueObj>() || a->IsInstance<VoidValueObj>() ||
      a->IsInstance<OpValueObj>() || a->IsInstance<ClosureValueObj>()) {
    return a;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not array-like";
  throw;
}

inline ir::Optional<value::Value> OptionalArrayLike(const value::Value& a) {
  if (!a.defined()) {
    return tvm::NullOpt;
  }
  return ArrayLike(a);
}

inline value::BaseTensorValue Tensor(const value::Value& a) {
  MNM_PRELUDE_ALLOW_NULL();
  if (const auto* v = a.as<BaseTensorValueObj>()) {
    return GetRef<BaseTensorValue>(v);
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a tensor";
  throw;
}

inline ir::Optional<value::BaseTensorValue> OptionalTensor(const value::Value& a) {
  if (!a.defined()) {
    return tvm::NullOpt;
  }
  return Tensor(a);
}

inline int64_t Int(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("an integer");
  if (const auto* v = a.as<IntValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not an integer";
  throw;
}
inline bool Bool(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("boolean");
  if (const auto* v = a.as<BoolValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a bool value";
  throw;
}
inline double Double(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("double");
  if (const auto* v = a.as<FloatValueObj>()) {
    return v->value;
  }
  if (const auto* v = a.as<IntValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a double";
  throw;
}
inline std::string String(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("string");
  if (const auto* v = a.as<StringValueObj>()) {
    return v->value;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not a string";
  throw;
}

inline std::vector<int64_t> TupleInt(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("tuple of integers");
  if (const auto* v = a.as<TupleValueObj>()) {
    std::vector<int64_t> ret;
    ret.reserve(v->fields.size());
    for (const ObjectRef& i : v->fields) {
      if (const auto* e = i.as<IntValueObj>()) {
        ret.push_back(e->value);
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of integers, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not tuple of integers";
  throw;
}

inline std::vector<int64_t> IntOrTupleInt(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("an integer or tuple of integers");
  if (const auto* v = a.as<IntValueObj>()) {
    return {v->value};
  }
  if (const auto* v = a.as<TupleValueObj>()) {
    std::vector<int64_t> ret;
    ret.reserve(v->fields.size());
    for (const ObjectRef& i : v->fields) {
      if (const auto* e = i.as<IntValueObj>()) {
        ret.push_back(e->value);
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not an integer or "
                    "tuple of integers, because the "
                 << ToOrdinal(ret.size()) << " member is of type \"" << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not an integer or tuple of integers";
  throw;
}

inline ir::Optional<Array<value::IntValue>> IntArray(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("array of integers");
  if (const auto* v = a.as<IntValueObj>()) {
    return Array<value::IntValue>{value::IntValue::make(v->dtype, v->value)};
  }
  if (const auto* v = a.as<value::TupleValueObj>()) {
    Array<value::IntValue> ret;
    ret.reserve(v->fields.size());
    for (const tvm::runtime::ObjectRef& i : v->fields) {
      if (const auto* e = i.as<value::IntValueObj>()) {
        ret.push_back(value::IntValue::make(e->dtype, e->value));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not an integer or "
                    "tuple of integers, because the "
                 << ToOrdinal(ret.size()) << " member is of type \"" << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  } else if (auto* v = a.as<value::TensorTypeValueObj>()) {
    return tvm::NullOpt;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not an integer or tuple of integers";
  throw;
}

inline std::vector<value::BaseTensorValue> TupleTensor(const value::Value& a) {
  MNM_PRELUDE_DISALLOW_NULL("tuple of tensors");
  if (const auto* v = a.as<TupleValueObj>()) {
    std::vector<BaseTensorValue> ret;
    ret.reserve(v->fields.size());
    for (const ObjectRef& i : v->fields) {
      if (const auto* e = i.as<BaseTensorValueObj>()) {
        ret.push_back(Downcast<BaseTensorValue>(i));
        continue;
      }
      LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" is not tuple of tensors, "
                 << "because the " << ToOrdinal(ret.size()) << " member is of type \""
                 << i->GetTypeKey() << '"';
      throw;
    }
    return ret;
  }
  LOG(FATAL) << "TypeError: In operator \"{op}\", argument \"{arg}\" of type \"" << a->GetTypeKey()
             << "\" is not tuple of tensors";
  throw;
}

#undef MNM_PRELUDE_DISALLOW_NULL
#undef MNM_PRELUDE_ALLOW_NULL

}  // namespace value2schema
}  // namespace regs
}  // namespace op
}  // namespace mnm
