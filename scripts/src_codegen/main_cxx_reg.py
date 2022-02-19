# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from . import def_op
from . import def_schema
from .codegen_utils import NORM_MAP, snake_to_pascal, write_to_file

# pylint: disable=invalid-name

# TODO(@junrushao1994): this is used to be compatible with legacy
NORM_CONVERTER = {
    "ToAny": "ArrayLike",
    "ToAnyOptional": "OptionalArrayLike",
    "ToTensor": "Tensor",
    "ToOptionalTensor": "OptionalTensor",
    "ToInt": "Int",
    "ToBool": "Bool",
    "ToString": "String",
    "ToDouble": "Double",
    "ToIntTuple": "IntOrTupleInt",
    "ToIntArray": "IntArray",
    "ToTensorTuple": "TupleTensor",
}


def gen_file(filename):
    FILE = """
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
 * Auto generated. Do not touch.
 * \\file {FILENAME}
 * \\brief Register op schemas.
 */
#include <algorithm>
#include <array>
#include "./regs_utils.h"
#include "./ffi2expr.h"
#include "./ffi2schema.h"
#include "./value2schema.h"
#include "./schema2value.h"
#include "../schema/list_args.h"
{INCLUDES}

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::registry;
using namespace mnm::binding;
using mnm::op::FMNMSchema;
using mnm::op::FMNMSchemaFieldIndex;
using mnm::executor::interpreter::InvokePrimitive;

// Part 0. Op names
namespace mnm {{
namespace op {{
namespace regs {{
namespace names {{
{OP_NAMES}
}}  // namespace names
}}  // namespace regs
}}  // namespace op
}}  // namespace mnm

// Part 1.1. FFI to schema (for each schema)
{FFI_TO_SCHEMA_PRELUDE}

{FFI_TO_SCHEMAS}

{FFI_TO_SCHEMA_EPILOG}

// Part 1.2. Imperative API, uses "Part 1.1. FFI to schema"
{IMPERATIVE_API_PRELUDE}

{IMPERATIVE_APIS}

{IMPERATIVE_API_EPILOG}

// Part 2.1. FFI to Array<Expr> (for each schema)
{FFI_TO_EXPR_PRELUDE}

{FFI_TO_EXPRS}

{FFI_TO_EXPR_EPILOG}

// Part 2.2. Symbolic API, uses "Part 2.1. FFI to Array<Expr>"
{SYMBOLIC_API_PRELUDE}

{SYMBOLIC_APIS}

{SYMBOLIC_API_EPILOG}

// Part 3.1. Array<Value> to schema (for each schema)
{VALUE_TO_SCHEMA_PRELUDE}

{VALUE_TO_SCHEMAS}

{VALUE_TO_SCHEMA_EPILOG}

// Part 3.2. Schema field index (for each schema)
{SCHEMA_FIELD_IDX_PRELUDE}

{SCHEMA_FIELD_IDX}

{SCHEMA_FIELD_IDX_EPILOG}

// Part 3.3. FMNMSchema API, uses Part 3.1 and Part 3.2
{F_MNM_SCHEMA_PRELUDE}

{F_MNM_SCHEMAS}

{F_MNM_SCHEMA_EPILOG}

// The last part: registering schemas
namespace mnm {{
namespace op {{
namespace schema {{
namespace {{
MNM_REGISTER_OBJECT_REFLECT(ListArgs);
{SCHEMA_REGS}
}}  // namespace
}}  // namespace schema
}}  // namespace op
}}  // namespace mnm
""".strip()
    schema_headers = def_schema.by_file()
    ops = def_op.by_name()
    schemas = dict()
    for sub_schemas in schema_headers.values():
        schemas.update(sub_schemas)
    ops = [ops[name] for name in sorted(ops.keys())]
    schemas = [(name, schemas[name]) for name in sorted(schemas.keys())]
    schema_headers = list(sorted(schema_headers.keys()))
    # includes
    includes = "\n".join(map(gen_include, schema_headers))
    # Part 0. Op names
    op_names = "\n".join(map(gen_op_name, ops))
    # Part 1.1. FFI to schema (for each schema)
    ffi2schemas = "\n\n".join(map(gen_ffi_to_schema, schemas))
    # Part 1.2. Imperative API, uses "Part 1.1. FFI to schema"
    imperative_apis = "\n\n".join(map(gen_imperative_api, ops))
    # Part 2.1. FFI to Array<Expr> (for each schema)
    ffi2exprs = "\n\n".join(map(gen_ffi_to_expr, schemas))
    # Part 2.2. Symbolic API, uses "Part 2.1. FFI to Array<Expr>"
    symbolic_apis = "\n".join(map(gen_symbolic_api, ops))
    # Part 3.1. Array<Value> to schema (for each schema)
    value2schemas = "\n\n".join(map(gen_value_to_schema, schemas))
    # Part 3.2. Schema field index (for each schema)
    schema_field_idx = "\n\n".join(map(gen_schema_field_idx, schemas))
    # Part 3.2. FMNMSchema API, uses "Part 3.1. Array<Value> to schema"
    f_mnm_schemas = "\n".join(map(gen_f_mnm_schema, ops))
    # The last part: registering schemas
    schema_regs = "\n".join(map(gen_schema_reg, schemas))
    if filename.startswith("./"):
        filename = filename[2:]
    return FILE.format(
        FILENAME=filename,
        INCLUDES=includes,
        OP_NAMES=op_names,
        FFI_TO_SCHEMAS=ffi2schemas,
        IMPERATIVE_APIS=imperative_apis,
        FFI_TO_EXPRS=ffi2exprs,
        SYMBOLIC_APIS=symbolic_apis,
        VALUE_TO_SCHEMAS=value2schemas,
        SCHEMA_FIELD_IDX=schema_field_idx,
        F_MNM_SCHEMAS=f_mnm_schemas,
        SCHEMA_REGS=schema_regs,
        **globals()
    )


def gen_include(filename):
    INCLUDE = """
#include "../schema/{FILE}"
""".strip()
    return INCLUDE.format(FILE=filename)


def gen_op_name(op):
    OP_NAME = """
static const char {OP_VAR}[] = "mnm.op.{OP_NAME}";
""".strip()
    return OP_NAME.format(OP_VAR=op.name.replace(".", "_"), OP_NAME=op.name)


def gen_schema_reg(_schema):
    SCHEMA_REG = """
MNM_REGISTER_OBJECT_REFLECT({CLASS_NAME});
""".strip()
    name, _ = _schema
    class_name = snake_to_pascal(name) + "Args"
    return SCHEMA_REG.format(CLASS_NAME=class_name)


def add_no_lint(line):
    if len(line) >= 100:
        return line + "  // NOLINT(whitespace/line_length)"
    return line


#### Part 1.1. FFI to schema (for each schema) ####


FFI_TO_SCHEMA_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace ffi2schema {

#define MNM_TAPE(i, norm, name)               \\
  try {                                       \\
    attrs->name = norm(values[i], tapes + i); \\
  } catch (const dmlc::Error& e) {            \\
    FillError(e, "{arg}", #name);             \\
  }

#define MNM_POD(i, norm, name)     \\
  try {                            \\
    attrs->name = norm(values[i]); \\
  } catch (const dmlc::Error& e) { \\
    FillError(e, "{arg}", #name);  \\
  }

#define MNM_PRELUDE(obj, n)                                                                \\
  const int size = values.size();                                                          \\
  CHECK_EQ(size, n) << "TypeError: Mismatched number of arguments for operator \\"{op}\\": " \\
                    << "Expected " << n << ", but get " << size;                           \\
  auto attrs = make_object<obj>();
""".strip()

FFI_TO_SCHEMA_EPILOG = """
#undef MNM_PRELUDE
#undef MNM_POD
#undef MNM_TAPE
}  // namespace ffi2schema
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_ffi_to_schema(_schema):
    FFI_TO_SCHEMA = """
Attrs {SCHEMA_NAME}(const TVMArgs& values, GradTape* tapes) {{
  MNM_PRELUDE(schema::{SCHEMA_NAME}Args, {N_ARGS});  // NOLINT(whitespace/line_length)
{ARGS}
  return Attrs(attrs);
}}
""".strip()
    ARG = (
        " " * 2
        + """
  MNM_{OPTION}({I}, ffi2schema::{NORM}, {ARG_NAME});
""".strip()
    )
    schema_name, schema = _schema
    schema_name = snake_to_pascal(schema_name)
    n_args = len(schema)
    args = []
    for i, entry in enumerate(schema):
        norm = NORM_CONVERTER[NORM_MAP[entry.cxx_normalizer or entry.cxx_type]]
        arg_name = entry.name
        if norm in ["ArrayLike", "Tensor", "OptionalArrayLike", "OptionalTensor"]:
            option = "TAPE"
        else:
            option = "POD"
        args.append(ARG.format(I=i, NORM=norm, ARG_NAME=arg_name, OPTION=option))
    args = "\n".join(map(add_no_lint, args))
    return FFI_TO_SCHEMA.format(SCHEMA_NAME=schema_name, N_ARGS=n_args, ARGS=args)


##### Part 1.2. Imperative API, uses "Part 1.1. FFI to schema" ####

IMPERATIVE_API_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace imperative {

#define MNM_PRELUDE(op, n_args, func, obj)                                                     \\
  const auto* opack = OpPack<names::op, n_args>::Get();                                        \\
  const auto* vpack = VarPack::Get();                                                          \\
  std::array<GradTape, n_args> prev_tapes;                                                     \\
  std::vector<Expr> grads(opack->grads.begin(), opack->grads.end());                           \\
  Attrs _schema;                                                                               \\
  try {                                                                                        \\
    _schema = func(args, prev_tapes.data());                                                   \\
  } catch (const dmlc::Error &e) {                                                             \\
    FillError(e, "{op}", names::op);                                                           \\
  }                                                                                            \\
  Value value = InvokePrimitive(CallValues::make(opack->opv, _schema));                        \\
  int n_tapes = grads.size();                                                                  \\
  bool full_grads = RemoveNoGrad(prev_tapes.data(), grads.data(), &n_tapes);                   \\
  /* case 1: no grad required */                                                               \\
  if (n_tapes == 0) {                                                                          \\
    *ret = DeTuple(value);                                                                     \\
    return;                                                                                    \\
  }                                                                                            \\
  Expr body = Tuple({grads.begin(), grads.begin() + n_tapes});                                 \\
  std::vector<const ExprNode*> used_vars;                                                      \\
  if (full_grads) {                                                                            \\
    /* case 2: full grad required, use pre-computed results */                                 \\
    used_vars = opack->grad_used_vars;                                                         \\
  } else {                                                                                     \\
    /* case 3: partial grad required, have to collect vars */                                  \\
    CollectVars(body, &used_vars);                                                             \\
  }                                                                                            \\
  const auto *schema = _schema.as<obj>();                                                      \\
  Map<Var, Value> env;

#define MNM_SET_ENV(var, value)                                                        \\
  {                                                                                    \\
    const auto &_v = (var);                                                            \\
    if (std::binary_search(used_vars.begin(), used_vars.end(), _v.operator->())) {     \\
        env.Set(_v, value);                                                            \\
    }                                                                                  \\
  }

#define MNM_RET()                                                                      \\
  DeStruct(std::move(value),                                                           \\
           ClosureValue::make(/*env=*/std::move(env),                                  \\
                              /*func=*/Function({vpack->dy}, body, {}, {})),           \\
           {prev_tapes.begin(), prev_tapes.begin() + n_tapes});
""".strip()

IMPERATIVE_API_EPILOG = """
#undef MNM_RET
#undef MNM_SET_ENV
#undef MNM_PRELUDE

}  // namespace imperative
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_imperative_api(op):
    IMPERATIVE_API = """
MNM_REGISTER_GLOBAL("mnm.op.imp.{OP_NAME}")
.set_body([](TVMArgs args, TVMRetValue* ret) {{
  MNM_PRELUDE({OP_VAR}, {N_ARGS}, ffi2schema::{SCHEMA_NAME}, schema::{SCHEMA_NAME}Args);  // NOLINT(whitespace/line_length)
{ARGS}
  MNM_SET_ENV(vpack->y, value);
  *ret = MNM_RET();
}});
""".strip()
    ARG = (
        " " * 2
        + """
  MNM_SET_ENV(vpack->x[{I}], schema2value::{NORM}(schema->{ARG_NAME}));
""".strip()
    )
    n_args = len(op.schema)
    schema_name = snake_to_pascal(op.schema_name)
    args = []
    for i, entry in enumerate(op.schema):
        norm = NORM_CONVERTER[NORM_MAP[entry.cxx_normalizer or entry.cxx_type]]
        arg_name = entry.name
        args.append(ARG.format(I=i, NORM=norm, ARG_NAME=arg_name))
    args = "\n".join(map(add_no_lint, args))
    return IMPERATIVE_API.format(
        OP_NAME=op.name,
        OP_VAR=op.name.replace(".", "_"),
        SCHEMA_NAME=schema_name,
        N_ARGS=n_args,
        ARGS=args,
    )


#### Part 2.1. FFI to Array<Expr> (for each schema) ####


FFI_TO_EXPR_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace ffi2expr {

#define MNM_PRELUDE(n)                                                                     \\
  const int size = values.size();                                                          \\
  CHECK_EQ(size, n) << "TypeError: Mismatched number of arguments for operator \\"{op}\\": " \\
                    << "Expected " << n << ", but get " << size;                           \\
  std::vector<Expr> result;

#define MNM_ARG(i, norm, name)                \\
  try {                                       \\
    result.push_back(norm(values[i]));        \\
  } catch (const dmlc::Error& e) {            \\
    FillError(e, "{arg}", #name);             \\
  }

#define MNM_RET() return Array<Expr>(result);
""".strip()

FFI_TO_EXPR_EPILOG = """
#undef MNM_RET
#undef MNM_ARG
#undef MNM_PRELUDE

}  // namespace ffi2expr
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_ffi_to_expr(_schema):
    FFI_TO_EXPR = """
Array<Expr> {SCHEMA_NAME}(const TVMArgs& values) {{
  MNM_PRELUDE({N_ARGS});
{ARGS}
  MNM_RET();
}}
""".strip()
    ARG = (
        " " * 2
        + """
MNM_ARG({I}, ffi2expr::{NORM}, {ARG_NAME});
""".strip()
    )
    schema_name, schema = _schema
    schema_name = snake_to_pascal(schema_name)
    n_args = len(schema)
    args = []
    for i, entry in enumerate(schema):
        norm = NORM_CONVERTER[NORM_MAP[entry.cxx_normalizer or entry.cxx_type]]
        arg_name = entry.name
        args.append(ARG.format(I=i, NORM=norm, ARG_NAME=arg_name))
    args = "\n".join(map(add_no_lint, args))
    return FFI_TO_EXPR.format(SCHEMA_NAME=schema_name, N_ARGS=n_args, ARGS=args)


# Part 2.2. Symbolic API, uses "Part 2.1. FFI to Array<Expr>"

SYMBOLIC_API_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace symbolic {

#define MNM_SYMBOLIC_API(op_name, n_args, schema)                               \\
  [](TVMArgs args, TVMRetValue* ret) {                                          \\
    auto *pack = regs::OpPack<names::op_name, n_args>::Get();                   \\
    try {                                                                       \\
        *ret = BindSymbol(Call(pack->op, ffi2expr::schema(args)));              \\
    } catch (const dmlc::Error &e) {                                            \\
        FillError(e, "{op}", names::op_name);                                   \\
    }                                                                           \\
  }
""".strip()

SYMBOLIC_API_EPILOG = """
#undef MNM_SYMBOLIC_API

}  // namespace symbolic
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_symbolic_api(op):
    SYMBOLIC_API = """
MNM_REGISTER_GLOBAL("mnm.op.sym.{OP_NAME}")
.set_body(MNM_SYMBOLIC_API({OP_VAR}, {N_ARGS}, {SCHEMA_NAME}));
""".strip()
    n_args = len(op.schema)
    schema_name = snake_to_pascal(op.schema_name)
    return SYMBOLIC_API.format(
        OP_NAME=op.name, OP_VAR=op.name.replace(".", "_"), N_ARGS=n_args, SCHEMA_NAME=schema_name
    )


# Part 3.1. Array<Value> to schema (for each schema)


VALUE_TO_SCHEMA_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace value2schema {

#define MNM_PRELUDE(lb, ub, schema)                                             \\
  const int size = values.size();                                               \\
  CHECK(size >= lb) << "TypeError: Too few arguments for operator \\"{op}\\". "   \\
                    << "Expected at least " << lb << ", but get " << size;      \\
  CHECK(size <= ub) << "TypeError: Too many arguments for operator \\"{op}\\". "  \\
                    << "Expected at most " << ub << ", but get " << size;       \\
  auto attrs = make_object<schema>();

#define MNM_REQUIRED(i, norm, name)       \\
  try {                                   \\
    attrs->name = norm(values[i]);        \\
  } catch (const dmlc::Error& e) {        \\
    try {                                 \\
      FillError(e, "{arg}", #name);       \\
    } catch (const dmlc::Error &ee) {     \\
      FillError(ee, "{op}", op_name);     \\
    }                                     \\
  }

#define MNM_OPTIONAL(i, norm, name)       \\
  if (size > i) {                         \\
    try {                                 \\
      attrs->name = norm(values[i]);      \\
    } catch (const dmlc::Error& e) {      \\
      try {                               \\
        FillError(e, "{arg}", #name);     \\
      } catch (const dmlc::Error &ee) {   \\
        FillError(ee, "{op}", op_name);   \\
      }                                   \\
    }                                     \\
  }
""".strip()

VALUE_TO_SCHEMA_EPILOG = """
#undef MNM_OPTIONAL
#undef MNM_REQUIRED
#undef MNM_PRELUDE

}  // namespace value2schema
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_value_to_schema(_schema):
    VALUE_TO_SCHEMA = """
template <const char* op_name>
Attrs {SCHEMA_NAME}(const Array<Value>& values) {{
  MNM_PRELUDE({N_ARG_LB}, {N_ARG_UB}, schema::{SCHEMA_NAME}Args);
{ARGS}
  return Attrs(attrs);
}}
""".strip()
    ARG = (
        " " * 2
        + """
MNM_{OPTION}({I}, value2schema::{NORM}, {ARG_NAME});
""".strip()
    )
    schema_name, schema = _schema
    schema_name = snake_to_pascal(schema_name)
    n_arg_lb = sum(int(entry.cxx_default is None) for entry in schema)
    n_arg_ub = len(schema)
    args = []
    for i, entry in enumerate(schema):
        option = "REQUIRED" if entry.cxx_default is None else "OPTIONAL"
        norm = NORM_CONVERTER[NORM_MAP[entry.cxx_normalizer or entry.cxx_type]]
        arg_name = entry.name
        args.append(ARG.format(I=i, NORM=norm, ARG_NAME=arg_name, OPTION=option))
    args = "\n".join(map(add_no_lint, args))
    return VALUE_TO_SCHEMA.format(
        SCHEMA_NAME=schema_name, ARGS=args, N_ARG_LB=n_arg_lb, N_ARG_UB=n_arg_ub
    )


# Part 3.2. Schema field index (for each schema)


SCHEMA_FIELD_IDX_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace schema_field_idx {
""".strip()

SCHEMA_FIELD_IDX_EPILOG = """
}  // namespace schema_field_idx
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_schema_field_idx(_schema):
    VALUE_TO_SCHEMA = """
template <const char* op_name>
int {SCHEMA_NAME}(const std::string& field) {{
{ARGS}
  LOG(WARNING) << "Cannot find " << field << " in the schema of op " << op_name;
  return -1;
}}
""".strip()
    ARG = """
  if (field == "{FIELD}") {{
    return {I};
  }}
""".strip()
    schema_name, schema = _schema
    schema_name = snake_to_pascal(schema_name)
    args = []
    for i, entry in enumerate(schema):
        args.append(ARG.format(I=i, FIELD=entry.name))
    args = "\n".join(map(add_no_lint, args))
    return VALUE_TO_SCHEMA.format(SCHEMA_NAME=schema_name, ARGS=args)


# Part 3.3. FMNMSchema API, uses Part 3.1 and Part 3.2

F_MNM_SCHEMA_PRELUDE = """
namespace mnm {
namespace op {
namespace regs {
namespace f_mnm_schema {

#define MNM_BIND_SCHEMA(op_str, op_name, schema) \\
  MNM_REGISTER_OP(op_str).set_attr<FMNMSchema>("FMNMSchema", schema<op_name>);

#define MNM_BIND_SCHEMA_FIELD_INDEX(op_str, op_name, schema) \\
  MNM_REGISTER_OP(op_str).set_attr<FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex", schema<op_name>);
""".strip()

F_MNM_SCHEMA_EPILOG = """
#undef MNM_BIND_SCHEMA
#undef MNM_BIND_SCHEMA_FIELD_INDEX

}  // namespace f_mnm_schema
}  // namespace regs
}  // namespace op
}  // namespace mnm
""".strip()


def gen_f_mnm_schema(op):
    FMNMSchema = """
MNM_BIND_SCHEMA("mnm.op.{OP_NAME}", names::{OP_VAR}, value2schema::{SCHEMA_NAME});  // NOLINT(whitespace/line_length)
MNM_BIND_SCHEMA_FIELD_INDEX("mnm.op.{OP_NAME}", names::{OP_VAR}, schema_field_idx::{SCHEMA_NAME});  // NOLINT(whitespace/line_length)
""".strip()
    schema_name = snake_to_pascal(op.schema_name)
    return FMNMSchema.format(
        OP_NAME=op.name, OP_VAR=op.name.replace(".", "_"), SCHEMA_NAME=schema_name
    )


def main(path="./src/op/regs/regs.cc"):
    result = gen_file(path)
    write_to_file(path, result)


if __name__ == "__main__":
    main()
