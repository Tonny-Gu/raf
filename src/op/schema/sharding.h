/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/sharding.h
 * \brief Operator schema.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/value.h"
namespace mnm {
namespace op {
namespace schema {
class ShardUnaryArgs : public ir::AttrsNode<ShardUnaryArgs> {
 public:
  value::BaseTensorValue x;
  value::Value spec;
  MNM_OP_SCHEMA(ShardUnaryArgs, "mnm.args.shard_unary");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
