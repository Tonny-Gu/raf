/*!
 * Auto generated. Do not touch.
 * \file src/op/schema/sharding.h
 * \brief Operator schema.
 */
#pragma once
#include <vector>
#include <string>
#include "raf/op.h"
#include "raf/value.h"
namespace raf {
namespace op {
namespace schema {
class ShardUnaryArgs : public ir::AttrsNode<ShardUnaryArgs> {
 public:
  value::BaseTensorValue x;
  value::Value spec;
  RAF_OP_SCHEMA(ShardUnaryArgs, "raf.args.shard_unary");
};
}  // namespace schema
}  // namespace op
}  // namespace raf
