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
class GetSliceRangeArgs : public ir::AttrsNode<GetSliceRangeArgs> {
 public:
  value::BaseTensorValue x;
  sharding::ShardSpec shard_spec;
  MNM_OP_SCHEMA(GetSliceRangeArgs, "mnm.args.get_slice_range");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
