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

# pylint: disable=invalid-name,redefined-builtin,line-too-long
# pylint: disable=missing-class-docstring,missing-function-docstring
"""Auto generated. Do not touch."""
from mnm._lib import _APIS

# Defined in ./src/pass/stream_schedule_asap.cc
ASAPStreamSchedule = _APIS.get("mnm.pass_.ASAPStreamSchedule", None)
# Defined in ./src/pass/annotate_collective_ops.cc
AnnotateCollectiveOps = _APIS.get("mnm.pass_.AnnotateCollectiveOps", None)
# Defined in ./src/pass/annotate_target.cc
AnnotateTarget = _APIS.get("mnm.pass_.AnnotateTarget", None)
# Defined in ./src/pass/assign_device.cc
AssignDevice = _APIS.get("mnm.pass_.AssignDevice", None)
# Defined in ./src/pass/auto_cast.cc
AutoCast = _APIS.get("mnm.pass_.AutoCast", None)
# Defined in ./src/pass/data_parallel.cc
AutoDataParallel = _APIS.get("mnm.pass_.AutoDataParallel", None)
# Defined in ./src/pass/gradient.cc
AutoDiff = _APIS.get("mnm.pass_.AutoDiff", None)
# Defined in ./src/pass/fold_const.cc
BindParam = _APIS.get("mnm.pass_.BindParam", None)
# Defined in ./src/pass/canonicalize_ops.cc
CanonicalizeOps = _APIS.get("mnm.pass_.CanonicalizeOps", None)
# Defined in ./src/pass/context_analysis.cc
ContextAnalysis = _APIS.get("mnm.pass_.ContextAnalysis", None)
# Defined in ./src/pass/data_parallel_schedule.cc
DataParallelSchedule = _APIS.get("mnm.pass_.DataParallelSchedule", None)
# Defined in ./src/pass/dead_code.cc
DeadCodeElimination = _APIS.get("mnm.pass_.DeadCodeElimination", None)
# Defined in ./src/pass/deduplicate.cc
Deduplicate = _APIS.get("mnm.pass_.Deduplicate", None)
# Defined in ./src/pass/dispatch_dialect.cc
DispatchDialect = _APIS.get("mnm.pass_.DispatchDialect", None)
# Defined in ./src/pass/enforce_sync.cc
EnforceSync = _APIS.get("mnm.pass_.EnforceSync", None)
# Defined in ./src/pass/type_erase.cc
EraseType = _APIS.get("mnm.pass_.EraseType", None)
# Defined in ./src/pass/estimate_flops.cc
EstimateGFLOPS = _APIS.get("mnm.pass_.EstimateGFLOPS", None)
# Defined in ./src/pass/sharding.cc
ExpandShardOpCall = _APIS.get("mnm.pass_.ExpandShardOpCall", None)
# Defined in ./src/pass/substitute.cc
ExprAppend = _APIS.get("mnm.pass_.ExprAppend", None)
# Defined in ./src/pass/extract_binding.cc
ExtractBinding = _APIS.get("mnm.pass_.ExtractBinding", None)
# Defined in ./src/pass/flatten_closure.cc
FlattenClosure = _APIS.get("mnm.pass_.FlattenClosure", None)
# Defined in ./src/pass/fold_const.cc
FoldConstant = _APIS.get("mnm.pass_.FoldConstant", None)
# Defined in ./src/pass/from_relay.cc
FromRelay = _APIS.get("mnm.pass_.FromRelay", None)
# Defined in ./src/pass/fuse_dialect.cc
FuseDialect = _APIS.get("mnm.pass_.FuseDialect", None)
# Defined in ./src/pass/fuse_tvm.cc
FuseTVM = _APIS.get("mnm.pass_.FuseTVM", None)
# Defined in ./src/pass/grad_arg_select.cc
GradientInputSelection = _APIS.get("mnm.pass_.GradientInputSelection", None)
# Defined in ./src/pass/stream_schedule_ios.cc
IOSStreamSchedule = _APIS.get("mnm.pass_.IOSStreamSchedule", None)
# Defined in ./src/pass/type_infer.cc
InferType = _APIS.get("mnm.pass_.InferType", None)
# Defined in ./src/pass/inline_backward.cc
InlineBackward = _APIS.get("mnm.pass_.InlineBackward", None)
# Defined in ./src/pass/inline_closure.cc
InlineClosure = _APIS.get("mnm.pass_.InlineClosure", None)
# Defined in ./src/pass/inline_let.cc
InlineLet = _APIS.get("mnm.pass_.InlineLet", None)
# Defined in ./src/pass/inline_primitives.cc
InlinePrimitives = _APIS.get("mnm.pass_.InlinePrimitives", None)
# Defined in ./src/pass/inplace_update.cc
InplaceUpdate = _APIS.get("mnm.pass_.InplaceUpdate", None)
# Defined in ./src/pass/lambda_lift.cc
LambdaLift = _APIS.get("mnm.pass_.LambdaLift", None)
# Defined in ./src/pass/lift_branch_body.cc
LiftBranchBody = _APIS.get("mnm.pass_.LiftBranchBody", None)
# Defined in ./src/pass/liveness_analysis.cc
LivenessAnalysis = _APIS.get("mnm.pass_.LivenessAnalysis", None)
# Defined in ./src/pass/pass_manager.cc
MNMSequential = _APIS.get("mnm.pass_.MNMSequential", None)
# Defined in ./src/pass/manifest_alloc.cc
ManifestAlloc = _APIS.get("mnm.pass_.ManifestAlloc", None)
# Defined in ./src/pass/memory_plan.cc
MemoryPlan = _APIS.get("mnm.pass_.MemoryPlan", None)
# Defined in ./src/pass/memory_schedule.cc
MemorySchedule = _APIS.get("mnm.pass_.MemorySchedule", None)
# Defined in ./src/pass/merge_compiler_regions.cc
MergeCompilerRegions = _APIS.get("mnm.pass_.MergeCompilerRegions", None)
# Defined in ./src/pass/anf_partition.cc
PartitionANF = _APIS.get("mnm.pass_.PartitionANF", None)
# Defined in ./src/pass/partition_gradient.cc
PartitionGradient = _APIS.get("mnm.pass_.PartitionGradient", None)
# Defined in ./src/pass/partition_graph.cc
PartitionGraph = _APIS.get("mnm.pass_.PartitionGraph", None)
# Defined in ./src/pass/print_ir.cc
PrintIR = _APIS.get("mnm.pass_.PrintIR", None)
# Defined in ./src/pass/rematerialization.cc
Rematerialization = _APIS.get("mnm.pass_.Rematerialization", None)
# Defined in ./src/pass/rename_vars.cc
RenameVars = _APIS.get("mnm.pass_.RenameVars", None)
# Defined in ./src/pass/sharding.cc
SetShardOpCallAttrs = _APIS.get("mnm.pass_.SetShardOpCallAttrs", None)
# Defined in ./src/pass/simplify_expr.cc
SimplifyExpr = _APIS.get("mnm.pass_.SimplifyExpr", None)
# Defined in ./src/pass/substitute.cc
Substitute = _APIS.get("mnm.pass_.Substitute", None)
# Defined in ./src/pass/to_a_normal_form.cc
ToANormalForm = _APIS.get("mnm.pass_.ToANormalForm", None)
# Defined in ./src/pass/to_basic_block_normal_form.cc
ToBasicBlockNormalForm = _APIS.get("mnm.pass_.ToBasicBlockNormalForm", None)
# Defined in ./src/pass/to_graph_normal_form.cc
ToGraphNormalForm = _APIS.get("mnm.pass_.ToGraphNormalForm", None)
# Defined in ./src/pass/inplace_update.cc
ValidateInplaceUpdate = _APIS.get("mnm.pass_.ValidateInplaceUpdate", None)
# Defined in ./src/pass/stream_schedule_wavefront.cc
WavefrontStreamSchedule = _APIS.get("mnm.pass_.WavefrontStreamSchedule", None)
# Defined in ./src/pass/dataflow_matcher.cc
dataflow_pattern_match = _APIS.get("mnm.pass_.dataflow_pattern_match", None)
# Defined in ./src/pass/dataflow_matcher.cc
dataflow_pattern_partition = _APIS.get("mnm.pass_.dataflow_pattern_partition", None)
# Defined in ./src/pass/dataflow_matcher.cc
dataflow_pattern_rewrite = _APIS.get("mnm.pass_.dataflow_pattern_rewrite", None)
# Defined in ./src/pass/fold_const.cc
is_constant = _APIS.get("mnm.pass_.is_constant", None)
# Defined in ./src/pass/from_relay.cc
validate_relay_param_name = _APIS.get("mnm.pass_.validate_relay_param_name", None)
