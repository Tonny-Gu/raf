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
 * \file src/op/dialect/cutlass/timer.h
 * \brief Timer for cutlass kernel
 */
#pragma once

#include "./cutlass_utils.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace mnm {
namespace op {
namespace cutlass {

/*! \brief Module with a single PackedFunc */
class CutlassModuleNode : public tvm::runtime::ModuleNode {
 public:
  explicit CutlassModuleNode(registry::PackedFunc pf) : pf_(pf) {
  }

  virtual const char* type_key() const final {
    return "CutlassModule";
  }

  virtual registry::PackedFunc GetFunction(const std::string& name,
                                           const ir::ObjectPtr<ir::Object>& sptr_to_self) final {
    return pf_;
  }

 private:
  /*! \brief The packed function contained by this module */
  registry::PackedFunc pf_;
};

/*! \brief Wraps a packed function with CutlassModule
 *  \param pf The packed function to be wrapped
 *  \return The CutlassModule
 */
tvm::runtime::Module MakeCutlassModule(registry::PackedFunc pf);

/*! \brief Evaluates the running time of a packed function
 *  \param pf The packed function to be evaluated
 *  \param dev The execution device
 *  \param number The number of times to run this function for taking average.
 *                We call these runs as one `repeat` of measurement.
 *  \param repeat The number of times to repeat the measurement.
 *                In total, the function will be invoked (1 + number x repeat) times,
 *                where the first one is warm up and will be discarded.
 *                The returned result contains `repeat` costs,
 *                each of which is an average of `number` costs.
 *  \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
 *                       By default, one `repeat` contains `number` runs. If this parameter is set,
 *                       the parameters `number` will be dynamically adjusted to meet the
 *                       minimum duration requirement of one `repeat`.
 *                       i.e., When the run time of one `repeat` falls below this time, the `number`
 *                       parameter will be automatically increased.
 *  \return The function that takes same argument as func
 *          and returns Array<FloatValue>, which reports `repeat` time costs in seconds.
 */
registry::PackedFunc TimeEvaluator(registry::PackedFunc pf, Device dev, int number = 10,
                                   int repeat = 1, int min_repeat_ms = 0);

}  // namespace cutlass
}  // namespace op
}  // namespace mnm
