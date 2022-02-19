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
 * \file src/impl/build_info.cc
 * \brief Reflect build-time information and expose to the frontend
 */
#include "mnm/registry.h"
#ifdef MNM_USE_NCCL
#include <nccl.h>
#endif

namespace mnm {
namespace build_info {

std::string GitVersion() {
  return MNM_GIT_VERSION;
}

bool UseCUDA() {
#ifdef MNM_USE_CUDA
  return true;
#else
  return false;
#endif
}

std::string UseCuBLAS() {
  return MNM_USE_CUBLAS;
}

std::string UseCuDNN() {
  return MNM_USE_CUDNN;
}

std::string UseLLVM() {
  return MNM_USE_LLVM;
}

bool UseMPI() {
#ifdef MNM_USE_MPI
  return true;
#else
  return false;
#endif
}

bool UseNCCL() {
#ifdef MNM_USE_NCCL
  return true;
#else
  return false;
#endif
}

int NCCLVersion() {
#ifdef MNM_USE_NCCL
  return NCCL_VERSION_CODE;
#else
  return 0;
#endif
}
std::string UseCUTLASS() {
  return MNM_USE_CUTLASS;
}

std::string CudaVersion() {
  return MNM_CUDA_VERSION;
}

std::string CudnnVersion() {
  return MNM_CUDNN_VERSION;
}

std::string CmakeBuildType() {
  return MNM_CMAKE_BUILD_TYPE;
}

MNM_REGISTER_GLOBAL("mnm.build_info.git_version").set_body_typed(GitVersion);
MNM_REGISTER_GLOBAL("mnm.build_info.cuda_version").set_body_typed(CudaVersion);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cuda").set_body_typed(UseCUDA);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cublas").set_body_typed(UseCuBLAS);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cudnn").set_body_typed(UseCuDNN);
MNM_REGISTER_GLOBAL("mnm.build_info.cudnn_version").set_body_typed(CudnnVersion);
MNM_REGISTER_GLOBAL("mnm.build_info.cmake_build_type").set_body_typed(CmakeBuildType);
MNM_REGISTER_GLOBAL("mnm.build_info.use_llvm").set_body_typed(UseLLVM);
MNM_REGISTER_GLOBAL("mnm.build_info.use_mpi").set_body_typed(UseMPI);
MNM_REGISTER_GLOBAL("mnm.build_info.use_nccl").set_body_typed(UseNCCL);
MNM_REGISTER_GLOBAL("mnm.build_info.use_cutlass").set_body_typed(UseCUTLASS);
MNM_REGISTER_GLOBAL("mnm.build_info.nccl_version").set_body_typed(NCCLVersion);
}  // namespace build_info
}  // namespace mnm
