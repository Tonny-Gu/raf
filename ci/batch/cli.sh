#!/usr/bin/env bash
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

# The CLI scripts for running CI tasks. This is supposed to be used by
# automated CI so it assumes
#   1. The script is run in the Meta folder.
#   2. The repo has been wekll-configured.
# Example usages:
#   bash batch-cli.sh config_cmake GPU
#   bash batch-cli.sh compile ./build
#   bash batch-cli.sh unit_test multi-GPU
set -e
set -o pipefail

# Generate build/config.cmake given the platform.
function config_cmake() {
    PLATFORM=$1 # CPU or GPU

    rm -rf build
    mkdir -p build
    cd build
    cp ../cmake/config.cmake .
    echo "set(MNM_USE_LLVM llvm-config-8)" >> config.cmake
    echo "set(MNM_USE_GTEST ON)" >> config.cmake
    echo "set(CMAKE_BUILD_TYPE Release)" >> config.cmake

    if [[ $PLATFORM == "CPU" ]]; then
        echo "set(MNM_USE_CUDA OFF)" >> config.cmake
        echo "set(MNM_USE_CUDNN OFF)" >> config.cmake
        echo "set(MNM_USE_CUBLAS OFF)" >> config.cmake
    elif [[ $PLATFORM == "GPU" ]]; then
        CUDA_ARCH=$2 # 75 (T4), 70 (V100), etc
        echo "set(MNM_USE_CUDA ON)" >> config.cmake
        echo "set(MNM_CUDA_ARCH $CUDA_ARCH)" >> config.cmake
        echo "set(MNM_USE_CUDNN ON)" >> config.cmake
        echo "set(MNM_USE_CUBLAS ON)" >> config.cmake
        echo "set(MNM_USE_MPI ON)" >> config.cmake
        echo "set(MNM_USE_NCCL ON)" >> config.cmake
        echo "set(MNM_USE_CUTLASS ON)" >> config.cmake
    else
        echo "Unrecognized platform: $PLATFORM"
        return 1
    fi
    echo "[CLI] config.cmake is generated for $PLATFORM"
    return 0
}

# Compile for the given path.
function compile() {
    BUILD_DIR=$1
    PLATFORM=$2
    JOB_TAG=$3

    # Load ccache if available.
    bash ./ci/batch/backup-ccache.sh download $PLATFORM $JOB_TAG || true

    # Compile. Note that compilation errors will not result in crash in this function.
    # We use return exit code to let the caller decide the action.
    bash ./ci/task_clean.sh $BUILD_DIR
    bash ./ci/task_build.sh $BUILD_DIR -j$(($(nproc) - 1)) || true
    RET=$?
    echo "[CLI] Compiled at $BUILD_DIR"

    # Backup the ccache.
    bash ./ci/batch/backup-ccache.sh upload $PLATFORM $JOB_TAG || true
    return $RET
}

# Run unit tests.
function unit_test() {
    MODE=$1

    if [[ $MODE == "CPU" ]]; then
        bash ./ci/task_cpp_unittest.sh
        bash ./ci/task_python_unittest.sh
    elif [[ $MODE == "GPU" ]]; then
        nvidia-smi -L
        bash ./ci/task_cpp_unittest.sh
        bash ./ci/task_python_unittest.sh
    elif [[ $MODE == "multi-GPU" ]]; then
        nvidia-smi -L
        bash ./ci/task_python_distributed.sh
    else
        echo "Unrecognized mode: $MODE"
        return 1
    fi
    echo "[CLI] Unit tests are done"
    return 0
}

# Run the function from command line.
if declare -f "$1" > /dev/null
then
    # Call arguments verbatim if the function exists
    "$@"
else
    # Show a helpful error
    echo "'$1' is not a known function name" >&2
    exit 1
fi
