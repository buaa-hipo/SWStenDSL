#!/bin/bash -e

export LLVM_INSTALL=../llvm-project/install
export LLVM_BUILD=../llvm-project/build
find . ! \( -name rebuild.sh -or -name '.' \) -exec rm -rf {} \;

# Debug Version
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug .. -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir
# Release Version
# cmake -G Ninja .. -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir
cmake --build . --target stenCC
