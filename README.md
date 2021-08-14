# SWStenDSL

A stencil domain specific language supporting Sunway TaihuLight.

## Build Instructions

SWStenDSL depends on a build of llvm  including MLIR. So you need to build and install llvm first. The repository is developed based on LLVM project commit e2dee9af8db.

### Requirement
| Dependency | Version |
| :----: | :----: |
| cmake | 3.19.7 |
| ninja | 1.10.2 |
| gcc | 10.2.0 |
| python | 3.9.6 |

we also have tested on `Ubuntu 20.04.1 LTS`

### Build LLVM project

We assume that you build LLVM and MLIR in `build` directory and install them to `<INSTALL_DIR>`. 

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout e2dee9af8db
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=ON #-DLLVM_PARALLEL_LINK_JOBS=2
cmake --build . --target install
```

### Build SWStenDSL project

```bash
git clone https://github.com/JackMoriarty/SWStenDSL.git
cd SWStenDSL
export LLVM_INSTALL=<path_to_llvm_install>
export LLVM_BUILD=<path_to_llvm_build_directory>
make build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_INSTALL/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit
cmake --build . --target stenCC
```
The execuatable binary will be generated in path `SWSten/build/bin`, named `stenCC`.
### Build SWStenDSL parallel communication library
```bash
cd SWStenDSL/utils
make
```
The library will generated in current directory `SWsten/utils`, named `libswstenmpi.a`
