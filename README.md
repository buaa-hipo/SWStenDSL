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
mkdir build && cd build
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

## How To Use

### simple run

After building all project, You can run the example program provided by SWStenDSL. There is a sample command.

```bash
cd SWStenDSL/build
./bin/stenCC ../test/Examples/StencilDialect/laplaceAddInputIteration/laplaceAddInputIteration.dsl --emit=sw > laplaceAddInputIteration.sw 2>&1
python3 ../tool/translate.py laplaceAddInputIteration.sw
```

After running those command, three C source file will be generated. We also provide driver program and build script in `${SWStenProject}/test/Examples/StencilDialect/laplaceAddInputIteration/`. 

You need create a directory on Sunway Taihulight supercomputer, then copy three source file, driver program, buildscript and the `${SWStenDSL}/utils` into the directory you created.

```bash
cd <path_to_directory_you_created>
make
make run
```

Then, the executable program will be generated and be submited to Sunway TaihuLight supercomputer. 

### MPI large-scale computing support

If you need run large-scale stencil computing,  you need modify the dsl file and build script.

In dsl file, you need add `mpiTile` and `mpiHalo` keyword. For more information please reference `${SWStenDSL}/doc/StencilDSL.md`.

In build script, you need declare a marco `SWStenMPI`, the build script in `${SWStenDSL}/examples/2d9pt_box/makefileMPI` is an example.

### enable optimization option

Now, we provide two optimization options: `--kernel-fusion`, `--enable-vector=<vector-width>`.

`kernel-fusion` will merge the kernels if one kernel depends on another kernel result. This option will  reduce the number of spe launching, and reduce the number of data transport bewteen main memory and LDM.

`enable-vector`will enable vectorizable computation on spe. 

There is a sample command.

```
./bin/stenCC --kernel-fusion --enable-vector=4 ../test/Examples/StencilDialect/laplaceAddInputIteration/laplaceAddInputIteration.dsl --emit=sw > laplaceAddInputIteration.sw 2>&1
```

