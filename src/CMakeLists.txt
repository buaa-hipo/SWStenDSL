get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRTestTransforms

    Stencil
    SW
    StencilToSW
    StencilToVector
    CodeGen
)

add_llvm_executable(stenCC stenCC.cpp)

llvm_update_compile_flags(stenCC)
target_link_libraries(stenCC PRIVATE ${LIBS})
