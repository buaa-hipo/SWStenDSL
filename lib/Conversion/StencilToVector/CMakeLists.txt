add_mlir_dialect_library(StencilToVector
    ConvertStencilToVector.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Conversion/StencilToVector

    DEPENDS
    MLIRStencilToVectorPassIncGen
)

target_link_libraries(StencilToVector PUBLIC MLIRIR)
