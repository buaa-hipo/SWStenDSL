add_mlir_dialect_library(StencilToSW
    ConvertStencilToSW.cpp
    SWOutlining.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Conversion/StencilToSW

    DEPENDS
    MLIRStencilToSWPassIncGen
)

target_link_libraries(StencilToSW PUBLIC MLIRIR)
