add_mlir_dialect_library(Stencil
    StencilDialect.cpp
    StencilOps.cpp
    StencilTypes.cpp
    StencilKernelFusion.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Stencil
    
    DEPENDS
    MLIRStencilOpsIncGen
    MLIRStencilInterfacesIncGen
    MLIRStencilPassIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)