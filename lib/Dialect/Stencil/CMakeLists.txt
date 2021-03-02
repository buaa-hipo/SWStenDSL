add_mlir_dialect_library(Stencil
    StencilDialect.cpp
    StencilOps.cpp
    StencilTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Stencil
    
    DEPENDS
    MLIRStencilOpsIncGen
    MLIRStencilInterfacesIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)