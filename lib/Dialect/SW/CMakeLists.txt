add_mlir_dialect_library(SW
    SWDialect.cpp
    SWOps.cpp
    SWTypes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/SW
    
    DEPENDS
    MLIRSWOpsEnumsIncGen
    MLIRSWOpsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)