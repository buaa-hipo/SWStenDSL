add_mlir_dialect_library(CodeGen
    AST.cpp
    MLIRGen.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Parser
    
    LINK_LIBS PUBLIC
)