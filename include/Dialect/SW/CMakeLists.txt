add_mlir_dialect(SWOps sw)

set(LLVM_TARGET_DEFINITIONS SWOps.td)
mlir_tablegen(SWOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SWOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSWOpsEnumsIncGen)