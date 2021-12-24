#ifndef _DIALECT_STENCIL_PASS_TD_
#define _DIALECT_STENCIL_PASS_TD_

include "mlir/Pass/PassBase.td"

def StencilKernelFusionPass : Pass<"kernel-fusion", "ModuleOp"> {
    let summary = "kernel fusion";
    let constructor = "mlir::createStencilKernelFusionPass()";
}

#endif // _DIALECT_STENCIL_PASS_TD_