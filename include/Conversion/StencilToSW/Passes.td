#ifndef _CONVERSION_STENCILTOSW_PASS_TD_
#define _CONVERSION_STENCILTOSW_PASS_TD_

include "mlir/Pass/PassBase.td"

def StencilToSWPass : Pass<"convert-stencil-to-sw", "ModuleOp"> {
    let summary = "Convert stencil dialect to sw dialect";
    let constructor = "mlir::createConvertStencilToSWPass()";
}

def SWOutliningPass : Pass<"sw-outlining", "ModuleOp"> {
    let summary = "Outlining the launchOp";
    let constructor = "mlir::createSWOutliningPass()";
}

#endif // _CONVERSION_STENCILTOSW_PASS_TD_