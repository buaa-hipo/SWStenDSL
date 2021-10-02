#ifndef _CONVERSION_STENCILTOVECTOR_PASS_TD_
#define _CONVERSION_STENCILTOVECTOR_PASS_TD_

include "mlir/Pass/PassBase.td"

def StencilToVectorPass : Pass<"convert-stencil-to-vector", "ModuleOp"> {
    let summary = "Convert stencil dialect to vector dialect";
    let constructor = "mlir::createConvertStencilToVectorPass()";
    let options = [
        Option<"VectorWidth", "vector-width", "unsigned", /*default=*/"2",
                "Width of vector register">,
    ];
}

#endif // _CONVERSION_STENCILTOVECTOR_PASS_TD_