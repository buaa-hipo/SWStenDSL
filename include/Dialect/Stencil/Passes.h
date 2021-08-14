/**
 * @file Passes.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief Pass for stencil dialect
 * @version 0.1
 * @date 2021-08-14
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */
#ifndef _DIALECT_STENCIL_PASSES_H_
#define _DIALECT_STENCIL_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<OperationPass<FuncOp>> createStencilKernelFusionPass();

//===----------------------------------------------------------------------===//
// 注册
//===----------------------------------------------------------------------===//
#define GEN_PASS_REGISTRATION
#include "Dialect/Stencil/Passes.h.inc"

} // end of namespace mlir

#endif // _DIALECT_STENCIL_PASSES_H_