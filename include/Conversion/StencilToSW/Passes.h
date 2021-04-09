/**
 * @file Passes.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief Stencil to SW Pass
 * @version 0.1
 * @date 2021-03-22
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */
#ifndef _CONVERSION_STENCILTOSW_PASSES_H_
#define _CONVERSION_STENCILTOSW_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertStencilToSWPass();
// std::unique_ptr<Pass> createSWOutliningPass();

//============================================================================//
// 注册
//============================================================================//
#define GEN_PASS_REGISTRATION
#include "Conversion/StencilToSW/Passes.h.inc"


} // end of mlir

#endif // _CONVERSION_STENCILTOSW_PASSES_H_