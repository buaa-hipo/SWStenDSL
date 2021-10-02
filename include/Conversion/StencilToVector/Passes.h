/**
 * @file Passes.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief Stencil to Vector Pass
 * @version 0.1
 * @date 2021-09-25
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _CONVERSION_STENCILTOVECTOR_PASSES_H_
#define _CONVERSION_STENCILTOVECTOR_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<Pass> createConvertStencilToVectorPass(unsigned int vectorWidth=2);

//============================================================================//
// 注册
//============================================================================//
#define GEN_PASS_REGISTRATION
#include "Conversion/StencilToVector/Passes.h.inc"

} // end of namespace mlir

#endif // _CONVERSION_STENCILTOVECTOR_PASSES_H_