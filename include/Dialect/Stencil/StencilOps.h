/**
 * @file StencilOps.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief Stencil 方言中操作头文件, 该文件为TableGen生成的头文件封装
 * @version 0.1
 * @date 2021-02-25
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _DIALECT_STENCIL_STENCIL_OPS_H_
#define _DIALECT_STENCIL_STENCIL_OPS_H_

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LogicalResult.h>
#include <cstdint>
#include <numeric>

#include "Dialect/Stencil/StencilTypes.h"

namespace mlir {
namespace stencil {

#include "Dialect/Stencil/StencilInterfaces.h.inc"
#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.h.inc"

// apply 操作正则化
struct ApplyOpPattern : public OpRewritePattern<stencil::ApplyOp> {
    ApplyOpPattern(MLIRContext *context, PatternBenefit benefit = 1);

    // 清理apply操作中的冗余参数
    stencil::ApplyOp cleanupOpArguments(stencil::ApplyOp apply,
                                        PatternRewriter &rewriter) const;
};
} // end of namespace stencil
} // end of namespace mlir

#endif // end of _DIALECT_STENCIL_STENCIL_OPS_H_