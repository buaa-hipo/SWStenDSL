/**
 * @file SWOps.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief SW方言中操作头文件, 为TableGen生成的头文件封装
 * @version 0.1
 * @date 2021-03-06
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _DIALECT_SW_SW_OPS_H_
#define _DIALECT_SW_SW_OPS_H_

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

#include "Dialect/SW/SWTypes.h"

using namespace mlir;
static void buildIfOpTerminatedBody(OpBuilder &builder, Location loc);

#include "Dialect/SW/SWOpsEnums.h.inc"
namespace mlir {
namespace sw {

#define GET_OP_CLASSES
#include "Dialect/SW/SWOps.h.inc"

} // end of namespace sw
} // end of namespace mlir

#endif // end of _DIALECT_SW_SW_OPS_H_