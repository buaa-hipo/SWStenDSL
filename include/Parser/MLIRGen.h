/**
 * @file MLIRGen.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief MLIR dump 头文件
 * @version 0.1
 * @date 2021-06-12
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTENDSL_MLIRGEN_H_
#define _SWSTENDSL_MLIRGEN_H_

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // End of namespace mlir

namespace swsten {
class ModuleAST;

// 根据给定的ModuleAST生成MLIR module, 如果失败则返回nullptr
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
};

#endif /* End of _SWSTENDSL_MLIRGEN_H_ */