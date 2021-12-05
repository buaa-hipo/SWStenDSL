/**
 * @file StencilTypes.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief Stencil方言中类型的实现
 * @version 0.1
 * @date 2021-02-23
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/IR/Types.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <cstdint>

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"

using namespace mlir;
using namespace stencil;

namespace mlir {
namespace stencil {
namespace StencilTypeStorage {

//============================================================================//
// stencil方言变量存储类型
//============================================================================//
// GridTypeStorage 存储类
} // end of namespace StencilTypeStorage
} // end of namespace stencil
} // end of namespace mlir

//============================================================================//
// stencil变量类型
//============================================================================//
//===----------------------------------------------------------------------===//
/*******************************GridType***************************************/
//===----------------------------------------------------------------------===//
bool GridType::classof(Type type) { return type.isa<FieldType>(); }
// 获取元素类型
Type GridType::getElementType() const {
    return static_cast<ImplType *>(impl)->getElementType();
}
// 获取结构化网格的形状(各个维度的大小)
ArrayRef<int64_t> GridType::getShape() const {
    return static_cast<ImplType *>(impl)->getShape();
}
//===----------------------------------------------------------------------===//
/******************************FieldType***************************************/
//===----------------------------------------------------------------------===//
FieldType FieldType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
    return Base::get(elementType.getContext(), elementType, shape);
}
//===----------------------------------------------------------------------===//
/******************************ResultType**************************************/
//===----------------------------------------------------------------------===//
ResultType ResultType::get(Type resultType) {
    return Base::get(resultType.getContext(), resultType);
}

Type ResultType::getResultType() const { return getImpl()->getResultType(); }