/**
 * @file SWTypes.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-03-03
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/IR/Types.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <cstdint>

#include "Dialect/SW/SWDialect.h"
#include "Dialect/SW/SWTypes.h"

using namespace mlir;
using namespace sw;

namespace mlir {
namespace sw {
namespace SWTypeStorage {

//============================================================================//
// sw方言变量存储类型
//============================================================================//
// GridTypeStorage 存储类
} // end of namespace SWTypeStorage

//============================================================================//
// sw变量类型
//============================================================================//
bool GridType::classof(Type type) { return type.isa<MemRefType>(); }
// 获取元素类型
Type GridType::getElementType() const {
    return static_cast<ImplType *>(impl)->getElementType();
}

// 获取数组的形状(各个维度的大小)
ArrayRef<int64_t> GridType::getShape() const {
    return static_cast<ImplType *>(impl)->getShape();
}

MemRefType MemRefType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
    return Base::get(elementType.getContext(), elementType, shape);
}

} // end of namespace sw
} // end of namespace mlir
