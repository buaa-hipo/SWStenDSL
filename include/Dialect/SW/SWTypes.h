/**
 * @file SWTypes.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 定义SW方言中使用的类型
 * @version 0.1
 * @date 2021-03-03
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _DIALECT_SW_SW_TYPES_H_
#define _DIALECT_SW_SW_TYPES_H_

#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <llvm/ADT/STLExtras.h>
#include "SWDialect.h"

namespace mlir {
namespace sw {

// 变量类型的存储类型
namespace SWTypeStorage {
    struct GridTypeStorage;
    struct MemRefTypeStorage;
} // end of namespace SWTypeStorage

// Grid类型, 表示数组, 作为其他类型的基类
class GridType : public Type {
public:
    using ImplType = SWTypeStorage::GridTypeStorage;
    using Type::Type;

    static bool classof(Type type);

    // 获取域中元素类型
    Type getElementType() const;

    // 获取类型的形状(各个维度的大小)
    ArrayRef<int64_t> getShape() const;
    // 获取类型的维度数
    int64_t getRank() const { return getShape().size(); }
};

// MemRefType, 用来表示计算时使用的数组
class MemRefType
    : public Type::TypeBase<MemRefType, GridType, SWTypeStorage::MemRefTypeStorage> {
public:
    using Base::Base;

    static MemRefType get(Type elementType, ArrayRef<int64_t> shape);
};

} // end of namespace sw
} // end of namespace mlir

#endif // end of _DIALECT_SW_SW_TYPES_H_
