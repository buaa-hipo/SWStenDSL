/**
 * @file StencilTypes.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 定义stencil方言中使用的类型
 * @version 0.1
 * @date 2021-02-23
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _DIALECT_STENCIL_STENCIL_TYPES_H_
#define _DIALECT_STENCIL_STENCIL_TYPES_H_

#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <llvm/ADT/STLExtras.h>
#include "StencilDialect.h"

namespace mlir {
namespace stencil {

// 变量类型的存储类型
namespace StencilTypeStorage {
    struct GridTypeStorage;
    struct FieldTypeStorage;
    struct ResultTypeStorage;
} // end of namespace StencilTypeStorage

// Grid类型, 表示结构化数组, 作为其他结构化数组类型的基类
class GridType : public Type {
public:
    using ImplType = StencilTypeStorage::GridTypeStorage;
    using Type::Type;

    static bool classof(Type type);

    // 获取域中元素类型
    Type getElementType() const;

    // 获取类型的形状(各个维度的大小)
    ArrayRef<int64_t> getShape() const;
    // 获取类型的维度数
    int64_t getRank() const {return getShape().size(); }
};

// FieldType, 用来表示计算过程中使用的结构化数组
class FieldType
    : public Type::TypeBase<FieldType, GridType, StencilTypeStorage::FieldTypeStorage> {
public:
    using Base::Base;

    static FieldType get(Type elementType, ArrayRef<int64_t> shape);
};

// ResultType, 用来保存相应点的中间结果
class ResultType
    : public Type::TypeBase<ResultType, Type, StencilTypeStorage::ResultTypeStorage> {
public:
    using Base::Base;
    
    static ResultType get(Type resultType);
    // 获取结果类型
    Type getResultType() const;
};
} // namespace stencil
} // namespace mlir

#endif // end of _DIALECT_STENCIL_STENCIL_TYPES_H_