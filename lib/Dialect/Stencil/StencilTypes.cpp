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
struct GridTypeStorage : public TypeStorage {

    Type elementType;
    const size_t size;
    const int64_t *shape;

    GridTypeStorage(Type elementTy, size_t size, const int64_t *shape)
        : TypeStorage(), elementType(elementTy), size(size), shape(shape) {}
    
    // 哈希键, 用来类型区分
    using KeyTy = std::pair<Type, ArrayRef<int64_t>>;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(elementType, getShape());
    }

    Type getElementType() const { return elementType; }
    ArrayRef<int64_t> getShape() const { return {shape, size}; }
};

// FieldTypeStorage 存储类
struct FieldTypeStorage : public GridTypeStorage {
    using GridTypeStorage::GridTypeStorage;

    // 构造
    static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
        // 复制域的各个维度
        ArrayRef<int64_t> shape = allocator.copyInto(key.second);
        return new (allocator.allocate<FieldTypeStorage>())
                FieldTypeStorage(key.first, shape.size(), shape.data());
    }
};

// ResultTypeStorage 存储类
struct ResultTypeStorage : public TypeStorage {
    Type resultType;

    ResultTypeStorage(Type resultType) : TypeStorage(), resultType(resultType) {}

    // 哈希键, 用于类型区分
    using KeyTy = Type;

    bool operator==(const KeyTy &key) const { return key == resultType; }    
    Type getResultType() const { return resultType; }

    // 构造
    static ResultTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
        return new (allocator.allocate<ResultTypeStorage>()) ResultTypeStorage(key);
    }
};
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