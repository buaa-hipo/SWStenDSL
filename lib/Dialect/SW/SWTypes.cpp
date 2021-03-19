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
struct GridTypeStorage : public TypeStorage {
    Type elementType;
    const size_t size;
    const int64_t *shape;

    GridTypeStorage(Type elementTy, size_t size, const int64_t *shape)
        : TypeStorage(), elementType(elementTy), size(size), shape(shape) {}
    
    // 哈希键
    using KeyTy = std::pair<Type, ArrayRef<int64_t>>;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(elementType, getShape());
    }

    Type getElementType() const { return elementType; }
    ArrayRef<int64_t> getShape() const { return {shape, size}; }
};

// MemRefTypeStorage 存储类
struct MemRefTypeStorage : public GridTypeStorage {
    using GridTypeStorage::GridTypeStorage;

    // 构造
    static MemRefTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
        // 复制域的各个维度
        ArrayRef<int64_t> shape = allocator.copyInto(key.second);
        return new (allocator.allocate<MemRefTypeStorage>())
                MemRefTypeStorage(key.first, shape.size(), shape.data());
    }
};
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
