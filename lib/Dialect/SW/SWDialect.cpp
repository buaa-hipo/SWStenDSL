/**
 * @file SWDialect.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief SW 方言相关函数实现
 * @version 0.1
 * @date 2021-03-04
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <cstdint>

#include "Dialect/SW/SWDialect.h"
#include "Dialect/SW/SWOps.h"
#include "Dialect/SW/SWTypes.h"

using namespace mlir;
using namespace mlir::sw;

//============================================================================//
// SW Dialect
//============================================================================//
// 构造函数
SWDialect::SWDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<SWDialect>())
{
    addTypes<MemRefType>();

    addOperations<
    #define GET_OP_LIST
    #include "Dialect/SW/SWOps.cpp.inc"
    >();
}

// 类型解析
Type SWDialect::parseType(DialectAsmParser &parser) const
{
    StringRef prefix;

    // 解析前缀
    if (failed(parser.parseKeyword(&prefix))) {
        parser.emitError(parser.getNameLoc(), "expected type identifier");
        return Type();
    }
    // 解析类型
    // 解析形状
    SmallVector<int64_t, 3> shape;
    if (parser.parseLess() || parser.parseDimensionList(shape)) {
        parser.emitError(parser.getNameLoc(), "expected valid dimension list");
        return Type();
    }
    // 解析元素类型
    Type elementType;
    if (parser.parseType(elementType) || parser.parseGreater()) {
        parser.emitError(parser.getNameLoc(), "expected valid element type");
        return Type();
    }

    // 返回MemRef类型
    return MemRefType::get(elementType, shape);
}

// 类型打印
void SWDialect::printType(Type type, DialectAsmPrinter &printer) const
{
    printer << getMemrefTypeName();
    printer << "<";

    for (auto size : type.cast<GridType>().getShape()) {
        printer << size;
        printer << "x";
    }

    printer << type.cast<GridType>().getElementType();
}