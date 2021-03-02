/**
 * @file StencilDialect.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief stencil 方言相关函数实现
 * @version 0.1
 * @date 2021-02-25
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

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"

using namespace mlir;
using namespace mlir::stencil;

//============================================================================//
// Stencil Dialect
//============================================================================//
// 构造函数
StencilDialect::StencilDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StencilDialect>()) {
    
    addTypes<FieldType, ResultType>();

    addOperations<
    #define GET_OP_LIST
    #include "Dialect/Stencil/StencilOps.cpp.inc"
    >();

    // 本方言允许未注册的操作出现(比如scf, std中的操作等等)
    allowUnknownOperations();
}

// 类型解析
Type StencilDialect::parseType(DialectAsmParser &parser) const {
    StringRef prefix;

    // 解析前缀
    if (parser.parseKeyword(&prefix)) {
        parser.emitError(parser.getNameLoc(), "expected type identifier");
        return Type();
    }

    // 解析result类型
    if (prefix == getResultTypeName()) {
        // 解析元素类型
        Type resultType;
        if (parser.parseLess() || parser.parseType(resultType) || parser.parseGreater()) {
            parser.emitError(parser.getNameLoc(), "expected valid result type");
            return Type();
        }
        return ResultType::get(resultType);
    }

    // 解析field类型
    if (prefix == getFieldTypeName()) {
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

        // 返回field类型
        return FieldType::get(elementType, shape);
    }

    // 解析类型失败
    parser.emitError(parser.getNameLoc(), "unknown stencil type ")
        << parser.getFullSymbolSpec();
    return Type();
}

//============================================================================//
// 类型打印
//============================================================================//

namespace {
void printGridType(StringRef name, Type type, DialectAsmPrinter &printer) {
    printer << name;
    printer << "<";
    for (auto size : type.cast<GridType>().getShape()) {
        printer << size;
        printer << "x";
    }
    printer << type.cast<GridType>().getElementType() << ">";
}

void printResultType(StringRef name, Type type, DialectAsmPrinter &printer) {
    printer << name;
    printer << "<" << type.cast<ResultType>().getResultType() << ">";
}
} // end of anonymous namespace

void StencilDialect::printType(Type type, DialectAsmPrinter &printer) const {
    TypeSwitch<Type>(type)
        .Case<FieldType>(
            [&](Type) { printGridType(getFieldTypeName(), type, printer); })
        .Case<ResultType>(
            [&](Type) { printResultType(getResultTypeName(), type, printer);})
        .Default([](Type) { llvm_unreachable("unexpected 'shape' type kind"); });
}