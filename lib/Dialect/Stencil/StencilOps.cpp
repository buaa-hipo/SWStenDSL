/**
 * @file StencilDialect.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief stencil 方言类中操作的相关函数实现
 * @version 0.1
 * @date 2021-02-25
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <tuple>

#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"

using namespace mlir;

//============================================================================//
// apply操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseApplyOp(OpAsmParser &parser, OperationState &state) {
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<OpAsmParser::OperandType, 8> arguments;
    SmallVector<Type, 8> operandTypes;

    // 解析赋值列表
    if (succeeded(parser.parseLParen())) { // 解析左括号
        do {
            OpAsmParser::OperandType currentArgument, currentOperand;
            Type currentType;

            if (failed(parser.parseRegionArgument(currentArgument))
                || failed(parser.parseEqual())
                || failed(parser.parseOperand(currentOperand))
                || failed(parser.parseColonType(currentType)))
                return failure();
            
            arguments.push_back(currentArgument);
            operands.push_back(currentOperand);
            operandTypes.push_back(currentType);
        } while(succeeded(parser.parseOptionalComma())); // 解析可能存在的逗号(多参数情况下)
        if (failed(parser.parseRParen())) // 解析右括号
            return failure();
    }
    
    // 解析结果类型
    SmallVector<Type, 8> resultTypes;
    if (failed(parser.parseArrowTypeList(resultTypes)))
        return failure();

    // 解析operand类型
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands))
        || failed(parser.addTypesToList(resultTypes, state.types)))
    return failure();

    // 解析region域
    Region *body = state.addRegion();
    if (failed(parser.parseRegion(*body, arguments, operandTypes)))
        return failure();

    // 解析上下边界
    ArrayAttr lbAttr, ubAttr;
    if (succeeded(parser.parseKeyword("in"))) {
        if (failed(parser.parseLParen()) // 解析左括号
            || failed(parser.parseAttribute(lbAttr, stencil::ApplyOp::getLBAttrName(), state.attributes)) // 下界
            || failed(parser.parseColon()) // 解析冒号
            || failed(parser.parseAttribute(ubAttr, stencil::ApplyOp::getUBAttrName(), state.attributes)) //上界
            || failed(parser.parseRParen())) // 解析右括号
            return failure();
    } else {
        return failure();
    }

    // 解析tile
    ArrayAttr tileAttr;
    if (succeeded(parser.parseOptionalKeyword("tile"))) {
        if (failed(parser.parseLParen()) // 解析左括号
            || failed(parser.parseAttribute(tileAttr, stencil::ApplyOp::getTileAttrName(), state.attributes))
            || failed(parser.parseRParen())) // 解析右括号
            return failure(); 
    }

    // 解析cacheAt
    Attribute cacheAtAttr;
    if (succeeded(parser.parseOptionalKeyword("cacheAt"))) {
        if (failed(parser.parseLParen()) // 解析左括号
            || failed(parser.parseAttribute(cacheAtAttr, stencil::ApplyOp::getCacheAtAttrName(), state.attributes))
            || failed(parser.parseRParen())) // 解析右括号
            return failure();
    }

    return success();
}

// 打印函数
static void print(stencil::ApplyOp applyOp, OpAsmPrinter &printer) {
    printer << stencil::ApplyOp::getOperationName() << ' ';

    SmallVector<Value, 10> operands = applyOp.getOperands();
    // 输出参数列表
    if (!applyOp.region().empty() && !operands.empty()) {
        Block *body = applyOp.getBody();
        printer << "(";
        llvm::interleaveComma(
            llvm::seq<int>(0, operands.size()), printer, [&](int i) {
                printer << body->getArgument(i) << " = " << operands[i] << " : "
                        << operands[i].getType();
            }
        );
        printer << ")";
    }

    // 输出结果类型
    printer << "->";
    if (applyOp.res().size() > 1)
        printer << "(";
    llvm::interleaveComma(applyOp.res().getTypes(), printer);
    if (applyOp.res().size() > 1)
        printer << ")";
    
    // 输出region域
    printer.printRegion(applyOp.region(), /*printEntryBlockArgs=*/false);

    // 输出边界
    printer << " in (";
    printer.printAttribute(applyOp.lb());
    printer << " : ";
    printer.printAttribute(applyOp.ub());
    printer << ")";
    
    // 输出tile
    if (applyOp.tile().hasValue()) {
        printer << " tile(";
        printer.printAttribute(applyOp.tile().getValue());
        printer << ")";
    }

    // 输出cacheAt
    if (applyOp.cacheAt().hasValue()) {
        printer << " cacheAt(";
        printer << applyOp.cacheAt().getValue();
        printer << ")";
    }
}

// 获取使用了本Op返回值的Op中与返回值绑定的参数
OpOperand *stencil::StoreOp::getReturnOpOperand() {
    auto current = res();
    // 如果有使用返回值的OP
    while (current.hasOneUse()) {
        // 获取使用返回值的OP
        OpOperand *operand = current.getUses().begin().getOperand();
        // 如果找到return操作, 返回相应的参数
        if (isa<stencil::ReturnOp>(operand->getOwner()))
            return operand;
        
        // 如果没有找到, 则尝试寻找scf::YieldOp (循环展开未能整除符合该情况)
        auto yieldOp = dyn_cast<scf::YieldOp>(operand->getOwner());
        if (!yieldOp)
            return nullptr;
        // 在父region中继续搜索, 此时return操作使用的是yieldOp的父op的返回值
        current = yieldOp.getParentOp()->getResult(operand->getOperandNumber());
    }

    return nullptr;
}

//============================================================================//
// 正则化
//============================================================================//
stencil::ApplyOpPattern::ApplyOpPattern(MLIRContext *context,
                                        PatternBenefit benefit)
    : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

stencil::ApplyOp
stencil::ApplyOpPattern::cleanupOpArguments(stencil::ApplyOp applyOp,
                                            PatternRewriter &rewriter) const {
    // 计算新的参数列表并重新进行绑定
    llvm::DenseMap<Value, unsigned int> newIndex;
    SmallVector<Value, 10> newOperands;
    for (auto &en : llvm::enumerate(applyOp.getOperands())) {
        // 当前参数尚未重新绑定
        if (newIndex.count(en.value()) == 0) {
            // 当前参数在apply中被使用, 重新绑定
            if (!applyOp.getBody()->getArgument(en.index()).getUses().empty()) {
                newIndex[en.value()] = newOperands.size();
                newOperands.push_back(en.value());
            } else {
                // 未被使用的参数映射到第一个索引
                newIndex[en.value()] = 0;
            }
        }
    }

    // 创建新的apply操作符, 精简参数列表
    if (newOperands.size() < applyOp.getNumOperands()) {
        auto loc = applyOp.getLoc();
        auto shapeOp = cast<ShapeOp>(applyOp.getOperation());
        auto newOp = rewriter.create<stencil::ApplyOp>(loc, newOperands, 
                                                    shapeOp.getLB(),
                                                    shapeOp.getUB(),
                                                    applyOp.getTile(),
                                                    applyOp.cacheAt(),
                                                    applyOp.getResultTypes());
        
        // 生成参数绑定关系并且移动region域
        SmallVector<Value, 10> newArgs(applyOp.getNumOperands());
        llvm::transform(applyOp.getOperands(), newArgs.begin(), [&](Value value) {
            return newOperands.empty()
                    ? value // 新Op如果没有参数则直接传递默认值
                    : newOp.getBody()->getArgument(newIndex[value]);
        });
        rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(), newArgs);
        return newOp;
    }
    return nullptr;
}

namespace {

// 该模式用来移除apply操作中冗余的或者未使用的参数
struct ApplyOpArgumentCleaner : public stencil::ApplyOpPattern {
    using ApplyOpPattern::ApplyOpPattern;   

    LogicalResult matchAndRewrite(stencil::ApplyOp applyOp, 
                                    PatternRewriter &rewriter) const override {
        if (auto newOp = cleanupOpArguments(applyOp, rewriter)) {
            rewriter.replaceOp(applyOp, newOp.getResults());
            return success();
        }
        return failure();
    }
};

// 该模式用来将copy操作移动到最后面
struct CopyOpHoisting : public OpRewritePattern<stencil::CopyOp> {
    using OpRewritePattern<stencil::CopyOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(stencil::CopyOp copyOp,
                                    PatternRewriter &rewriter) const override {
        auto op = copyOp.getOperation();
        auto current = op;
        // 跳过计算的操作
        while (current->getNextNode() && !isa<stencil::CopyOp>(current->getNextNode())
                && !current->getNextNode()->isKnownTerminator())
            current = current->getNextNode();
        // 移动该操作
        if (current != op) {
            rewriter.setInsertionPointAfter(current);
            rewriter.replaceOp(op, rewriter.clone(*op)->getResults());
            return success();
        }
        return failure();
    }
};
} // end of anonymous namespace

//============================================================================//
// register canonicalization patterns
//============================================================================//
void stencil::ApplyOp::getCanonicalizationPatterns(
        OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<ApplyOpArgumentCleaner>(context);
}

void stencil::CopyOp::getCanonicalizationPatterns(
        OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CopyOpHoisting>(context);
}

namespace mlir {
namespace stencil {
#include "Dialect/Stencil/StencilInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"
} // end of namespace stencil
} // end of namespace mlir