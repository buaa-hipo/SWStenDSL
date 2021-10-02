/**
 * @file ConvertStencilToVector.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief stencil方言转化为vector方言pass实现
 * @version 0.1
 * @date 2021-09-25
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>
#include <iostream>

#include "Conversion/StencilToVector/ConvertStencilToVector.h"
#include "Conversion/StencilToVector/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "PassDetail.h"

using namespace mlir;
using namespace stencil;
using namespace vector;

namespace {
//============================================================================//
// ReWriting 模式
//============================================================================//
// access 需要变换为maskedLoad
class AccessOpLowering : public StencilOpToVectorPattern<stencil::AccessOp> {
public:
    using StencilOpToVectorPattern<stencil::AccessOp>::StencilOpToVectorPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto accessOp = cast<stencil::AccessOp>(operation);
        auto offsetOp = cast<OffsetOp>(operation);
        // 使用vector中的masked load替换, 开发使用的mlir版本没有提供load, 
        // 所以使用masked load 替换.
        Value masked_value = rewriter.create<vector::ConstantMaskOp>(loc, 
                VectorType::get(vectorWidth, rewriter.getI1Type()), 
                rewriter.getI64ArrayAttr(vectorWidth));
        auto elementType = operands[0].getType().cast<mlir::stencil::FieldType>().getElementType();
        Value pass_thru_value = rewriter.create<ConstantOp>(loc,
                VectorType::get(vectorWidth, elementType), 
                rewriter.getZeroAttr(VectorType::get(vectorWidth, elementType)));
        Value memrefCast_value = rewriter.create<stencil::CastToMemRefOp>(loc, 
                operands[0], offsetOp.getOffset(), vectorWidth);
        VectorType resType = VectorType::get(vectorWidth, elementType);

        Value maskedLoad_value = rewriter.create<vector::MaskedLoadOp>(loc,
                resType, memrefCast_value, masked_value, pass_thru_value);
        // 替换掉原有的access Op
        rewriter.replaceOp(accessOp, maskedLoad_value);
        return success();
    }
};

// load需要变换为BroadCastOp
class LoadOpLowering : public StencilOpToVectorPattern<stencil::LoadOp> {
public:
    using StencilOpToVectorPattern<stencil::LoadOp>::StencilOpToVectorPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto loadOp = cast<stencil::LoadOp>(operation);
        auto offsetOp = cast<OffsetOp>(operation);
        Value newLoadOpResult = rewriter.create<mlir::stencil::LoadOp>(loc, operands[0], offsetOp.getOffset());
        // 在load op后面创建一个broadcast op, 将单一的值扩展为指定的向量维度
        VectorType resType = VectorType::get(vectorWidth, newLoadOpResult.getType());
        Value broadcastOpResult = rewriter.create<vector::BroadcastOp>(loc, resType, newLoadOpResult);
        // 替换原有的load op
        rewriter.replaceOp(loadOp, broadcastOpResult);
        return success();
    }
};

// Constant op的结果类型从标量转换为向量
class ConstantOpLowering : public StencilOpToVectorPattern<ConstantOp> {
public:
    using StencilOpToVectorPattern<ConstantOp>::StencilOpToVectorPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto constantOp = cast<ConstantOp>(operation);
        auto value = constantOp.value().cast<FloatAttr>().getValue();
        auto type = constantOp.getResult().getType();

        // 直接替换
        VectorType newResType = VectorType::get(vectorWidth, type);
        SmallVector<APFloat, 4> attrValue;
        for (int i = 0; i < vectorWidth; i++)
            attrValue.push_back(value);
        auto newOpAttr = DenseFPElementsAttr::get(VectorType::get(vectorWidth, type), attrValue);
        Value newResult = rewriter.create<ConstantOp>(loc, newResType, newOpAttr);
        rewriter.replaceOp(constantOp, newResult);
        return success();
    };
};

using AddFOpLowering = StandardOpConvertToVectorPattern<AddFOp>;
using SubFOpLowering = StandardOpConvertToVectorPattern<SubFOp>;
using MulFOpLowering = StandardOpConvertToVectorPattern<MulFOp>;
using DivFOpLowering = StandardOpConvertToVectorPattern<DivFOp>;

//============================================================================//
// 转换目标
//============================================================================//
class StencilToVectorTarget : public ConversionTarget {
private:
    unsigned int vectorWidth;
public:
    explicit StencilToVectorTarget(MLIRContext &context, unsigned int vectorWidth)
        : ConversionTarget(context) {
            this->vectorWidth = vectorWidth;
        }

    // 返回使用指定变量的指定类型的操作
    template<typename OpTy>
    OpTy getUserOp(Value value) const {
        for (auto user : value.getUsers()) {
            if (OpTy op = dyn_cast<OpTy>(user))
                return op;
        }

        return nullptr;
    }

    bool isDynamicallyLegal(Operation *op) const override {
        // 获取这些操作的父applyOp, 判断最内层tile是否为vectorWidth的整数倍,
        // 如果不是则不能改写,即不启用向量化(此处认为其为合法op)
        auto applyOp = op->getParentOfType<mlir::stencil::ApplyOp>();
        auto innerestTile = applyOp.getTile().back();
        auto loc = applyOp.getLoc();
        if (innerestTile % vectorWidth) {
            llvm::raw_ostream &output = llvm::outs();
            output << "WARNING: Innerest tile is " << innerestTile 
                << ", disable vectorization for kernel ";
            loc.print(output);
            output << "\n";
            return true;
        }
        if (dyn_cast<AddFOp>(op) || dyn_cast<SubFOp>(op)
            || dyn_cast<MulFOp>(op) || dyn_cast<DivFOp>(op)
            || dyn_cast<ConstantOp>(op)) {
                auto resultType = op->getResult(0).getType();
                if (resultType.isa<VectorType>())
                    return true;
        }

        // stencil.load会被改写但仍保持使用, 只有当其结果被vector.broadcast使用时
        // 才是合法op
        auto loadOp = dyn_cast<mlir::stencil::LoadOp>(op);
        if (loadOp && getUserOp<vector::BroadcastOp>(loadOp.getResult())) {
            return true;
        }

        return false;
    }
};

//============================================================================//
// Rewriting pass
//============================================================================//
struct StencilToVectorPass : public StencilToVectorPassBase<StencilToVectorPass> {
    StencilToVectorPass() = default;
    StencilToVectorPass(unsigned int vectorWidth) {
        this->VectorWidth = vectorWidth;
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<VectorDialect>();
    }

    void runOnOperation() override;
};

void StencilToVectorPass::runOnOperation() {
    OwningRewritePatternList patterns;
    auto module = getOperation();

    StencilTypeConvertToVectorTypeConverter typeConverter(module.getContext());
    populateStencilToVectorConversionPatterns(typeConverter, patterns, VectorWidth);

    StencilToVectorTarget target(*(module.getContext()), VectorWidth);
    target.addLegalDialect<mlir::stencil::StencilDialect>();
    target.addDynamicallyLegalOp<mlir::stencil::AccessOp>();
    target.addDynamicallyLegalOp<mlir::stencil::LoadOp>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addDynamicallyLegalOp<AddFOp, SubFOp, MulFOp, DivFOp, ConstantOp>();
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<VectorDialect>();
    target.addLegalOp<FuncOp>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    if (failed(applyFullConversion(module, target, patterns))) {
        signalPassFailure();
    }
}

} // end of anonymous namespace

namespace mlir {
namespace stencil {

// 填充转换模式列表
void populateStencilToVectorConversionPatterns(
        StencilTypeConvertToVectorTypeConverter &typeConverter,
        mlir::OwningRewritePatternList &patterns, unsigned int vectorWidth) {
    patterns.insert<AccessOpLowering, LoadOpLowering, ConstantOpLowering, 
                    AddFOpLowering, SubFOpLowering, MulFOpLowering,
                    DivFOpLowering>(typeConverter, vectorWidth);
}

//============================================================================//
// Stencil类型转换器
//============================================================================//
StencilTypeConvertToVectorTypeConverter::StencilTypeConvertToVectorTypeConverter(
        MLIRContext *context_) : context(context_) {
    // 给field类型添加类型转换
    addConversion([&](stencil::GridType type){
        return VectorType::get(type.getShape(), type.getElementType());
    });

    addConversion([&](Type type) -> Optional<Type> {
        if (auto gridType = type.dyn_cast<stencil::GridType>())
            return llvm::None;
        return type;
    });
}

//============================================================================//
// Stencil转写vector模式基类
//============================================================================//
StencilToVectorPattern::StencilToVectorPattern(
    StringRef rootOpName, StencilTypeConvertToVectorTypeConverter &typeConverter,
    unsigned int &vectorWidth, PatternBenefit benefit) 
    : ConversionPattern(rootOpName, benefit, typeConverter.getContext()),
    typeConverter(typeConverter), vectorWidth(vectorWidth) {}
} // end of namespace stencil
} // end of namespace mlir

std::unique_ptr<Pass> mlir::createConvertStencilToVectorPass(unsigned int vectorWidth) {
    return std::make_unique<StencilToVectorPass>(vectorWidth);
}