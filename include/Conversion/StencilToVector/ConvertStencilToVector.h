/**
 * @file ConvertStencilToVector.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 将stencil方言转化为vector方言
 * @version 0.1
 * @date 2021-09-25
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _CONVERSION_STENCILTOVECTOR_CONVERTSTENCILTOVECTOR_H_
#define _CONVERSION_STENCILTOVECTOR_CONVERTSTENCILTOVECTOR_H_

#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <cstdint>
#include <tuple>
#include "Dialect/Stencil/StencilOps.h"

namespace mlir {
namespace stencil {
//============================================================================//
// 将stencil类型转化为vector类型
//============================================================================//
struct StencilTypeConvertToVectorTypeConverter : public TypeConverter {
    using TypeConverter::TypeConverter;

    StencilTypeConvertToVectorTypeConverter(MLIRContext *context);

    // 返回上下文
    MLIRContext *getContext() { return context; }

private:
    MLIRContext *context;
};

//============================================================================//
// 将stencil操作转化为vector操作的基类
//============================================================================//
class StencilToVectorPattern : public ConversionPattern {
public:
    StencilToVectorPattern(StringRef rootOpName, 
                            StencilTypeConvertToVectorTypeConverter &typeConverter,
                            unsigned int &vectorWidth, PatternBenefit benefit = 1);

protected:
    // 类型转换器的引用
    StencilTypeConvertToVectorTypeConverter &typeConverter;

    // 向量寄存器宽度
    unsigned int vectorWidth;
};

//============================================================================//
// 实现匹配每一个操作的模式的辅助类
//============================================================================//
template <typename OpTy>
class StencilOpToVectorPattern : public StencilToVectorPattern {
public:
    StencilOpToVectorPattern(StencilTypeConvertToVectorTypeConverter &typeConverter,
                        unsigned int &vectorWidth, PatternBenefit benefit = 1)
        : StencilToVectorPattern(OpTy::getOperationName(), typeConverter, vectorWidth, benefit) {}
};

// standard op 标量转换为向量
template <typename OpTy>
class StandardOpConvertToVectorPattern : public StencilToVectorPattern {
public:
    StandardOpConvertToVectorPattern(StencilTypeConvertToVectorTypeConverter &typeConverter,
                    unsigned int &vectorWidth, PatternBenefit benefit = 1)
    : StencilToVectorPattern(OpTy::getOperationName(), typeConverter, vectorWidth, benefit) {}

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto srcOp = cast<OpTy>(operation);
        
        VectorType resType = VectorType::get(vectorWidth, srcOp.getResult().getType());
        OperationState state(loc, OpTy::getOperationName());
        state.addTypes(resType);
        state.addOperands(operands);
        state.addAttributes(operation->getAttrs());
        Operation *newOp = rewriter.createOperation(state);
        rewriter.replaceOp(operation, newOp->getResult(0));

        return success();
    }
};

// 填充conversion模式列表的辅助函数
void populateStencilToVectorConversionPatterns(
    StencilTypeConvertToVectorTypeConverter &typeConverter, 
    OwningRewritePatternList &patterns, unsigned int vectorWidth);
} // end of namespace stencil
} // end of namespace mlir

#endif // _CONVERSION_STENCILTOVECTOR_CONVERTSTENCILTOVECTOR_H_