/**
 * @file ConvertStencilToSW.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 将stencil方言转化为sw方言
 * @version 0.1
 * @date 2021-03-22
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _CONVERSION_STENCILTOSW_CONVERTSTENCILTOSW_H_
#define _CONVERSION_STENCILTOSW_CONVERTSTENCILTOSW_H_

#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <cstdint>
#include <tuple>
#include "Dialect/Stencil/StencilOps.h"

namespace mlir {
namespace stencil {
//============================================================================//
// 将stencil类型转化为sw类型`
//============================================================================//
struct StencilTypeConverter : public TypeConverter {
    using TypeConverter::TypeConverter;

    StencilTypeConverter(MLIRContext *context);

    // 返回上下文
    MLIRContext *getContext() { return context; }

private:
    MLIRContext *context;
};

//============================================================================//
// 将stencil操作转化为sw操作的基类
//============================================================================//
class StencilToSWPattern : public ConversionPattern {
public:
    StencilToSWPattern(StringRef rootOpName, StencilTypeConverter &typeConverter,
                        DenseMap<Value, Index> &valueToLB,
                        DenseMap<Value, OpOperand *> &valueToOperand,
                        DenseMap<Value, unsigned int> &valueToApplyOpIndex,
                        PatternBenefit benefit = 1);
    
    // 返回嵌套循环的循环变量
    SmallVector<Value, 6> getInductionVars(Operation *operation) const;

    
    // 返回使用指定变量的指定类型的操作
    template<typename OpTy>
    OpTy getUserOp(Value value) const {
        for (auto user : value.getUsers()) {
            if (OpTy op = dyn_cast<OpTy>(user))
                return op;
        }

        return nullptr;
    }

protected:
    // 类型转换器的引用
    StencilTypeConverter & typeConverter;

    // 原程序下界存储集合
    DenseMap<Value, Index> &valueToLB;

    // 传递给return操作的结果集合
    DenseMap<Value, OpOperand *> &valueToOperand;

    // 原程序apply 的参数及其位置映射
    DenseMap<Value, unsigned int> &valueToApplyOpIndex;
};

//============================================================================//
// 实现匹配每一个操作的模式的辅助类
//============================================================================//
template <typename OpTy>
class StencilOpToSWPattern : public StencilToSWPattern {
public:
    StencilOpToSWPattern(StencilTypeConverter &typeConverter,
                            DenseMap<Value, Index> &valueToLB,
                            DenseMap<Value, OpOperand *> &valueToOperand,
                            DenseMap<Value, unsigned int> &valueToApplyOpIndex,
                            PatternBenefit benefit = 1)
        : StencilToSWPattern(OpTy::getOperationName(), typeConverter, valueToLB,
                                valueToOperand, valueToApplyOpIndex, benefit) {}
};

// 填充conversion模式列表的辅助函数
void populateStencilToSWConversionPatterns(
    StencilTypeConverter &typeConverter, DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, OpOperand *> &valueToOperand,
    DenseMap<Value, unsigned int> &valueToApplyOpIndex,
    OwningRewritePatternList &patterns);
} // end of namespace stencil
} // end of namespace mlir

#endif // _CONVERSION_STENCILTOSW_CONVERTSTENCILTOSW_H_