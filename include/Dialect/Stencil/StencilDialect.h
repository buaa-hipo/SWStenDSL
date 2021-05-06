/**
 * @file StencilDialect.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief Stencil方言定义
 * @version 0.1
 * @date 2021-02-19
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _DIALECT_STENCIL_STENCILDIALECT_H_
#define _DIALECT_STENCIL_STENCILDIALECT_H_
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include <cstdint>

namespace mlir {
namespace stencil {

// 维度标识
constexpr static int kIDimension = 0;
constexpr static int kJDimension = 1;
constexpr static int kKDimension = 2;

// 索引类型大小
constexpr static int kIndexSize = 3;

// 索引类型, 用来保存偏移量及域边界
typedef SmallVector<int64_t, kIndexSize> Index;

class StencilDialect : public Dialect {
public:
    explicit StencilDialect(MLIRContext *context);

    // 返回stencil方言中使用的IR对应的文本名称
    static StringRef getDialectNamespace() {return "stencil";}
    static StringRef getStencilProgarmAttrName() { return "stencil.program"; }
    static StringRef getStencilIterationAttrName() { return "stencil.iteration"; }
    static StringRef getFieldTypeName() {return "field";}
    static StringRef getResultTypeName() {return "result";}

    static bool isStencilProgram(FuncOp funcOp) {
        return !!funcOp.getAttr(getStencilProgarmAttrName());
    }

    static bool isStencilIteration(FuncOp funcOp) {
        return !!funcOp.getAttr(getStencilIterationAttrName());
    }

    // 解析本方言中的类型
    Type parseType(DialectAsmParser &parser) const override;
    // 输出本方言中的类型
    void printType(Type type, DialectAsmPrinter &os) const override;
};

} /* end of namespace stencil */
} /* end of namespace mlir */

#endif // end of _DIALECT_STENCIL_STENCILDIALECT_H_