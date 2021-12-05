/**
 * @file SWDialect.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief SW方言定义
 * @version 0.1
 * @date 2021-03-03
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _DIALECT_SW_SWDIALECT_H_
#define _DIALECT_SW_SWDIALECT_H_

#include <mlir/IR/Dialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <cstdint>

namespace mlir {
namespace sw {

// 索引类型大小
constexpr static int kIndexSize = 3;

// 索引类型, 用来保存坐标值
typedef SmallVector<int64_t, kIndexSize> Index;

class SWDialect : public Dialect {
    public:
        explicit SWDialect(MLIRContext *context);

    // 返回SW方言中使用的IR对应的文本名称
    static StringRef getDialectNamespace() { return "sw"; }
    static StringRef getSWProgramAttrName() { return "sw.program"; }
    static StringRef getMemrefTypeName() { return "memref"; }
    static StringRef getResultTypeName() { return "result"; }

    // 解析本方言中的类型
    Type parseType(DialectAsmParser &parser) const override;
    // 输出本方言中的类型
    void printType(Type type, DialectAsmPrinter &os) const override;    
};

} // end of namespace sw
} // end of namespace mlir

#endif // end of _DIALECT_SW_SWDIALECT_H_