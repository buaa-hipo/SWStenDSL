#ifndef _DIALECT_SW_SW_BASE_TD_
#define _DIALECT_SW_SW_BASE_TD_

include "mlir/IR/OpBase.td"

//============================================================================//
// SW Dialect
//============================================================================//
def SW_Dialect : Dialect {
    let name ="sw";
    let description = [{ SW Dialect for general expression }];
    let cppNamespace = "::mlir::sw";
}

//============================================================================//
// SW Types
//============================================================================//
def SW_IsMemRefType : CPred<"$_self.isa<::mlir::sw::MemRefType>()">;

def SW_MemRef : Type<SW_IsMemRefType, "multidimensional array">;

def SW_Element : AnyTypeOf<[F32, F64]>;

def SW_Index : Confined<I64ArrayAttr, [ArrayMinCount<1>]>;

//============================================================================//
// SW Operations
//============================================================================//
// 所有SW operation 的基类
class SW_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SW_Dialect, mnemonic, traits> {
    
    // 对于每一个SW Operation, 以下静态函数需要在SWOps.cpp中定义
    // static ParseResult parse<op-c++-class-name> (OpAsmParser &parser,
    //                                              OperationState &state);
    // static void print(OpAsmPrinter &p, <op-c++-class-name> op)

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
}

#endif // end of _DIALECT_SW_SW_BASE_TD_
