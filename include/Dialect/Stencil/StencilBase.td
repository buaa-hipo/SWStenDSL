#ifndef _DIALECT_STENCIL_STENCIL_BASE_TD_
#define _DIALECT_STENCIL_STENCIL_BASE_TD_

include "mlir/IR/OpBase.td"

//============================================================================//
// Stencil Dialect
//============================================================================//
def Stencil_Dialect : Dialect {
    let name = "stencil";
    let description = [{Stencil Dialect for general expression. }];
    let cppNamespace = "stencil";
}

//============================================================================//
// Stencil Types
//============================================================================//
def Stencil_IsFieldType : CPred<"$_self.isa<::mlir::stencil::FieldType>()">;
def Stencil_IsResultType : CPred<"$_self.isa<::mlir::stencil::ResultType>()">;

def Stencil_Field : Type<Stencil_IsFieldType, "structed array">;
def Stencil_Result : Type<Stencil_IsResultType, "a result">;

def Stencil_Element : AnyTypeOf<[F32, F64]>;

def Stencil_Index : Confined<I64ArrayAttr, [ArrayMinCount<1>]>;

//============================================================================//
// Stencil Operations
//============================================================================//

// 所有stencil operation的基类
class Stencil_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Stencil_Dialect, mnemonic, traits> {
    
    // 对于每一个Stencil Operation, 以下静态函数需要在StencilOps.cpp中定义

    // static ParseResult parse<op-c++-class-name> (OpAsmParser &parser,
    //                                              OperationState &state);
    // static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // static LogicalResult verify(<op-c++-class-name> op)

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this) }];
}

#endif // end of _DIALECT_STENCIL_STENCIL_BASE_TD_