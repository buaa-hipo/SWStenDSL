#ifndef _DIALECT_STENCIL_STENCIL_OPS_TD_
#define _DIALECT_STENCIL_STENCIL_OPS_TD_

include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect/Stencil/StencilBase.td"
include "Dialect/Stencil/StencilInterfaces.td"

//============================================================================//
// apply操作, 该操作描述了一个stencil计算的过程
// * 该OP引用了ShapeOp接口
// * 该OP的终结符为ReturnOp
// * OP的region域未引用OP前面定义的变量
// * 当该OP变为dead code时可以直接移去, 没有副作用
//============================================================================//
def Stencil_ApplyOp : Stencil_Op<"apply", [
    DeclareOpInterfaceMethods<ShapeOp>,
    SingleBlockImplicitTerminator<"ReturnOp">,
    IsolatedFromAbove,
    NoSideEffect]> {
    let summary = "apply operation, stencil expression";
    let description = [{
        This op takes a stencil function and parameters, then applies then
        function to the inputs, and returns the result.
    }];

    // 传入参数
    let arguments = (ins Variadic<AnyType>:$operands,
                        Stencil_Index:$lb,
                        Stencil_Index:$ub,
                        OptionalAttr<Stencil_Index>:$tile,
                        OptionalAttr<I64Attr>:$cacheAt);
    // 返回值
    let results = (outs Variadic<Stencil_Field>:$res);
    // 函数域
    let regions = (region SizedRegion<1>:$region);
    // IR表示形式
    // let assemblyFormat = [{
    //     `(` $operands `)` `(` $lb `:` $ub `)` attr-dict-with-keyword  `{` $region `}`
    // }];

    // 包含正则化函数
    let hasCanonicalizer = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, "
                    "ArrayRef<int64_t> lb, ArrayRef<int64_t> ub, "
                    "ArrayRef<int64_t> tile, Optional<long unsigned int> cacheAt, "
                    "TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
            state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
            state.addAttribute(getTileAttrName(), builder.getI64ArrayAttr(tile));
            state.addAttribute(getCacheAtAttrName(), builder.getI64IntegerAttr(cacheAt.getValue()));
            auto region = state.addRegion();
            region->push_back(new Block());
            for (auto operand : operands) {
                region->front().addArgument(operand.getType());
            }
            state.addTypes(resultTypes);
        }]>
    ];

    // 有效性检查
    let verifier = [{
        // 检查参数
        if (region().front().getNumArguments() != operands().size())
            return emitOpError("operand and argument counts do not match");
        unsigned int operandsNum = operands().size();
        for (unsigned int i = 0; i < operandsNum; i++) {
            if (region().front().getArgument(i).getType() != operands()[i].getType())
                return emitOpError("operand and argument types do not match");
        }
        // 检查结果
        auto shapeOp = cast<ShapeOp>(this->getOperation());
        for (auto result : res()) {
            auto fieldType = result.getType().cast<GridType>();
            if (shapeOp.getRank() != fieldType.getRank())
                return emitOpError("expected result rank to match the operation rank");
        }

        // 检查是否能够被tile整除
        if (tile().hasValue()) {
            auto LB = shapeOp.getLB();
            auto UB = shapeOp.getUB();
            auto tile = getTile();
            if (tile.size() != LB.size() || tile.size() != UB.size())
                return emitOpError("expected tile, lb, ub have same size");
            
            for (int i = 0; i < tile.size(); i++) {
                if ((UB[i] - LB[i]) % tile[i] != 0)
                    return emitOpError("ub-lb must  be divisible by tile");
            }
        }

        // 检查cacheAt是否不大于问题域维度
        if (cacheAt().hasValue()) {
            if (shapeOp.getRank() <= cacheAt().getValue())
                return emitOpError("cacheAt is bigger then field rank");
        }

        return success();
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getLBAttrName() { return "lb"; }
        static StringRef getUBAttrName() { return "ub"; }
        static StringRef getTileAttrName() { return "tile"; }
        static StringRef getCacheAtAttrName() { return "cacheAt"; }

        Index getTile() {
            Index result;
            Optional<ArrayAttr> tileAttr = tile();
            for (auto &elem : tileAttr.getValue())
                result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
            return result;
        }

        Attribute getCacheAtAttr() { return getAttr("cacheAt"); }

        Block *getBody() { return &region().front(); }
    }];
}

//============================================================================//
// access操作, 访问当前位置指定偏移量位置的点
// * 当该OP变为dead code时可以直接移去, 没有副作用
//============================================================================//
def Stencil_AccessOp : Stencil_Op<"access", [
    DeclareOpInterfaceMethods<OffsetOp>,
    NoSideEffect]> {
    let summary = "stencil access operation, visit the point at specfic offset of current point";
    let description = [{
        This operation accesses a element given a constant offset.
        The offset is specified relative to the current position.
    }];
    // 传入参数
    let arguments = (ins Stencil_Field:$field, Stencil_Index:$offset);
    // 返回值
    let results = (outs Stencil_Element:$res);
    // IR 表示
    let assemblyFormat = [{
        $field $offset attr-dict-with-keyword `:` functional-type($field, $res)
    }];

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value field, ArrayRef<int64_t> offset", [{
            state.addOperands(field);
            state.addAttribute(getOffsetAttrName(), builder.getI64ArrayAttr(offset));
            auto fieldType = field.getType().cast<stencil::GridType>();
            state.addTypes(fieldType.getElementType());
        }]>
    ];

    // 有效性检查
    let verifier = [{
        auto fieldType = field().getType().cast<stencil::GridType>();
        // 偏移量维度与域的维度必须相同
        if (offset().size() != fieldType.getRank())
            return emitOpError("offset and field dimensions do not match");
        // 返回值的类型必须与域中元素类型相同
        if (res().getType() != fieldType.getElementType())
            return emitOpError("result type and element type do not match");
        return success();
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getOffsetAttrName() { return "offset"; }
    }];

}

//============================================================================//
// load操作, 以绝对坐标访问相应位置的点, 用于数组式的参数
// * 当该OP变为dead code时可以直接移去, 没有副作用
//============================================================================//
def Stencil_LoadOp : Stencil_Op<"load", [
    DeclareOpInterfaceMethods<OffsetOp>,
    NoSideEffect]> {
    let summary = "stencil load operation, access point at specfic position";
    let description = [{
        This operation accesses a element given a position. This operation
        is used to access element in parameter array.
    }];
    // 传入参数
    let arguments = (ins Stencil_Field:$field, Stencil_Index:$offset);
    // 返回值
    let results = (outs Stencil_Element:$res);
    // IR表示形式
    let assemblyFormat = [{
        $field $offset attr-dict-with-keyword `:` functional-type($field, $res)
    }];

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState & state, "
                    "Value field, ArrayRef<int64_t> offset", [{
            state.addOperands(field);
            state.addAttribute(getOffsetAttrName(), builder.getI64ArrayAttr(offset));
            auto fieldType = field.getType().cast<stencil::GridType>();
            state.addTypes(fieldType.getElementType());
        }]>
    ];

    // 有效性检查
    let verifier = [{
        auto fieldType = field().getType().cast<stencil::GridType>();
        // 绝对坐标维度与域的维度必须相同
        if (offset().size() != fieldType.getRank())
            return emitOpError("absolute position and field dimensions do not match");
        // 返回值的类型必须与域中元素类型相同
        if (res().getType() != fieldType.getElementType())
            return emitOpError("result type and element type do not match");
        return success();
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getOffsetAttrName() { return "offset"; }
    }];
}

//============================================================================//
// store操作, 将计算的中间结果存储到StencilResult类型中, 返回一个值或0个值
//============================================================================//
def Stencil_StoreOp : Stencil_Op<"store", []> {
    let summary = "store operation, store the result to stencil";
    let description = [{
        The store operation stores an operand value or nothing.
    }];
    // 传入参数
    let arguments = (ins Variadic<Stencil_Element>:$operands);
    // 返回值
    let results = (outs Stencil_Result:$res);
    // IR表示形式
    let assemblyFormat = [{
        ($operands^)? attr-dict-with-keyword `:` functional-type($operands, $res)
    }];

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value value", [{
            state.addOperands({value});
            state.addTypes(stencil::ResultType::get(value.getType()));
        }]>
    ];

    // 有效性检查
    let verifier = [{
        // 参数至多为一个
        if (operands().size() > 1)
            return emitOpError("expected at most one operand");
        // 检查返回值类型
        auto resultType = res().getType().cast<stencil::ResultType>().getResultType();
        if (operands().size() == 1 && operands()[0].getType() != resultType)
            return emitOpError("operand type and result type are inconsistent");
        
        // 检查结果绑定
        if (!getReturnOpOperand())
            return emitOpError("result not mapped to a unique stencil.return operand");
        
        return success();
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        // 获取使用了本Op返回值的Op中与返回值绑定的参数
        OpOperand *getReturnOpOperand();
    }];
}

//============================================================================//
// return操作, 返回结果, apply操作的终结符
//============================================================================//
def Stencil_ReturnOp : Stencil_Op<"return", [
    Terminator,
    HasParent<"ApplyOp">]> {
    let summary = "return operation, terminator of apply operation";
    let description = [{
        This operation terminates the apply operation and writes the results
        of the stencil expression to the field value returned by the apply operation.

        The types and the number of operands must match the results of the
        apply operation.

        The optional unroll attribute enables the implementation of loop 
        unrolling at the stencil dialect level.
    }];
    // 传入参数
    let arguments = (ins Variadic<Stencil_Result>:$operands,
                        OptionalAttr<Stencil_Index>:$unroll);
    // IR表示形式
    let assemblyFormat = [{
        (`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)
    }];
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state", [{
            // 使用这个终结符的操作会执行检测操作，这个函数是喂给检测函数的
        }]>,
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, Optional<ArrayAttr> unroll", [{
            state.addOperands({operands});
            if(unroll.hasValue())
                state.addAttribute(getUnrollAttrName(), unroll.getValue());
        }]>
    ];

    // 有效性检验
    let verifier = [{
        auto applyOp = cast<stencil::ApplyOp>(getParentOp());
        unsigned int unrollFactor = getUnrollFactor();
        size_t unrollDim = getUnrollDimension();

        // 参数的数量与apply的结果相匹配
        auto results = applyOp.res();
        if (getNumOperands() != unrollFactor*results.size())
            return emitOpError("the operand and apply result counts do not match");
        
        // 检查元素类型是否匹配
        unsigned int resultSize = results.size();
        for (unsigned int i = 0; i < resultSize; i++) {
            auto fieldType = applyOp.getResult(i).getType().cast<GridType>();
            for (unsigned int j = 0; j < unrollFactor; j++) {
                if (getOperand(i*unrollFactor + j).getType().cast<stencil::ResultType>().getResultType() != fieldType.getElementType())
                    return emitOpError("the operand and apply result element types do not match");
            }
        }
        return success();
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getUnrollAttrName() { return "unroll"; }
        Index getUnroll() {
            Index result;
            ArrayAttr unrollAttr = unroll().getValue();
            for (auto &elem : unrollAttr)
                result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
            return result;
        }
        // 获取展开因子
        unsigned int getUnrollFactor() {
            unsigned int factor = 1;
            if (unroll().hasValue()) {
                Index unroll = getUnroll();
                factor = std::accumulate(unroll.begin(), unroll.end(), 1,
                                            std::multiplies<int64_t>());
            }
            return factor;
        }
        // 获取unroll的维度
        size_t getUnrollDimension() {
            size_t dimension = 0;
            if (unroll().hasValue()) {
                Index unroll = getUnroll();
                unsigned int unrollFactor = getUnrollFactor();
                auto it = llvm::find_if(unroll, [&](int64_t x){
                    return x == unrollFactor;
                });
                assert(it != unroll.end() &&
                        "expected to find unroll factor");
                dimension = std::distance(unroll.begin(), it);
            }
            return dimension;
        }
    }];
}

//============================================================================//
// copy操作, 将计算的结构拷贝到输出数组上
//============================================================================//
def Stecnil_CopyOp : Stencil_Op<"copy", [
    DeclareOpInterfaceMethods<ShapeOp>
    ]> {
    let summary = "copy operation, copy the result to output array";
    let description = [{
        This operation writes the result to the output array.
    }];
    // 传入参数
    let arguments = (ins Stencil_Field:$result, Stencil_Field:$output,
                        Stencil_Index:$lb, Stencil_Index:$ub);
    // IR表示形式
    let assemblyFormat = [{
        $result `to` $output `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($result) `to` type($output)
    }];
    // 包含正则化函数
    let hasCanonicalizer = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, Value result, "
                    "Value output, ArrayRef<int64_t> lb, ArrayRef<int64_t> ub", [{
            state.addOperands({result, output});
            state.addAttribute(getLBAttrName(), builder.getI64ArrayAttr(lb));
            state.addAttribute(getUBAttrName(), builder.getI64ArrayAttr(ub));
        }]>
    ];

    // 有效性检查
    let verifier = [{
        // 检查result和output的类型是否匹配
        auto resultType = result().getType().cast<stencil::GridType>();
        auto outputType = output().getType().cast<stencil::GridType>();
        if (resultType.getRank() != outputType.getRank())
            return emitOpError("the result type and output type have different rank");
        if (resultType.getElementType() != outputType.getElementType())
            return emitOpError("the result type and output type have different element type");
        
        // result参数必须是apply的返回值
        if (!dyn_cast<stencil::ApplyOp>(result().getDefiningOp()))
            return emitOpError("result parameter is not result of apply operation");
        // output参数不得是其他操作的输入
        if (llvm::count_if(output().getUsers(), [](Operation *op) { return isa_and_nonnull<stencil::ApplyOp>(op); }) != 0)
            return emitOpError("an output can not be an input");
        // 不得对output多次写入
        if (llvm::count_if(output().getUsers(), [](Operation *op) { return isa_and_nonnull<stencil::CopyOp>(op); }) != 1)
            return emitOpError("multiple copys to the same output");
        
        return success();

    }];
    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getLBAttrName() { return "lb"; }
        static StringRef getUBAttrName() { return "ub"; }
    }];
}

//============================================================================//
// iteration 操作, 用以实现多次计算迭代以及相关参数绑定
//============================================================================//
def Stencil_IterationOp : Stencil_Op<"iteration"> {
    let summary = "iteration operation, represents iterative process";
    let description = [{
        This operation represents iterative process, and binds the parameter.
    }];

    // 传入参数
    let arguments = (ins FlatSymbolRefAttr:$stencilFuncName, Variadic<Stencil_Field>:$operands, I64Attr:$bindParamNum, I64Attr:$iterNum);
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, SymbolRefAttr stencilFunc, "
                    "ValueRange bind_params, int64_t iterNum, int64_t bind_param_num", [{
            state.addOperands(bind_params);
            state.addAttribute(getStencilFuncAttrName(), stencilFunc);
            state.addAttribute(getBindParamNumAttrName(), builder.getI64IntegerAttr(bind_param_num));
            state.addAttribute(getIterNumAttrName(), builder.getI64IntegerAttr(iterNum));
        }]>
    ];

    // 正确性检查, 无需检查, 直接返回true
    let verifier = [{
        return success();
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getStencilFuncAttrName() { return "stencilFuncName"; }
        static StringRef getBindParamNumAttrName() { return "bindParamNum"; }
        static StringRef getIterNumAttrName() { return "iterNum"; }

        // 获取调用函数名称
        StringRef getStencilFuncName() { return stencilFuncName(); }
        unsigned int getBindParamNum() { 
            return getAttrOfType<IntegerAttr>(getBindParamNumAttrName()).getInt(); 
        }
        unsigned int  getIterNum() { 
            return getAttrOfType<IntegerAttr>(getIterNumAttrName()).getInt(); 
        }
    }];
}
#endif // end of _DIALECT_STENCIL_STENCIL_OPS_TD_