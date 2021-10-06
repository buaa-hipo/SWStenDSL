#ifndef _DIALECT_SW_SW_OPS_TD_
#define _DIALECT_SW_SW_OPS_TD_

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "Dialect/SW/SWBase.td"

//============================================================================//
// module操作, 其中包含一个从核函数
//============================================================================//
def SW_ModuleOp : SW_Op<"module", [
    IsolatedFromAbove, SymbolTable, Symbol,
    SingleBlockImplicitTerminator<"ModuleEndOp">]> {
    let summary = "A top level compilation unit containing code to be run on SPE.";
    let description = [{
        SW module contains code that is intended to be run on slave core.
        MPE can launch this code through a sw.launch_func that creates a fully
        qualified symbol through the sw.module's symbol and a sw.func symbol contained
        in the sw.module.
    }];

    // 域
    let regions = (region SizedRegion<1>:$region);
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, StringRef name", [{
            state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
            auto region = state.addRegion();
            region->push_back(new Block());
        }]>
    ];
}

//============================================================================//
// module 操作终结符
//============================================================================//
def SW_ModuleEndOp : SW_Op<"module_end", [
    Terminator, HasParent<"ModuleOp">]> {
    
    let summary = "A pseudo op that marks the end of a sw.module.";
    let description = [{
        This op terminates the only block inside the only region of a `sw.module`.
    }];
}

//============================================================================//
// func操作, 运行在从核上的函数, 包含在module操作中
//============================================================================//
def SW_FuncOp : SW_Op<"func", [
    HasParent<"ModuleOp">,
    AutomaticAllocationScope,
    // FunctionLike,
    Symbol,
    SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Function executable on spe";
    let description = [{
        Defines a function that can be executed on SPE.
    }];

    // 函数域
    let regions = (region SizedRegion<1>:$region);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, StringRef name, "
                    "FunctionType type, ArrayRef<Type> cacheRead, "
                    "ArrayRef<Type> cacheWrite", [{
            state.addAttribute(SymbolTable::getSymbolAttrName(), 
                                builder.getStringAttr(name));
            state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
            state.addAttribute(getCacheReadAttrNumName(), builder.getI64IntegerAttr(cacheRead.size()));
            state.addAttribute(getCacheWriteAttrNumName(), builder.getI64IntegerAttr(cacheWrite.size()));
            Region *body = state.addRegion();
            Block *entryBlock = new Block;
            entryBlock->addArguments(type.getInputs());
            entryBlock->addArguments(cacheRead);
            entryBlock->addArguments(cacheWrite);
            body->getBlocks().push_back(entryBlock);
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{

        static StringRef getTypeAttrName() { return "type"; }
        static StringRef getCacheReadAttrName() { return "cacheRead"; }
        static StringRef getCacheWriteAttrName() { return "cacheWrite"; }
        static StringRef getCacheReadAttrNumName() {return "cacheReadAttrNum"; }
        static StringRef getCacheWriteAttrNumName() { return "cacheWriteAttrNum"; }

        FunctionType getType() {
            return getAttrOfType<TypeAttr>(getTypeAttrName()).getValue().cast<FunctionType>();
        }

        unsigned int getNumFuncArguments() { return getType().getNumInputs(); }
        unsigned int getNumFuncResults() { return getType().getNumResults(); }

        // 返回cacheRead属性的数量
        unsigned int getNumCacheReadAttributions() {
            return getAttrOfType<IntegerAttr>(getCacheReadAttrNumName()).getInt();
        }
        // 返回cacheWrite属性的数量
        unsigned int getNumCacheWriteAttributions() {
            return getAttrOfType<IntegerAttr>(getCacheWriteAttrNumName()).getInt();
        }
        // 返回所有cacheRead属性
        ArrayRef<BlockArgument> getCacheReadAttributions() {
            auto begin = std::next(getBody()->args_begin(), getType().getNumInputs());
            auto end = std::next(begin, getNumCacheReadAttributions());

            return {begin, end};
        }
        // 返回所有cacheWrite属性
        ArrayRef<BlockArgument> getCacheWriteAttributions() {
            auto begin = std::next(getBody()->args_begin(), getType().getNumInputs()+getNumCacheReadAttributions());
            auto end = getBody()->args_end();
            return {begin, end};
        }
    }];
}

//============================================================================//
// func操作终结符
//============================================================================//
def SW_ReturnOp : SW_Op<"return", [
    HasParent<"FuncOp">,
    Terminator
    ]> {
    
    let summary = "Terminator for FuncOp";
    let description = [{
        This op terminates the only block inside the only region of a `sw.func`.
    }];
}

//============================================================================//
// main_func 操作
//============================================================================//
def SW_MainFuncOp : SW_Op <"main_func", [
        AutomaticAllocationScope,
        FunctionLike,
        Symbol
    ]> {
    let summary = "Function executable on mpe";
    let description = [{
        Defines a function that can be executed on MPE.
    }];

    // 函数域
    let regions = (region SizedRegion<1>:$region);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, StringRef name, "
                    "FunctionType type", [{
            state.addAttribute(SymbolTable::getSymbolAttrName(),
                                builder.getStringAttr(name));
            state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
            Region *body = state.addRegion();
            Block *entryBlock = new Block;
            entryBlock->addArguments(type.getInputs());
            body->getBlocks().push_back(entryBlock);
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        unsigned int getNumFuncArguments() { return getType().getNumInputs(); }
        unsigned int getNumFuncResults() { return getType().getNumResults(); }
    }];
}

//============================================================================//
// main_func操作终结符
//============================================================================//
def SW_MainReturnOp : SW_Op<"main_return", [
    HasParent<"MainFuncOp">,
    Terminator
    ]> {
    let summary = "Terminator for MainFuncOp";
    let description = [{
        This op terminates the only block inside the only region of a `sw.main_func`.
    }];
}

//============================================================================//
// main_iteration_func操作
//============================================================================//
def SW_MainIterationFuncOp : SW_Op<"main_iteration_func", [
        AutomaticAllocationScope,
        FunctionLike,
        Symbol
    ]> {
    let summary = "Iteration Func executable on mpe";
    let description = [{
        Defines a iteration that can be executed on MPE.
    }];

    // 函数域
    let regions = (region SizedRegion<1>:$region);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, StringRef name, "
                    "FunctionType type", [{
            state.addAttribute(SymbolTable::getSymbolAttrName(),
                                builder.getStringAttr(name));
            state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
            Region *body = state.addRegion();
            Block *entryBlock = new Block;
            entryBlock->addArguments(type.getInputs());
            body->getBlocks().push_back(entryBlock);
        }]>
    ];
    // 辅助函数
    let extraClassDeclaration = [{
        unsigned int getNumFuncArguments() { return getType().getNumInputs(); }
        unsigned int getNumFuncResults() { return getType().getNumResults(); }
    }];
}

//============================================================================//
// main_iteration_return
//============================================================================//
def SW_MainIterationReturnOp : SW_Op<"main_iteration_return", [
        HasParent<"MainIterationFuncOp">,
        Terminator
    ]> {
    let summary = "Terminator for MainIterationFuncOp";
    let description = [{
        This op terminates the only block inside the only region 
        of a `sw.main_iteration_return`.
    }];
}

//============================================================================//
// launch_func操作，MPE通过该函数调用相应的kernel函数
//============================================================================//
def SW_LaunchFuncOp : SW_Op<"launch_func"> {
    let summary = "Launches a spe function.";
    let description = [{
        host launches a spe function.
    }];

    // 传入参数
    let arguments = (ins FlatSymbolRefAttr:$kernel,
                        Variadic<SW_MemRef>:$operands);
    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "SymbolRefAttr kernelFunc, "
                    "ValueRange kernelOperands", [{
            state.addOperands(kernelOperands);
            state.addAttribute(getKernelAttrName(), kernelFunc);
        }]>
    ];

    let extraClassDeclaration = [{
        // 获取spe函数名称
        StringRef getKernelName() { return kernel(); }

        // 获取spe函数的第i个参数
        Value getKernelOperand(unsigned int i) { getOperand(i); }

        // 获取spe函数Attribute的名称
        static StringRef getKernelAttrName() { return "kernel"; }
    }];
}

//============================================================================//
// launch_main_func操作, 负责调用main_func
//============================================================================//
def SW_LaunchMainFuncOp : SW_Op<"launch_main_func"> {
    let summary = "Launches a main function.";
    let description = [{
        host launches a main function.
    }];

    // 传入参数
    let arguments = (ins FlatSymbolRefAttr: $mainFuncName,
                        Variadic<SW_MemRef>: $operands);

    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "SymbolRefAttr mainFuncName, "
                    "ValueRange mainFuncOperands", [{
            state.addOperands(mainFuncOperands);
            state.addAttribute(getMainFuncAttrName(), mainFuncName);
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        // 获取mainFunc函数的Attribute名称
        static StringRef getMainFuncAttrName() { return "mainFuncName"; }

        // 获取mainFunc名称
        StringRef getMainFuncName() { return mainFuncName(); }
    }];
}

//============================================================================//
// launch 操作, 内联形态的spe函数, 用于支持下一步outlining
//============================================================================//
def SW_LaunchOp : SW_Op<"launch", []> {
    let summary = "SW spe launch operation.";
    let description = [{
        launch a spe function. The body of the spe function is defined by the single 
        region that this operation contains.
    }];

    // 传入参数
    let arguments = (ins Variadic<AnyType>:$operands);

    // 域
    let regions = (region SizedRegion<1>:$region);

    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, "
                    "ArrayRef<Type> cacheRead, ArrayRef<Type> cacheWrite", [{
            // 处理参数和属性
            state.addOperands(operands);
            state.addAttribute(getCacheReadAttrNumName(), builder.getI64IntegerAttr(cacheRead.size()));
            state.addAttribute(getCacheWriteAttrNumName(), builder.getI64IntegerAttr(cacheWrite.size()));
            
            // 处理域
            auto region = state.addRegion();
            region->push_back(new Block());
            for (auto operand : operands) 
                region->front().addArgument(operand.getType());
            region->front().addArguments(cacheRead);
            region->front().addArguments(cacheWrite);
        }]>
    ];

    let extraClassDeclaration = [{
        // 获取cacheRead属性名称
        static StringRef getCacheReadAttrName() { return "cacheRead"; }
        // 获取cacheWrite属性名称
        static StringRef getCacheWriteAttrName() { return "cacheWrite"; }
        // 获取cacheRead数量属性名称
        static StringRef getCacheReadAttrNumName() { return "cacheReadAttrNum"; }
        // 获取cacheWrite数量属性名称
        static StringRef getCacheWriteAttrNumName() { return "cacheWriteAttrNum"; }
        // 返回cacheRead属性的数量
        unsigned int getNumCacheReadAttributions() {
            return getAttrOfType<IntegerAttr>(getCacheReadAttrNumName()).getInt();
        }
        // 返回cacheWrite属性的数量
        unsigned int getNumCacheWriteAttributions() {
            return getAttrOfType<IntegerAttr>(getCacheWriteAttrNumName()).getInt();
        }
        // 返回所有cacheRead属性
        ArrayRef<BlockArgument> getCacheReadAttributions() {
            auto begin = std::next(region().front().args_begin(), operands().size());
            auto end = std::next(begin, getNumCacheReadAttributions());

            return {begin, end};
        }
        // 返回所有cacheWrite属性
        ArrayRef<BlockArgument> getCacheWriteAttributions() {
            auto begin = std::next(region().front().args_begin(), operands().size() + getNumCacheReadAttributions());
            auto end = region().front().args_end();
            return {begin, end};
        }
        // 获取函数体
        Block *getBody() { return &region().front(); }
    }];
}

//============================================================================//
// launch 操作终结符
//============================================================================//
def SW_TerminatorOp : SW_Op<"terminator", [
    HasParent<"LaunchOp">,
    Terminator
    ]> {
    
    let summary = "Terminator for LaunchOp";
    let description = [{
        This op terminates the only block inside the only region of a `sw.launch`.
    }];
}
//============================================================================//
// for 操作, 表示一层循环
//============================================================================//
def SW_ForOp : SW_Op<"for", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveSideEffects]> {
    
    let summary = "sw for operation";
    let description = [{
        The "sw.for" operation represents a loop taking 3 SSA value as operands
        which represent the lower bound, upper bound and step respectively.
    }];

    // 传入参数
    let arguments = (ins    AnyTypeOf<[I32, I64]>: $lowerBound,
                            AnyTypeOf<[I32, I64]>: $upperBound,
                            AnyTypeOf<[I32, I64]>: $step,
                            Variadic<AnyType>:$initArgs);
    // for 域
    let regions = (region SizedRegion<1>:$region);
    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value lowerBound, Value upperBound, Value step", [{
            // 设置上下界以及步长
            state.addOperands({lowerBound, upperBound, step});
            
            // 设置迭代变量类型
            auto region = state.addRegion();
            region->push_back(new Block());
            Block &bodyBlock = region->front();
            bodyBlock.addArgument(lowerBound.getType());
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        // 获取迭代变量
        Value getInductionVar() { return region().front().getArgument(0); }
        void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
        void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
        void setStep(Value step) { getOperation()->setOperand(2, step); }

        unsigned int getNumControlOperands() { return 3; }
        Block *getBody() { return &region().front(); }
    }];
}

//============================================================================//
// for, if操作终结符
//============================================================================//
def SW_YieldOp : SW_Op<"yield", [
    ParentOneOf<["ForOp", "IfOp"]>,
    Terminator
    ]> {
    
    let summary = "Terminator for ForOp, IfOp";
    let description = [{
        This op terminates the only block inside the only region of a `sw.for`.
        or regions of a `sw.if`
    }];
}

//============================================================================//
// load 操作, 以绝对坐标访问相应位置的点
//============================================================================//
def SW_LoadOp : SW_Op<"load", [NoSideEffect]> {
    let summary = "sw load operation, access point at specified position";
    let description = [{
        This operation accesses a element given a position.
    }];

    // 传入参数
    let arguments = (ins SW_MemRef:$input, Variadic<AnyTypeOf<[I32, I64]>>:$pos);

    // 返回值
    let results = (outs SW_Element:$res);
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value input, ValueRange pos", [{
            state.addOperands(input);
            state.addOperands(pos);
            auto input_type = input.getType().cast<sw::GridType>();
            state.addTypes(input_type.getElementType());
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
    }];
}

//============================================================================//
// store 操作, 将计算的结果写回到相应位置
//============================================================================//
def SW_StoreOp : SW_Op<"store", [NoSideEffect]> {
    let summary = "sw store operation, store the element to the specified position";
    let description = [{
        This operation stores a element to given position.
    }];

    // 传入参数
    let arguments = (ins SW_Element:$elem,
                        SW_MemRef:$output,
                        Variadic<AnyTypeOf<[I32, I64]>>:$pos);

    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value element, Value outputArray, ValueRange pos", [{
            state.addOperands(element);
            state.addOperands(outputArray);
            state.addOperands(pos);
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
    }];
}

//============================================================================//
// constant 操作, 用来定义常量
//============================================================================//
def SW_ConstantOp : SW_Op<"constant", [
    ConstantLike, 
    NoSideEffect]> {
    
    let summary = "constant";
    let description = [{
        This operation assigns a constant value to a SSA value.
    }];

    // 传入参数
    let arguments = (ins AnyAttr:$value);
    // 返回值
    let results = (outs AnyType:$res);
    
    let hasFolder = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Attribute value, TypeRange resultTypes", [{
            state.addAttribute(getValueAttrName(), value);
            state.addTypes(resultTypes);
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getValueAttrName() { return "value"; }
        Attribute getValue() { return getAttr("value"); }
    }];
}

//============================================================================//
// getID操作, 用来获取当前从核号
//============================================================================//
def SW_GetIDOp : SW_Op<"getID", [
    NoSideEffect
]> {
    let summary = "getID";
    let description = [{
        This operation get the spe id.
    }];

    let results = (outs AnyType:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "TypeRange resultTypes", [{
                state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// addf操作, 浮点数加法
//============================================================================//
def SW_AddfOp : SW_Op<"addf", []> {
    let summary = "floating point addition operation";
    let description = [{
        The `addf` operation takes two operands and returns one result, each of 
        these is required to be the same type. This type is a floating point
        scalar type.
    }];

    // 传入参数
    let  arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    // 返回值
    let results = (outs FloatLike:$res);

    let hasFolder = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// addi操作, 整型加法, 用于坐标运算
//============================================================================//
def SW_AddiOp : SW_Op<"addi", []> {
    let summary = "integer addition operation";
    let description = [{
        The `addi` operation takes two operands and returns one result, each of
        these is required to be the same type.
    }];

    // 传入参数
    let  arguments = (ins AnyTypeOf<[I32, I64]>:$lhs, AnyTypeOf<[I32, I64]>:$rhs);
    // 返回值
    let results = (outs AnyTypeOf<[I32, I64]>:$res);

    let hasFolder = 1;
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// subf操作, 浮点数减法
//============================================================================//
def SW_SubfOp : SW_Op<"subf", []> {
    let summary = "floating point subtraction operation";
    let description = [{
        The `subf` operation takes two operands and returns one result, each of 
        these is required to be the same type. This type is a floating point
        scalar type.
    }];

    // 传入参数
    let  arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    // 返回值
    let results = (outs FloatLike:$res);
    let hasFolder = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// subi操作, 整型减法, 用于坐标运算
//============================================================================//
def SW_SubiOp : SW_Op<"subi", []> {
    let summary = "integer addition operation";
    let description = [{
        The `subi` operation takes two operands and returns one result, each of
        these is required to be the same type.
    }];

    // 传入参数
    let  arguments = (ins AnyTypeOf<[I32, I64]>:$lhs, AnyTypeOf<[I32, I64]>:$rhs);
    // 返回值
    let results = (outs AnyTypeOf<[I32, I64]>:$res);

    let hasFolder = 1;
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// mulf操作, 浮点数乘法
//============================================================================//
def SW_MulfOp : SW_Op<"mulf", []> {
    let summary = "floating point multiplication operation";
    let description = [{
        The `mulf` operation takes two operands and returns one result, each of
        these is required to be the same type. This type is a floating point
        scalar type.
    }];

    // 传入参数
    let  arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    // 返回值
    let results = (outs FloatLike:$res);
    let hasFolder = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// muli操作, 整型乘法, 用于坐标运算
//============================================================================//
def SW_MuliOp : SW_Op<"muli", []> {
    let summary = "integer multiplication operation";
    let description = [{
        The `muli` operation takes two operands and returns one result, each of 
        these is required to be the same type.
    }];

    // 传入参数
    let  arguments = (ins AnyTypeOf<[I32, I64]>:$lhs,AnyTypeOf<[I32, I64]>:$rhs);
    // 返回值
    let results = (outs AnyTypeOf<[I32, I64]>:$res);

    let hasFolder = 1;

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// divf操作, 浮点数除法
//============================================================================//
def SW_DivfOp : SW_Op<"divf", []> {
    let summary = "floating point division operation";
    let description = [{
        The `divf` operation takes two operands and returns one result, each of 
        these is required to be the same type. This type is a floating point 
        saclar type.
    }];

    // 传入参数
    let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
    // 返回值
    let results = (outs FloatLike:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands, TypeRange resultTypes", [{
            state.addOperands(operands);
            state.addTypes(resultTypes);
        }]>
    ];
}

//============================================================================//
// 与操作
//============================================================================//
def SW_LogicAndOp : SW_Op<"land", []> {
    let summary = "logical and operation";
    let description = [{
        The `land` operantion takes two operands and return one result, each of
        these is required to be integer type.
    }];

    // 传入参数
    let arguments = (ins I32:$lhs, I32:$rhs);
    // 返回值
    let results = (outs I32:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands", [{
            state.addOperands(operands);
            state.addTypes(builder.getI32Type());
        }]>
    ]; 
}

//============================================================================//
// 或操作
//============================================================================//
def SW_LogicOrOp : SW_Op<"lor", []> {
    let summary = "logical or operation";
    let description = [{
        The `lor` operantion takes two operands and return one result, each of
        these is required to be integer type.
    }];

    // 传入参数
    let arguments = (ins I32:$lhs, I32:$rhs);
    // 返回值
    let results = (outs I32:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands", [{
            state.addOperands(operands);
            state.addTypes(builder.getI32Type());
        }]>
    ]; 
}

//============================================================================//
// 非操作
//============================================================================//
def SW_LogicNotOp : SW_Op<"lnot", []> {
    let summary = "logical not operation";
    let description = [{
        The `lnot` operantion takes one operands and return one result, each of
        these is required to be integer type.
    }];

    // 传入参数
    let arguments = (ins I32:$input);
    // 返回值
    let results = (outs I32:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "ValueRange operands", [{
            state.addOperands(operands);
            state.addTypes(builder.getI32Type());
        }]>
    ]; 
}
//============================================================================//
// memcpyToLDM操作
//============================================================================//
def SW_MemcpyToLDMOp : SW_Op<"memcpyToLDM", []> {
    let summary = "SW memcpy operation";
    let description = [{
        The `memcpyToLDM` operation transfer data from main memory to LDM.
    }];

    // 传入参数
    let arguments = (ins SW_MemRef:$mem_addr,
                            SW_MemRef:$ldm_addr,
                            Variadic<AnyTypeOf<[I32, I64]>>:$mem_index,
                            I64Attr:$z_dim,
                            I64Attr:$cnt,
                            I64Attr:$stride,
                            I64Attr:$bsize);
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value src, Value dst, "
                    "ValueRange pos,"
                    "int64_t z_dim, int64_t cnt, "
                    "int64_t stride, int64_t bsize", [{
            state.addOperands(src);
            state.addOperands(dst);
            state.addOperands(pos);
            state.addAttribute(getZDimAttrName(), builder.getI64IntegerAttr(z_dim));
            state.addAttribute(getCntAttrName(), builder.getI64IntegerAttr(cnt));
            state.addAttribute(getStrideAttrName(), builder.getI64IntegerAttr(stride));
            state.addAttribute(getBsizeAttrName(), builder.getI64IntegerAttr(bsize));
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getZDimAttrName() { return "z_dim"; }
        static StringRef getCntAttrName() { return "cnt"; }
        static StringRef getStrideAttrName() { return "stride"; }
        static StringRef getBsizeAttrName() { return "bsize"; }
    }];
}

//============================================================================//
// memcpyToMem操作
//============================================================================//
def SW_MemcpyToMEMOp : SW_Op<"memcpyToMEM", []> {
    let summary = "SW memcpy operation";
    let description = [{
        The `memcpy` operation transfer data from LDM to main memory.
    }];

    // 传入参数
    let arguments = (ins SW_MemRef:$ldm_addr,
                            SW_MemRef:$mem_addr,
                            Variadic<AnyTypeOf<[I32, I64]>>:$mem_index,
                            I64Attr:$z_dim,
                            I64Attr:$cnt,
                            I64Attr:$stride,
                            I64Attr:$bsize);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value src, Value dst, "
                    "ValueRange pos, int64_t z_dim, int64_t cnt, "
                    "int64_t stride, int64_t bsize", [{
            state.addOperands(src);
            state.addOperands(dst);
            state.addOperands(pos);
            state.addAttribute(getZDimAttrName(), builder.getI64IntegerAttr(z_dim));
            state.addAttribute(getCntAttrName(), builder.getI64IntegerAttr(cnt));
            state.addAttribute(getStrideAttrName(), builder.getI64IntegerAttr(stride));
            state.addAttribute(getBsizeAttrName(), builder.getI64IntegerAttr(bsize));
        }]>
    ];

    let extraClassDeclaration = [{
        static StringRef getZDimAttrName() { return "z_dim"; }
        static StringRef getCntAttrName() { return "cnt"; }
        static StringRef getStrideAttrName() { return "stride"; }
        static StringRef getBsizeAttrName() { return "bsize"; }
    }];
}

//============================================================================//
// 向量化不对界加载
//============================================================================//
def SW_VectorLoadUOp : SW_Op<"vectorLoadU", []> {
    let summary = "SW vector loadu operation";
    let description = [{
        The `vectorLoadU` operation loads unaligned data into vector.
    }];

    // 传入参数
    let arguments = (ins AnyVector:$dst, 
                        SW_MemRef:$ldm_addr, 
                        Variadic<AnyTypeOf<[I32, I64]>>:$pos);

    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, Value dst, "
                    "Value src, ValueRange pos", [{
            state.addOperands(dst);
            state.addOperands(src);
            state.addOperands(pos);
        }]>
    ];
}

//============================================================================//
// 向量化不对界写回
//============================================================================//
def SW_VectorStoreUOp : SW_Op<"vectorStoreU", []> {
    let summary = "SW vector storeu operation";
    let description = [{
        The `vectorStoreU` operation stores data into unaligned address.
    }];

    // 传入参数
    let arguments = (ins AnyVector:$vectorData, 
                        SW_MemRef:$ldm_addr, 
                        Variadic<AnyTypeOf<[I32, I64]>>:$pos);

    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value src, Value dst, ValueRange pos", [{
            state.addOperands(src);
            state.addOperands(dst);
            state.addOperands(pos);
        }]>
    ];
}

//============================================================================//
// 向量化对界加载
//============================================================================//
def SW_VectorLoadOp : SW_Op<"vectorLoad", []> {
    let summary = "SW vector load operation";
    let description = [{
        The `vectorLoad` operation loads aligned data into vector.
    }];

    // 传入参数
    let arguments = (ins AnyVector:$dst, 
                        SW_MemRef:$ldm_addr,
                        Variadic<AnyTypeOf<[I32,I64]>>:$pos);
    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, Value dst, "
                    "Value src, ValueRange pos", [{
            state.addOperands(dst);
            state.addOperands(src);
            state.addOperands(pos);
        }]>
    ];
}

//============================================================================//
// 将一个标量扩展为一个向量
//============================================================================//
def SW_VectorBroadCastOp : SW_Op<"vectorBroadCast", []> {
    let summary = "SW vector BroadCast operation";
    let description = [{
        The `vectorBroadCast` operation broadcast a scalar value to vector value.
    }];

    // 传入参数
    let arguments = (ins SW_Element:$input);
    // 返回值
    let results = (outs AnyVector:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, Value input, "
        "unsigned vectorWidth", [{
            state.addOperands(input);
            state.addTypes(VectorType::get(vectorWidth, input.getType().cast<FloatType>()));
        }]>
    ];
}

//============================================================================//
// alloc操作, 用于申请空间
//============================================================================//
def SW_AllocOp : SW_Op<"alloc", []> {
    let summary = "alloc operation";
    let description = [{
        The `alloc` operation request a specified size of memory space.
    }];

    // 返回值
    let results = (outs SW_MemRef:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "TypeRange resultTypes", [{
            state.addTypes(resultTypes);
        }]>
    ];
}
//============================================================================//
// dealloc操作, 用于释放空间
//============================================================================//
def SW_DeAllocOp : SW_Op<"dealloc", []> {
    let summary = "dealloc operation";
    let description = [{
        The `dealloc` operation release a specified size of memory space.
    }];

    // 传入参数
    let arguments = (ins SW_MemRef:$input);

    let skipDefaultBuilders = 1;
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value operand", [{
            state.addOperands(operand);
        }]>
    ];
}

//============================================================================//
// getMpiRank操作, 用于获取当前主核进程在整个通信域的rank
//============================================================================//
def SW_GetMpiRankOp : SW_Op<"getMpiRank",[]> {
    let summary = "getMpiRank operation";
    let description = [{
        The `getMpiRank` operation get the rank of mpi process.
    }];

    let results = (outs I32:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state", [{
            state.addTypes({builder.getI32Type()});
        }]>
    ];
}

//============================================================================//
// mpiExchangeHalo操作, 在主核之间交换数据
//============================================================================//
def SW_MpiExchangeHaloOp : SW_Op<"mpiExchangeHalo", []> {
    let summary = "mpi data exchange";
    let description = [{
        The `mpiExchange` operation transfer data between mpi process.
    }];

    // 传入参数
    let arguments = (ins SW_MemRef:$dataArray, I32:$rank, 
                        Confined<I64ArrayAttr, [ArrayMinCount<2>]>:$mpiTile,
                        Confined<I64ArrayAttr, [ArrayMinCount<2>]>:$mpiHaloL,
                        Confined<I64ArrayAttr, [ArrayMinCount<2>]>:$mpiHaloU);
    
    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value dataArray, Value rank, ArrayRef<int64_t> mpiTile,"
                    "ArrayRef<int64_t> mpiHaloL, ArrayRef<int64_t> mpiHaloU", [{
            state.addOperands({dataArray, rank});
            state.addAttribute(getMpiTileName(), builder.getI64ArrayAttr(mpiTile));
            state.addAttribute(getMpiHaloLName(), builder.getI64ArrayAttr(mpiHaloL));
            state.addAttribute(getMpiHaloUName(), builder.getI64ArrayAttr(mpiHaloU));
        }]>
    ];

    // 有效性检查
    let verifier = [{
        // mpiTile, mpiHaloL, mpiHaloU的维度必须相同
        int mpiTileSize = mpiTile().size();
        int mpiHaloLSize = mpiHaloL().size();
        int mpiHaloUSize = mpiHaloU().size();

        if (mpiTileSize < 2 || mpiTileSize > 3)
            return emitOpError("mpiTile's and mpiHalo's dimension must is 2 or 3");
        
        if (mpiTileSize == mpiHaloLSize && mpiTileSize == mpiHaloUSize)
            return success();
        
        return emitOpError("mpiTile and mpiHalo must have same dimension");
    }];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getMpiTileName() { return "mpiTile"; }
        static StringRef getMpiHaloLName() { return "mpiHaloL"; }
        static StringRef getMpiHaloUName() { return "mpiHaloU"; }
    }];
}

//============================================================================//
// if-then-else op
//============================================================================//
def SW_IfOp : SW_Op<"if",
    [SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveSideEffects]> {
    let summary = "if-then-else operation";
    let description = [{
        The `sw.if` operation represents an if-then-else construct for
        conditionally executing to regions of code. The operand to an if operation
        is a boolean value.
    }];

    // 传入参数
    let arguments = (ins I32:$condition);
    let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

    // 构造函数
    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value cond, bool withElseRegion", [{
            auto addTerminator = [&](OpBuilder &nested, Location loc) {
                IfOp::ensureTerminator(*nested.getInsertionBlock()->getParent(),
                    nested, loc);
            };

            build(builder, state, cond, addTerminator, 
                    withElseRegion ? addTerminator 
                    : function_ref<void(OpBuilder &, Location)>());
        }]>,
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "Value cond, "
                    "function_ref<void(OpBuilder &, Location)> thenBuilder "
                    " = buildIfOpTerminatedBody, "
                    "function_ref<void(OpBuilder &, Location)> elseBuilder "
                    " = nullptr", [{
            assert(thenBuilder && "the builder callback for 'then' must be present");

            state.addOperands(cond);
            
            OpBuilder::InsertionGuard guard(builder);
            Region *thenRegion = state.addRegion();
            builder.createBlock(thenRegion);
            thenBuilder(builder, state.location);

            Region *elseRegion = state.addRegion();
            if (!elseBuilder)
                return;
            builder.createBlock(elseRegion);
            elseBuilder(builder, state.location);
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        OpBuilder getThenBodyBuilder() {
            Block* body = getBody(0);
            return OpBuilder::atBlockTerminator(body);
        }
        OpBuilder getElseBodyBuilder() {
            Block* body = getBody(1);
            return OpBuilder::atBlockTerminator(body);
        }
    }];
}

//============================================================================//
// Cmp操作,用于浮点数比较
//============================================================================//
// 以下定义了数值比较提供的操作
def SWCmp_EQ    : I64EnumAttrCase<"eq", 0>;     // 判断两个数相等
def SWCmp_GT    : I64EnumAttrCase<"gt", 1>;     // 判断第一个数大于第二个数
def SWCmp_GE    : I64EnumAttrCase<"ge", 2>;     // 判断第一个数大于等于第二个数
def SWCmp_LT    : I64EnumAttrCase<"lt", 3>;     // 判断第一个数小于第二个数
def SWCmp_LE    : I64EnumAttrCase<"le", 4>;     // 判断第一个数小于等于第二个数
def SWCmp_NE    : I64EnumAttrCase<"ne", 5>;     // 判断两个数不相等

def SWCmpPredictateAttr : I64EnumAttr<
    "SWCmpPredicate", "",
    [SWCmp_EQ, SWCmp_GT, SWCmp_GE, SWCmp_LT, SWCmp_LE, SWCmp_NE]> {
    let cppNamespace = "mlir::sw";
}

def SW_CmpOp : SW_Op<"cmp" , [
    NoSideEffect, 
    ]> {
    let summary = "comparison operation";
    let description = [{
        The `cmp` operation compares its two operands according to the float comparison.
    }];

    // 传入参数
    let arguments = (ins SWCmpPredictateAttr:$predicate,
        AnyType:$lhs,
        AnyType:$rhs);
    // 返回值
    let results = (outs I32:$res);

    // 构造函数
    let builders = [
        OpBuilder<"OpBuilder &builder, OperationState &state, "
                    "mlir::sw::SWCmpPredicate predicate, Value lhs, Value rhs", [{
            state.addOperands({lhs, rhs});
            state.addAttribute(CmpOp::getPredicateAttrName(),
                    builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
            state.addTypes(builder.getI32Type());
        }]>
    ];

    // 辅助函数
    let extraClassDeclaration = [{
        static StringRef getPredicateAttrName() { return "predicate"; }
        static mlir::sw::SWCmpPredicate getPredicateByName(StringRef name) {
            auto pred = symbolizeSWCmpPredicate(name);
            assert(pred.hasValue() && "invalid predicate name");
            return pred.getValue();
        }
        mlir::sw::SWCmpPredicate getPredicate() {
            return (mlir::sw::SWCmpPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName()).getInt();
        }
    }];
}


#endif // end of _DIALECT_SW_SW_OPS_TD_