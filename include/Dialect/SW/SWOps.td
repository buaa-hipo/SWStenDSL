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
// for操作终结符
//============================================================================//
def SW_YieldOp : SW_Op<"yield", [
    HasParent<"ForOp">,
    Terminator
    ]> {
    
    let summary = "Terminator for ForOp";
    let description = [{
        This op terminates the only block inside the only region of a `sw.for`.
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
    let  arguments = (ins SW_Element:$lhs, SW_Element:$rhs);
    // 返回值
    let results = (outs SW_Element:$res);

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
    let  arguments = (ins SW_Element:$lhs, SW_Element:$rhs);
    // 返回值
    let results = (outs SW_Element:$res);
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
    let  arguments = (ins SW_Element:$lhs, SW_Element:$rhs);
    // 返回值
    let results = (outs SW_Element:$res);
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
    let arguments = (ins SW_Element:$lhs, SW_Element:$rhs);
    // 返回值
    let results = (outs SW_Element:$res);

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
#endif // end of _DIALECT_SW_SW_OPS_TD_