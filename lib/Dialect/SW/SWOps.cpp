/**
 * @file SWOps.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief SW 方言类中操作的相关函数实现
 * @version 0.1
 * @date 2021-03-06
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/FunctionImplementation.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <tuple>

#include "Dialect/SW/SWOps.h"
#include "Dialect/SW/SWDialect.h"
#include "Dialect/SW/SWTypes.h"

using namespace mlir;

// 解析属性信息
static ParseResult parseAttributions(OpAsmParser &parser, StringRef keyword, 
                        SmallVectorImpl<OpAsmParser::OperandType> &args,
                        SmallVectorImpl<Type> &argTypes)
{
    // 如果没能找到对应关键字, 出错
    if (failed(parser.parseOptionalKeyword(keyword)))
        return failure();
    // 解析左括号
    if (failed(parser.parseLParen()))
        return failure();

    do {
        OpAsmParser::OperandType arg;
        Type type;

        if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
            return failure();
        
        args.push_back(arg);
        argTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma())); //解析逗号
    // 解析右括号
    if (failed(parser.parseRParen()))
        return failure();
    
    return success();
}

// 数学运算符的解析函数
static ParseResult parseMathOp(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    SmallVector<Type, 8> resultTypes;

    // 解析参数
    do {
        OpAsmParser::OperandType currentOperand;

        if (failed(parser.parseOperand(currentOperand)))
            return failure();
        operands.push_back(currentOperand);
    } while (succeeded(parser.parseOptionalComma())); // 解析逗号

    // 解析参数和结果类型
    Type currentType;
    if (failed(parser.parseColonType(currentType)))
        return failure();
    // 设定参数的类型到operandTypes中
    for (int i = 0; i < operands.size(); i++)
        operandTypes.push_back(currentType);
    // 设定结果的类型到resultTypes中
    resultTypes.push_back(currentType);
    
    // 解析参数和结果类型到state中
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands))
        || failed(parser.addTypesToList(resultTypes, state.types)))
        return failure();
    
    return success();
}

static ParseResult parseKeywordAttr(OpAsmParser &parser, 
                                    OperationState &state, StringRef keyword)
{
    Attribute attr;
    // 解析关键字
    if (failed(parser.parseKeyword(keyword))) 
        return failure();
    // 解析左括号
    if (failed(parser.parseLParen()))
        return failure();
    // 解析属性
    if (failed(parser.parseAttribute(attr, keyword, state.attributes)))
        return failure();
    // 解析右括号
    if (failed(parser.parseRParen()))
        return failure();

    return success();
}

// 解析Memcpy操作, 包括memcpyToLDM和memcpyToMEM
static ParseResult parseMemcpyOp(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    OpAsmParser::OperandType currentOperand;
    Type currentType;

    // 解析参数列表
    // 解析src_addr
    if (failed(parser.parseOperand(currentOperand)))
        return failure();
    operands.push_back(currentOperand);
    // 解析dst_addr
    if (failed(parser.parseComma())
        || failed(parser.parseOperand(currentOperand)))
        return failure();
    operands.push_back(currentOperand);
    // 解析mem_addr的偏移量
    if (succeeded(parser.parseLSquare())) { // 解析左括号
        do {
            if (failed(parser.parseOperand(currentOperand)))
                return failure();
            operands.push_back(currentOperand);
        } while(succeeded(parser.parseOptionalComma())); // 解析可能存在的逗号

        if (failed(parser.parseRSquare())) // 解析右括号
            return failure();
    } else {
        return failure();
    }

    // 解析attributes
    if (failed(parser.parseColon())) // 解析冒号
        return failure();
    if (failed(parseKeywordAttr(parser, state, "z_dim"))
        || failed(parseKeywordAttr(parser, state, "cnt"))
        || failed(parseKeywordAttr(parser, state, "stride"))
        || failed(parseKeywordAttr(parser, state, "bsize")))
        return failure();

    // 解析参数类型
    if (failed(parser.parseColon())) // 解析冒号
        return failure();

    if (succeeded(parser.parseLParen())) { // 解析左括号
        do {
            if (failed(parser.parseType(currentType)))
                return failure();
            operandTypes.push_back(currentType);
        } while (succeeded(parser.parseOptionalComma())); // 解析逗号

        // 现在已经将所有类型添加到operandTypes中了, 但是由于偏移量有多维, 还需补充剩余的维数
        for (int iter = 0; iter < operands.size()-3; iter++)
            operandTypes.push_back(currentType);
        
        if (failed(parser.parseRParen())) // 解析右括号
            return failure();
    }
    
    // 解析参数并将其放到state中
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands)))
        return failure();
        
    return success();
}

// 输出数组形式的Attributes
static void printAttributions(OpAsmPrinter &printer, ArrayRef<BlockArgument> values)
{
    if (values.empty())
        return ;

    auto elemType = values[0].getType().cast<mlir::sw::GridType>().getElementType();

    if (elemType.cast<mlir::FloatType>().getWidth() == 64)
        printer << "double ";
    else
        printer << "float ";
    
    llvm::interleaveComma(values, printer, [&](BlockArgument v){
        printer << v;
        for (auto size : v.getType().cast<mlir::sw::GridType>().getShape()) {
            printer <<"[" << size << "]";
        }
    });

}

//============================================================================//
// module操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseModuleOp(OpAsmParser &parser, OperationState &state)
{
    StringAttr nameAttr;
    SmallVector<OpAsmParser::OperandType, 8> entryArgs;
    SmallVector<Type, 8> argTypes;

    Builder &builder = parser.getBuilder();

    // 解析module名称
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), 
                                state.attributes))
        return failure();

    // 解析region域
    auto *body = state.addRegion();
    if (parser.parseRegion(*body, entryArgs, argTypes))
        return failure();
    
    return success();
}

// 输出函数
static void print(sw::ModuleOp moduleOp, OpAsmPrinter &printer)
{
    printer << "$moduleBegin" << '\n';
    // 输出头文件
    printer << "#include <slave.h>\n";
    printer << "#include <stdio.h>\n";
    printer << "#include <stdlib.h>\n";
    printer << "#include <math.h>\n";
    printer << "#include <string.h>\n";
    printer << "#include <stdint.h>\n";
    printer << "#include \"utils/dma_lib.h\"\n\n";
    
    // 输出域
    printer << "\n$delete";
    printer.printRegion(moduleOp.region(), /*printEntryBlockArgs=*/false, 
                        /*printBlockTerminators=*/false);
    printer << " $delete\n";
    printer << "$moduleEnd\n";
}

//============================================================================//
// module_end 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseModuleEndOp(OpAsmParser &parser, OperationState &state)
{
    // do nothing
    return success();
}

// 输出函数
static void print(sw::ModuleEndOp moduleEndOp, OpAsmPrinter &printer)
{
    // do nothing
}

//============================================================================//
// func操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &state)
{
    // 解析函数名称
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), state.attributes))
        return failure();
    
    // 解析func函数
    SmallVector<OpAsmParser::OperandType, 8> entryArgs;
    SmallVector<Type, 8> argTypes;
    SmallVector<Type, 8> resultTypes;
    SmallVector<NamedAttrList, 1> argAttrs;
    SmallVector<NamedAttrList, 1> resultAttrs;

    bool isVariadic;
    auto signatureLocation = parser.getCurrentLocation();
    if (failed(impl::parseFunctionSignature(parser, /*allowVariadic=*/false, 
        entryArgs, argTypes, argAttrs, isVariadic, resultTypes, resultAttrs)))
        return failure();
    if (entryArgs.empty() && !argTypes.empty())
        return parser.emitError(signatureLocation) 
            << "sw.func requires named arguments";
    
    Builder &builder = parser.getBuilder();
    auto type = builder.getFunctionType(argTypes, resultTypes);
    state.addAttribute(FuncOp::getTypeAttrName(), TypeAttr::get(type));

    // 解析cacheRead属性信息
    if (failed(parseAttributions(parser, sw::FuncOp::getCacheReadAttrName(), entryArgs, argTypes)))
        return failure();
    unsigned int cacheRead_num = entryArgs.size();
    state.addAttribute(sw::FuncOp::getCacheReadAttrNumName(), builder.getI64IntegerAttr(cacheRead_num));

    // 解析cacheWrite属性信息
    if (failed(parseAttributions(parser, sw::FuncOp::getCacheWriteAttrName(), entryArgs, argTypes)))
        return failure();
    state.addAttribute(sw::FuncOp::getCacheWriteAttrNumName(), builder.getI64IntegerAttr(entryArgs.size()-cacheRead_num));

    // 解析域
    auto *body = state.addRegion();
    return parser.parseRegion(*body, entryArgs, argTypes);
}

// 输出函数
static void print(sw::FuncOp funcOp, OpAsmPrinter &printer)
{
    // 输出参数结构体
    FunctionType type = funcOp.getType();
    Region &body = funcOp.getOperation()->getRegion(0);
    ArrayRef<Type> argTypes = type.getInputs();
    printer << "$shareBegin\n";
    printer << "struct " << funcOp.getName() << "_arg {\n";
    int struct_arg_counter = 0;
    for (int iter = 0; iter < argTypes.size(); iter++) {
        printer << "\t";
        auto elemType = argTypes[iter].cast<mlir::sw::GridType>().getElementType();
        if (elemType.cast<mlir::FloatType>().getWidth() == 64)
            printer << "double *";
        else
            printer << "float *";
        printer << "arg" << struct_arg_counter << ";\n";
        struct_arg_counter++;
    }
    printer << "};\n";
    printer << "$shareEnd\n";
    printer << "void slave_" << funcOp.getName() << "(struct " << funcOp.getName() << "_arg * arg); $speDeclare\n";
    printer << "void " << funcOp.getName() << "(struct " << funcOp.getName() << "_arg * arg)\n";

    // 输出初始化部分, 该部分要移动到域中, 交由后期处理
    printer << "$moveInToRegionBegin\n";
    // 输出cacheRead 和 cacheWrite数组
    printAttributions(printer, funcOp.getCacheReadAttributions());
    printer << "; // cacheRead\n";
    
    printAttributions(printer, funcOp.getCacheWriteAttributions());
    printer << "; // cacheWrite\n";
    struct_arg_counter = 0;
    for (int iter = 0; iter < argTypes.size(); iter++) {
        auto elemType = argTypes[iter].cast<mlir::sw::GridType>().getElementType();
        std::string prefix, suffix;
        if (elemType.cast<mlir::FloatType>().getWidth() == 64)
            prefix = "double (*";
        else 
            prefix = "float (*";
        
        suffix = ")";
        auto shape = argTypes[iter].cast<mlir::sw::GridType>().getShape();
        for (int iter_j = 1; iter_j < shape.size(); iter_j++)
            suffix += "[" + std::to_string(shape[iter_j]) + "]";
        
        printer << prefix;
        printer.printOperand(body.getArgument(iter));
        printer << suffix;

        printer << " = (" << prefix << suffix << ")(arg->arg" << struct_arg_counter << ");\n";
        struct_arg_counter++;
    }
    printer << "$moveInToRegionEnd\n";

    // 输出域
    printer.printRegion(funcOp.region(), /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/false);
}

//============================================================================//
// ReturnOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &state)
{
    // do nothing
    return success();
}

// 输出函数
static void print(sw::ReturnOp returnOp, OpAsmPrinter &printer)
{
    // do nothing
}

//============================================================================//
// MainFuncOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseMainFuncOp(OpAsmParser &parser, OperationState &state)
{
    // 解析函数名称
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), state.attributes))
        return failure();
    
    // 解析func函数
    SmallVector<OpAsmParser::OperandType, 8> entryArgs;
    SmallVector<Type, 8> argTypes;
    SmallVector<Type, 8> resultTypes;
    SmallVector<NamedAttrList, 1> argAttrs;
    SmallVector<NamedAttrList, 1> resultAttrs;

    bool isVariadic;
    auto signatureLocation = parser.getCurrentLocation();
    if (failed(impl::parseFunctionSignature(parser, /*allowVariadic=*/false, 
        entryArgs, argTypes, argAttrs, isVariadic, resultTypes, resultAttrs)))
        return failure();
    if (entryArgs.empty() && !argTypes.empty())
        return parser.emitError(signatureLocation) 
            << "sw.main_func requires named arguments";
    
    Builder &builder = parser.getBuilder();
    auto type = builder.getFunctionType(argTypes, resultTypes);
    state.addAttribute(FuncOp::getTypeAttrName(), TypeAttr::get(type));

    // 解析域
    auto *body = state.addRegion();
    return parser.parseRegion(*body, entryArgs, argTypes);
}

// 打印函数
static void print(sw::MainFuncOp mainFuncOp, OpAsmPrinter &printer)
{
    printer << "$mainModuleBegin\n";
    // 输出头文件
    printer << "#include <athread.h>\n";
    printer << "#include <stdio.h>\n";
    printer << "#include <stdlib.h>\n";
    printer << "#include <math.h>\n";
    printer << "#include <time.h>\n";
    printer << "#include <sys/time.h>\n";
    printer << "#include <string.h>\n";
    printer << "#include <stdint.h>\n";
    printer << "#ifdef SWStenMPI\n";
    printer << "#include \"utils/mpi_lib.h\"\n";
    printer << "#endif\n\n";

    // mpe和spe共享内容的插入点
    printer << "$shareInsertPoint\n";

    // 输出函数
    printer << "void " << mainFuncOp.getName() << "(";
    // 输出参数列表
    ArrayRef<Type> argTypes = mainFuncOp.getType().getInputs();
    Region &body = mainFuncOp.getOperation()->getRegion(0);
    for (int iter = 0; iter < argTypes.size(); iter++) {
        auto elemType = argTypes[iter].cast<mlir::sw::GridType>().getElementType();
        if (elemType.cast<mlir::FloatType>().getWidth() == 64)
            printer << "double ";
        else
            printer << "float ";
        printer.printOperand(body.getArgument(iter));
        auto shape = argTypes[iter].cast<mlir::sw::GridType>().getShape();
        for (int iter_j = 0; iter_j < shape.size(); iter_j++)
            printer << "[" << shape[iter_j] << "]";
        
        if (iter+1 != argTypes.size())
            printer << ", ";
    }
    printer << ")";
    // 输出函数域
    printer.printRegion(mainFuncOp.region(), /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/false);
    printer << "\n$mainModuleEnd\n";                    
}

//============================================================================//
// MainReturnOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseMainReturnOp(OpAsmParser &parser, OperationState &state)
{
    // do nothing
    return success();
}

// 输出函数
static void print(sw::MainReturnOp returnOp, OpAsmPrinter &printer)
{
    // do nothing
}

//============================================================================//
// MainIterationFuncOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseMainIterationFuncOp(OpAsmParser &parser, OperationState &state)
{
    // 解析函数名称
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), state.attributes))
        return failure();
    
    // 解析func函数
    SmallVector<OpAsmParser::OperandType, 8> entryArgs;
    SmallVector<Type, 8> argTypes;
    SmallVector<Type, 8> resultTypes;
    SmallVector<NamedAttrList, 1> argAttrs;
    SmallVector<NamedAttrList, 1> resultAttrs;

    bool isVariadic;
    auto signatureLocation = parser.getCurrentLocation();
    if (failed(impl::parseFunctionSignature(parser, /*allowVariadic=*/false,
            entryArgs, argTypes, argAttrs, isVariadic, resultTypes, resultAttrs)))
        return failure();
    if (entryArgs.empty() && !argTypes.empty())
        return parser.emitError(signatureLocation) 
            << "sw.main_iteration_func requires named arguments";
    
    Builder &builder = parser.getBuilder();
    auto type = builder.getFunctionType(argTypes, resultTypes);
    state.addAttribute(FuncOp::getTypeAttrName(), TypeAttr::get(type));

    // 解析域
    auto *body = state.addRegion();
    return parser.parseRegion(*body, entryArgs, argTypes);
}

// 打印函数
static void print(sw::MainIterationFuncOp mainIterationFuncOp, OpAsmPrinter &printer)
{
    printer << "$mainModuleBegin\n";
    // 输出函数
    printer << "void " << mainIterationFuncOp.getName() << "(";
    // 输出参数列表
    ArrayRef<Type> argTypes = mainIterationFuncOp.getType().getInputs();
    Region &body = mainIterationFuncOp.getOperation()->getRegion(0);
    for (int iter = 0; iter < argTypes.size(); iter++) {
        auto elemType = argTypes[iter].cast<mlir::sw::GridType>().getElementType();
        if (elemType.cast<mlir::FloatType>().getWidth() == 64)
            printer << "double ";
        else
            printer << "float ";
        printer.printOperand(body.getArgument(iter));
        auto shape = argTypes[iter].cast<mlir::sw::GridType>().getShape();
        for (int iter_j = 0; iter_j < shape.size(); iter_j++)
            printer << "[" << shape[iter_j] << "]";
        
        if (iter+1 != argTypes.size())
            printer << ", ";
    }
    printer << ")";
    
    // 输出函数域
    printer.printRegion(mainIterationFuncOp.region(), /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/false);

    printer << "\n$mainModuleEnd\n";
}

//============================================================================//
// MainIterationReturnOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseMainIterationReturnOp(OpAsmParser &parser, OperationState &state)
{
    // do nothing
    return success();
}

// 输出函数
static void print(sw::MainIterationReturnOp mainIterationReturnOp, OpAsmPrinter &printer)
{
    // do nothing
}

//============================================================================//
// launch_func操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseLaunchFuncOp(OpAsmParser &parser, OperationState &state)
{
    FlatSymbolRefAttr kernelNameAttr;

    // 解析kernel名称
    if (parser.parseAttribute(kernelNameAttr, sw::LaunchFuncOp::getKernelAttrName(), state.attributes))
        return failure();
    // 解析参数列表
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    if (succeeded(parser.parseLParen())) { // 解析左括号
        do {
            OpAsmParser::OperandType currentOperand;
            Type currentType;

            if (parser.parseOperand(currentOperand) ||
                parser.parseColonType(currentType))
                return failure();
            
            operands.push_back(currentOperand);
            operandTypes.push_back(currentType);
        } while(succeeded(parser.parseOptionalComma())); // 解析逗号
        // 解析右括号
        if (failed(parser.parseRParen())) // 解析右括号
            return failure();
    }
    // 解析参数类型
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands)))
        return failure();
    return success();
}

// 输出函数
static void print(sw::LaunchFuncOp launchFuncOp, OpAsmPrinter &printer)
{
    auto kernelName = launchFuncOp.getKernelName();
    printer << "struct " << kernelName << "_arg " << kernelName << "_param;\n";

    for (int iter = 0; iter < launchFuncOp.operands().size(); iter++)
        printer << kernelName << "_param.arg" << iter << "=" << launchFuncOp.operands()[iter] << ";\n";
    printer << "athread_spawn(";
    printer << launchFuncOp.getKernelName();
    printer <<", &" << launchFuncOp.getKernelName() << "_param";
    printer << ");\n";
    printer << "athread_join();\n";
}

//============================================================================//
// launch_main_func 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseLaunchMainFuncOp(OpAsmParser &parser, OperationState &state)
{
    // 解析函数名称
    FlatSymbolRefAttr nameAttr;
    if (parser.parseAttribute(nameAttr, sw::LaunchMainFuncOp::getMainFuncAttrName(), state.attributes))
        return failure();

    // 解析参数列表
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    if (failed(parser.parseLParen())) // 解析左括号
        return failure();

    do {
        OpAsmParser::OperandType currentOperand;
        Type currentType;

        if (parser.parseOperand(currentOperand) ||
            parser.parseColonType(currentType))
            return failure();

        operands.push_back(currentOperand);
        operandTypes.push_back(currentType);
    } while(succeeded(parser.parseOptionalComma())); // 解析逗号

    if (failed(parser.parseRParen())) // 解析右括号
        return failure();

    // 解析参数类型
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands)))
        return failure();

    return success();
}

// 打印函数
static void print(sw::LaunchMainFuncOp launchMainFuncOp, OpAsmPrinter &printer)
{
    printer << launchMainFuncOp.getMainFuncName() << "(";

    // 输出参数
    auto operands = launchMainFuncOp.operands();
    for (int iter = 0; iter < operands.size(); iter++) {
        printer << operands[iter];
        if (iter+1 != operands.size())
            printer << ", ";
    }

    printer << ");";
}

//============================================================================//
// launch操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseLaunchOp(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<OpAsmParser::OperandType, 8>arguments;
    // SmallVector<Type, 8> operandTypes;
    SmallVector<Type, 8> argTypes;

    Builder &builder = parser.getBuilder();

    // 解析赋值列表
    if (succeeded(parser.parseLParen())) { // 解析左括号
        do {
            OpAsmParser::OperandType currentArgument, currentOperand;
            Type currentType;

            if (failed(parser.parseRegionArgument(currentArgument))
                || failed(parser.parseEqual())
                || failed(parser.parseOperand(currentOperand))
                || failed(parser.parseColonType(currentType)))
                return failure();
            arguments.push_back(currentArgument);
            operands.push_back(currentOperand);
            argTypes.push_back(currentType);
        } while(succeeded(parser.parseOptionalComma())); // 解析可能存在的逗号(多参数情况下)
        if (failed(parser.parseRParen())) // 解析右括号
            return failure();
    } else {
        return failure();
    }

    // 解析operand参数类型
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, argTypes, loc, state.operands)))
        return failure();

    if (failed(parser.parseColon())) // 解析冒号
        return failure();

    // 记录参数数量
    unsigned int origin_num = arguments.size();
    // 解析cacheRead属性信息
    if (failed(parseAttributions(parser, sw::LaunchOp::getCacheReadAttrName(), arguments, argTypes)))
        return failure();
    unsigned int cacheRead_num = arguments.size() - origin_num;
    state.addAttribute(sw::LaunchOp::getCacheReadAttrNumName(), builder.getI64IntegerAttr(cacheRead_num));

    // 解析cacheWrite属性信息
    if (failed(parseAttributions(parser, sw::LaunchOp::getCacheWriteAttrName(), arguments, argTypes)))
        return failure();
    unsigned int cacheWrite_num = arguments.size() - origin_num - cacheRead_num;
    state.addAttribute(sw::LaunchOp::getCacheWriteAttrNumName(), builder.getI64IntegerAttr(cacheWrite_num));

    // 解析region域
    Region *body = state.addRegion();
    if (failed(parser.parseRegion(*body, arguments, argTypes)))
        return failure();

    return success();
}

// 输出函数
static void print(sw::LaunchOp launchOp, OpAsmPrinter &printer)
{
    // launchOP只起到临时表示作用, 后续会执行outline操作, 生成相应的launch_func和func,
    // 故此处输出要求并不严格, 不生成为C语言
    printer << launchOp.getOperationName() << ' ';
    
    // 输出参数列表
    SmallVector<Value, 10> operands = launchOp.getOperands();
    if (!launchOp.region().empty() && !operands.empty()) {
        Block &body = launchOp.region().front();
        printer << "(";
        llvm::interleaveComma(
            llvm::seq<int>(0, operands.size()), printer, [&](int i) {
                printer << body.getArgument(i) << " = " << operands[i] << " : "
                        << operands[i].getType();
            }
        );
        printer << ") ";
    }

    // 输出cacheRead和cacheWrite属性信息
    printer << "cacheRead(";
    printAttributions(printer, launchOp.getCacheReadAttributions());
    printer << ") ";
    printer << "cacheWrite(";
    printAttributions(printer, launchOp.getCacheWriteAttributions());
    printer << ") ";

    // 输出region域
    printer.printRegion(launchOp.region(), /*printEntryBlockArgs=*/false);
}

//============================================================================//
// TerminatorOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseTerminatorOp(OpAsmParser &parser, OperationState &state)
{
    // do nothing
    return success();
}

// 输出函数
static void print(sw::TerminatorOp terminatorOp, OpAsmPrinter &printer)
{
    // do nothing
}

//============================================================================//
// for 操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseForOp(OpAsmParser &parser, OperationState &state)
{
    auto &builder = parser.getBuilder();
    OpAsmParser::OperandType inductionVariable, lb, ub, step;
    
    // 解析迭代变量以及之后的 '='
    if (failed(parser.parseRegionArgument(inductionVariable))
        || failed(parser.parseEqual()))
        return failure();

    // 解析循环边界, 以及跨步大小
    if (failed(parser.parseOperand(lb))
        || failed(parser.parseKeyword("to"))
        || failed(parser.parseOperand(ub))
        || failed(parser.parseKeyword("step"))
        || failed(parser.parseOperand(step)))
        return failure();
    // 解析迭代变量类型
    Type inductionVariableType;
    if (failed(parser.parseColonType(inductionVariableType)))
        return failure();

    if (failed(parser.resolveOperand(lb, inductionVariableType, state.operands))
        || failed(parser.resolveOperand(ub, inductionVariableType, state.operands))
        || failed(parser.resolveOperand(step, inductionVariableType, state.operands)))
        return failure();

    // 解析region域
    Region *body = state.addRegion();
    if (failed(parser.parseRegion(*body, inductionVariable, inductionVariableType)))

    return success();
}

// 输出函数
static void print(sw::ForOp forOp, OpAsmPrinter &printer)
{
    // 定义循环变量
    int integer_width = forOp.getInductionVar().getType().cast<mlir::IntegerType>().getWidth();
    if (integer_width == 32)
        printer << "int ";
    else
        printer << "long ";
    printer << forOp.getInductionVar() << ";\n";
    // 输出for循环头
    printer << "for ( ";
    printer << forOp.getInductionVar() << " = " << forOp.lowerBound() << "; ";
    printer << forOp.getInductionVar() << " < " << forOp.upperBound() << "; ";
    printer << forOp.getInductionVar() << " += " <<forOp.step() << " )";
    
    // 输出循环体
    printer.printRegion(forOp.region(), /*printEntryBlockArgs=*/false);
}

//============================================================================//
// YieldOp 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &state)
{
    // do nothing
    return success();
}

// 输出函数
static void print(sw::YieldOp yieldOp, OpAsmPrinter &printer)
{
    // do nothing
}

//============================================================================//
// load操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    SmallVector<Type, 8> resultTypes;
    OpAsmParser::OperandType currentOperand;
    Type currentType;

    // 解析参数名
    if (failed(parser.parseOperand(currentOperand)))
        return failure();
    operands.push_back(currentOperand);

    // 解析偏移量
    if (succeeded(parser.parseLSquare())) { // 解析左括号
        do {
            if (failed(parser.parseOperand(currentOperand)))
                return failure();
            operands.push_back(currentOperand);
        } while(succeeded(parser.parseOptionalComma())); // 解析可能存在的逗号

        if (failed(parser.parseRSquare())) // 解析右括号
            return failure();
    } else {
        return failure();
    }

    // 解析参数类型
    // 解析冒号
    if (failed(parser.parseColon()))
        return failure();
    if (succeeded(parser.parseLParen())) { // 解析左括号
        // 解析矩阵的类型
        if (failed(parser.parseType(currentType)))
            return failure();
        operandTypes.push_back(currentType);

        // 解析偏移量类型
        if (failed(parser.parseComma())
            || failed(parser.parseType(currentType)))
            return failure();
        for (int iter = 0 ; iter < operands.size()-1; iter++)
            operandTypes.push_back(currentType);

        if (failed(parser.parseRParen())) // 解析右括号
            return failure();
    }

    // 解析结果类型
    if (failed(parser.parseArrowTypeList(resultTypes)))
        return failure();

    // 解析参数及结果
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands))
        || failed(parser.addTypesToList(resultTypes, state.types)))
        return failure();

    return success();
}

// 输出函数
static void print(sw::LoadOp loadOp, OpAsmPrinter &printer)
{
    printer << loadOp.input();
    for(int iter = 0; iter < loadOp.pos().size(); iter ++)
        printer << "[" << loadOp.pos()[iter] << "]";
    printer << ";";
    if (loadOp.res().getType().cast<mlir::FloatType>().getWidth() == 64)
        printer << "$moveToHead<-double";
    else
        printer << "$moveToHead<-float";
}

//============================================================================//
// store操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &state)
{
    // 解析参数列表
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    OpAsmParser::OperandType currentOperand;
    Type currentType;
    
    do {
        if (failed(parser.parseOperand(currentOperand)))
            return failure();

        operands.push_back(currentOperand);
    } while (succeeded(parser.parseOptionalComma())); // 解析逗号

    // 解析偏移量
    if (succeeded(parser.parseLSquare())) { // 解析左括号
        do {
            if (failed(parser.parseOperand(currentOperand)))
                return failure();
            operands.push_back(currentOperand);
        } while(succeeded(parser.parseOptionalComma())); // 解析可能存在的逗号

        if (failed(parser.parseRSquare())) // 解析右括号
            return failure();
    } else {
        return failure();
    }
    
    // 解析参数类型
    // 解析元素类型
    if (failed(parser.parseColonType(currentType)))
        return failure();
    operandTypes.push_back(currentType);\

    if (failed(parser.parseOptionalKeyword("to")))
        return failure();
    if (succeeded(parser.parseLParen())) { // 解析左括号
        // 解析矩阵类型
        if (failed(parser.parseType(currentType)))
            return failure();
        operandTypes.push_back(currentType);

        // 解析偏移量类型
        if (failed(parser.parseComma())
            || failed(parser.parseType(currentType)))
            return failure();
        for (int iter = 0; iter < operands.size()-2; iter++)
            operandTypes.push_back(currentType);

        if (failed(parser.parseRParen())) // 解析右括号
            return failure();
    } else {
        return failure();
    }

    // 解析参数及其类型并添加到state中
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands)))
        return failure();
    
    return success();
}

// 输出函数
static void print(sw::StoreOp storeOp, OpAsmPrinter &printer)
{
    SmallVector<Value, 10> operands = storeOp.getOperands();
    printer << operands[1];
    for (int iter = 0; iter < storeOp.pos().size(); iter ++)
        printer << "[" << storeOp.pos()[iter] << "]";
    printer << " = " << operands[0] << ";";
    
}

//============================================================================//
// constant函数相关操作定义
//============================================================================//
// 解析函数
static ParseResult parseConstantOp(OpAsmParser &parser, OperationState &result)
{
    // 解析常量部分
    Attribute valueAttr;
    if (failed(parser.parseAttribute(valueAttr, sw::ConstantOp::getValueAttrName(), result.attributes)))
        return failure();

    // 解析常量的类型
    Type type;
    if (!valueAttr.isa<SymbolRefAttr>())
        type = valueAttr.getType();
    else if (failed(parser.parseColonType(type)))
        return failure();

    // 将常量的类型设置为返回值的类型
    return parser.addTypeToList(type, result.types);
}

// 输出函数
static void print(sw::ConstantOp constantOp, OpAsmPrinter &printer)
{
    auto elemType = constantOp.value().getType();
    if (elemType.isa<mlir::IntegerType>()) {
        printer << constantOp.value().cast<mlir::IntegerAttr>().getValue() << ";";
        if (elemType.cast<mlir::IntegerType>().getWidth() == 64)
            printer << "$moveToHead<-long";
        else
            printer << "$moveToHead<-int";
    } else if (elemType.isa<mlir::FloatType>()) {
        if (elemType.cast<mlir::FloatType>().getWidth() == 64) {
            printer << constantOp.value().cast<FloatAttr>().getValue().convertToDouble() << ";";
            printer << "$moveToHead<-double";
        } else {
            printer << constantOp.value().cast<FloatAttr>().getValue().convertToFloat() << ";";
            printer << "$moveToHead<-float";
        }
    } else {
        printer << "$error";
    }
}

// fold函数
OpFoldResult sw::ConstantOp::fold(ArrayRef<Attribute> operands) 
{
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

//============================================================================//
// getID操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseGetIDOp(OpAsmParser &parser, OperationState &result)
{
    // 解析类型
    Type type;
    if (failed(parser.parseColonType(type)))
        return failure();
    
    // 将类型设置为返回值的类型
    return parser.addTypeToList(type, result.types);
}

// 输出函数
static void print(sw::GetIDOp getIDOp, OpAsmPrinter &printer)
{
    auto elemType = getIDOp.res().getType();
    if (elemType.isa<mlir::IntegerType>()) {
        printer << "athread_get_id(-1);";
        if (elemType.cast<mlir::IntegerType>().getWidth() == 64)
            printer << "$moveToHead<-long";
        else 
            printer << "$moveToHead<-int";
    } else {
        printer << "$error";
    }
}

//============================================================================//
// addf 相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseAddfOp(OpAsmParser &parser, OperationState &result)
{
    return parseMathOp(parser, result);
}

// 输出函数
static void print(sw::AddfOp addfOp, OpAsmPrinter &printer)
{
    printer << addfOp.lhs() << " + " << addfOp.rhs() << ";";

    // 输出结果类型
    if (addfOp.res().getType().cast<mlir::FloatType>().getWidth() == 64)
        printer << "$moveToHead<-double";
    else
        printer << "$moveToHead<-float";
}

// fold函数
OpFoldResult sw::AddfOp::fold(ArrayRef<Attribute> operands)
{
    return constFoldBinaryOp<FloatAttr>(
        operands, [](APFloat a, APFloat b) { return a + b; }
    );
}

//============================================================================//
// subf 操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseSubfOp(OpAsmParser &parser, OperationState &result)
{
    return parseMathOp(parser, result);
}

// 输出函数
static void print(sw::SubfOp subfOp, OpAsmPrinter &printer)
{
    printer << subfOp.lhs() << " - " << subfOp.rhs() << ";";

    // 输出结果类型
    if (subfOp.res().getType().cast<mlir::FloatType>().getWidth() == 64)
        printer << "$moveToHead<-double";
    else
        printer << "$moveToHead<-float";
}

// fold函数
OpFoldResult sw::SubfOp::fold(ArrayRef<Attribute> operands)
{
    return constFoldBinaryOp<FloatAttr>(
        operands, [](APFloat a, APFloat b) { return a - b; }
    );
}

//============================================================================//
// mulf 操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseMulfOp(OpAsmParser &parser, OperationState &result)
{
    return parseMathOp(parser, result);
}

// 输出函数
static void print(sw::MulfOp mulfOp, OpAsmPrinter &printer)
{
    printer << mulfOp.lhs() << " * " << mulfOp.rhs() << ";";

    // 输出结果类型
    if (mulfOp.res().getType().cast<mlir::FloatType>().getWidth() == 64)
        printer << "$moveToHead<-double";
    else
        printer << "$moveToHead<-float";
}

// fold函数
OpFoldResult sw::MulfOp::fold(ArrayRef<Attribute> operands)
{
    return constFoldBinaryOp<FloatAttr> (
        operands, [](APFloat a, APFloat b) { return a * b; }
    );
}

//============================================================================//
// divf操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseDivfOp(OpAsmParser &parser, OperationState &result)
{
    return parseMathOp(parser, result);
}

// 输出函数
static void print(sw::DivfOp divfOp, OpAsmPrinter &printer)
{
    printer << divfOp.lhs() << " / " << divfOp.rhs() << ";";

    // 输出结果类型
    if (divfOp.res().getType().cast<mlir::FloatType>().getWidth() == 64)
        printer << "$moveToHead<-double";
    else
        printer << "$moveToHead<-float";
}

//============================================================================//
// addi操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseAddiOp(OpAsmParser &parser, OperationState &state)
{
    return parseMathOp(parser, state);
}

// 输出函数
static void print(sw::AddiOp addiOp, OpAsmPrinter &printer)
{
    printer << addiOp.lhs() << " + " << addiOp.rhs() << ";";

    // 输出结果类型
    if (addiOp.res().getType().cast<mlir::IntegerType>().getWidth() == 64)
        printer << "$moveToHead<-long";
    else
        printer << "$moveToHead<-int";
}

// fold函数
OpFoldResult sw::AddiOp::fold(ArrayRef<Attribute> operands)
{
    // addi(x, 0) -> x
    if (matchPattern(lhs(), m_Zero()))
        return lhs();
    
    return constFoldBinaryOp<IntegerAttr>(
        operands, [](APInt a, APInt b) { return a + b; }
    );
}

//============================================================================//
// subi操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseSubiOp(OpAsmParser &parser, OperationState &state)
{
    return parseMathOp(parser, state);
}

// 输出函数
static void print(sw::SubiOp subiOp, OpAsmPrinter &printer)
{
    printer << subiOp.lhs() << " - " << subiOp.rhs() << ";";

    // 输出结果类型
    if (subiOp.res().getType().cast<mlir::IntegerType>().getWidth() == 64)
        printer << "$moveToHead<-long";
    else
        printer << "$moveToHead<-int";
}

// fold函数
OpFoldResult sw::SubiOp::fold(ArrayRef<Attribute> operands)
{
    // subi(x, x) -> 0
    if (getOperand(0) == getOperand(1))
        return Builder(getContext()).getZeroAttr(getType());
    // subi(x,0) -> x
    if (matchPattern(rhs(), m_Zero()))
        return lhs();
    
    return constFoldBinaryOp<IntegerAttr>(
        operands, [](APInt a, APInt b) { return a - b; }
    );
}

//============================================================================//
// muli操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseMuliOp(OpAsmParser &parser, OperationState &state)
{
    return parseMathOp(parser, state);
}

// 输出函数
static void print(sw::MuliOp muliOp, OpAsmPrinter &printer)
{
    printer << muliOp.lhs() << " * " << muliOp.rhs() << ";";

    // 输出结果类型
    if (muliOp.res().getType().cast<mlir::IntegerType>().getWidth() == 64)
        printer << "$moveToHead<-long";
    else
        printer << "$moveToHead<-int";
}

// fold函数
OpFoldResult sw::MuliOp::fold(ArrayRef<Attribute> operands)
{
    // muli(x, 0) -> 0
    if (matchPattern(rhs(), m_Zero()))
        return rhs();
    
    // muli(x, 1) -> x
    if (matchPattern(rhs(), m_One()))
        return getOperand(0);
    
    return constFoldBinaryOp<IntegerAttr>(
        operands, [](APInt a, APInt b) { return a * b; }
    );
}

//============================================================================//
// memcpyToLDM操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseMemcpyToLDMOp(OpAsmParser &parser, OperationState &state)
{
    return parseMemcpyOp(parser, state);
}
// 输出函数
static void print(sw::MemcpyToLDMOp memcpyToLDMOp, OpAsmPrinter &printer)
{
    auto index_size = memcpyToLDMOp.mem_index().size();
    // 使用按面加载的方法, 索引为三维的情况下需要为最高维度加一个迭代变量
    printer << "DMA_get(" << memcpyToLDMOp.mem_addr();

    // 输出MEM部分的索引
    for (int iter = 0; iter < index_size; iter ++) {
        if (iter == 0 && index_size == 3)
            printer << "[" << memcpyToLDMOp.mem_index()[0] << "+z_iter" << "]";
        else
            printer << "[" << memcpyToLDMOp.mem_index()[iter] << "]";
    }

    printer << ", " << memcpyToLDMOp.ldm_addr();

    // 输出LDM索引部分
    for (int iter = 0; iter < index_size; iter ++) {
        if (iter == 0 && index_size == 3)
            printer << "[z_iter]";
        else
            printer << "[0]";
    }
    printer << ", " << memcpyToLDMOp.z_dim() << ", ";

    // 获取数据类型 
    auto elemType = memcpyToLDMOp.mem_addr().getType().cast<mlir::sw::GridType>().getElementType();
    auto TypeWidth = (elemType.cast<mlir::FloatType>().getWidth() == 64) ?
                        "*sizeof(double)" : "*sizeof(float)";

    printer << memcpyToLDMOp.cnt() << TypeWidth << ", ";
    printer << memcpyToLDMOp.stride() << TypeWidth << ", ";
    printer << memcpyToLDMOp.bsize() << TypeWidth << ");";
}

//============================================================================//
// memcpyToMEM操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseMemcpyToMEMOp(OpAsmParser &parser, OperationState &state)
{
    return parseMemcpyOp(parser, state);
}
// 输出函数
static void print(sw::MemcpyToMEMOp memcpyToMEMOp, OpAsmPrinter &printer)
{
    auto index_size = memcpyToMEMOp.mem_index().size();
    // 使用按面加载的方法, 索引为三维的情况下需要为最高维度加一个迭代变量
    printer << "DMA_put(" << memcpyToMEMOp.ldm_addr();

    // 输出ldm索引部分
    for (int iter = 0; iter < index_size; iter ++) {
        if (iter == 0 && index_size == 3)
            printer << "[z_iter]";
        else
            printer << "[0]";
    }

    printer << ", " << memcpyToMEMOp.mem_addr();

    // 输出mem索引部分
    for (int iter = 0; iter < index_size; iter ++) {
        if (iter == 0 && index_size == 3)
            printer << "[" << memcpyToMEMOp.mem_index()[0] << "+z_iter" << "]";
        else
            printer << "[" << memcpyToMEMOp.mem_index()[iter] << "]";
    }

    printer << ", " << memcpyToMEMOp.z_dim() << ", ";

    // 获取数据类型
    auto elemType = memcpyToMEMOp.mem_addr().getType().cast<mlir::sw::GridType>().getElementType();
    auto TypeWidth = (elemType.cast<mlir::FloatType>().getWidth() == 64) ?
                    "*sizeof(double)" : "*sizeof(float)";
    
    printer << memcpyToMEMOp.cnt() << TypeWidth << ", ";
    printer << memcpyToMEMOp.stride() << TypeWidth <<", ";
    printer << memcpyToMEMOp.bsize() << TypeWidth << ");";
}

//============================================================================//
// alloc操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseAllocOp(OpAsmParser &parser, OperationState &state)
{
    Type type;
    if (failed(parser.parseColonType(type)))
        return failure();

    // 将类型设置为返回值类型
    return parser.addTypeToList(type, state.types);
}

// 输出函数
static void print(sw::AllocOp allocOp, OpAsmPrinter &printer)
{
    auto resultType = allocOp.getResult().getType().cast<sw::GridType>();
    auto shape = resultType.getShape();
    auto elemType = resultType.getElementType();
    auto elemTypeString = (elemType.cast<mlir::FloatType>().getWidth() == 64) ?
                            "double" : "float";
    int shapeSize = 1;
    for (int i = 0; i < shape.size(); i++) {
        shapeSize *= shape[i];
    }
    auto shapeString = std::to_string(shapeSize);

    printer << "malloc(sizeof(" << elemTypeString << ")*" << shapeString << ");";
    printer << "$moveToHead<-" << elemTypeString << " *";
}

//============================================================================//
// dealloc操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseDeAllocOp(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::OperandType, 8> operands;
    SmallVector<Type, 8> operandTypes;
    OpAsmParser::OperandType currentOperand;
    Type currentOperandType;

    if (failed(parser.parseOperand(currentOperand)))
        return failure();
    if (failed(parser.parseColonType(currentOperandType)))
        return failure();

    operands.push_back(currentOperand);
    operandTypes.push_back(currentOperandType);
    
    // 解析参数和结果类型到state中
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands)))
        return failure();
}

// 输出函数
static void print(sw::DeAllocOp deAllocOp, OpAsmPrinter &printer)
{
    printer << "free(" << deAllocOp.input() << ");";
}

//============================================================================//
// getMpiRank操作相关函数实现
//============================================================================//
// 解析函数
static ParseResult parseGetMpiRankOp(OpAsmParser &parser, OperationState &state)
{
    // 此函数返回值一定是int类型
    Builder &builder = parser.getBuilder();
    Type type = builder.getI32Type();

    // 将返回值的类型设置为int
    return parser.addTypeToList(type, state.types);
}

// 输出函数
static void print(sw::GetMpiRankOp getMpiRankOp, OpAsmPrinter &printer)
{
    printer << "mpiGetMyRank();$moveToHead<-int";
}

//============================================================================//
// mpiExchangeHalo操作相关函数
//============================================================================//
// 解析函数
static ParseResult parseMpiExchangeHaloOp(OpAsmParser &parser, OperationState &state)
{
    SmallVector<OpAsmParser::OperandType, 2> operands;
    SmallVector<Type, 2> operandTypes;
    OpAsmParser::OperandType currentOperand;
    Type currentType;

    // 解析待交换数据的数组
    if (failed(parser.parseOperand(currentOperand)))
        return failure();
    operands.push_back(currentOperand);

    // 解析当前进程rank
    if (failed(parser.parseComma())
        || failed(parser.parseOperand(currentOperand)))
        return failure();
    operands.push_back(currentOperand);

    // 解析mpiTile
    if (failed(parser.parseColon()) || failed(parseKeywordAttr(parser, state, "mpiTile")))
        return failure();

    // 解析mpiHalo, 此处分为mpiHaloL和mpiHaloU
    Attribute mpiHaloLAttr, mpiHaloUAttr;
    // 解析关键字及左括号
    if (failed(parser.parseKeyword("mpiHalo"))
        || failed(parser.parseLParen()))
        return failure();
    // 解析mpiHaloL
    if (failed(parser.parseAttribute(mpiHaloLAttr, sw::MpiExchangeHaloOp::getMpiHaloLName(), state.attributes)))
        return failure();
    // 解析冒号
    if (failed(parser.parseColon()))
        return failure();
    // 解析mpiHaloU
    if (failed(parser.parseAttribute(mpiHaloUAttr, sw::MpiExchangeHaloOp::getMpiHaloUName(), state.attributes)))
        return failure();
    // 解析右括号
    if (failed(parser.parseRParen()))
        return failure();
    
    // 解析待交换数组的类型
    if (failed(parser.parseColon()) || failed(parser.parseType(currentType)))
        return failure();
    operandTypes.push_back(currentType);

    // 由于rank一定是int类型, 此处直接指定
    Builder &builder = parser.getBuilder();
    currentType = builder.getI32Type();
    operandTypes.push_back(currentType);

    // 执行参数解析
    auto loc = parser.getCurrentLocation();
    if (failed(parser.resolveOperands(operands, operandTypes, loc, state.operands)))
        return failure();

    return success();
}

// 输出函数
static void print(sw::MpiExchangeHaloOp mpiExchangeHaloOp, OpAsmPrinter &printer)
{
    // 获取待交换数组的维度和数据类型
    std::string arrayDimString, arrayElemTypeString;
    auto arrayType = mpiExchangeHaloOp.dataArray().getType().cast<mlir::sw::GridType>();
    auto arrayDim = arrayType.getRank();
    auto arrayShape = arrayType.getShape();
    auto arrayElemType = arrayType.getElementType().cast<mlir::FloatType>();
    // 生成维度字符串
    if (arrayDim == 2) {
        arrayDimString = "2D";
    } else if (arrayDim == 3) {
        arrayDimString = "3D";
    } else {
        printer << "$error";
    }
    // 生成类型字符串
    arrayElemTypeString = (arrayElemType.getWidth() == 64) ? "double" : "float";

    // 构造输出
    printer << "exchange_halo_" << arrayDimString << "_" << arrayElemTypeString << "(";
    // 输出待交换数组名及其维度信息
    printer << mpiExchangeHaloOp.dataArray() << ", ";
    for (int iter = 0; iter < arrayShape.size(); iter ++)
        printer << arrayShape[iter] << ", ";
    // 输出mpiTile信息
    auto mpiTileAttr = mpiExchangeHaloOp.mpiTile();
    for (int iter = 0; iter < mpiTileAttr.size(); iter ++) {
        printer << mpiTileAttr[iter].cast<mlir::IntegerAttr>().getInt();
        printer << ", ";
    }
    // 输出mpiHalo信息, 分为mpiHaloL和mpiHaloU, 这两者的维度是相同的
    auto mpiHaloLAttr = mpiExchangeHaloOp.mpiHaloL();
    auto mpiHaloUAttr = mpiExchangeHaloOp.mpiHaloU();
    for (int iter = 0; iter < mpiHaloLAttr.size(); iter ++) {
        printer << mpiHaloLAttr[iter].cast<mlir::IntegerAttr>().getInt();
        printer << ", ";
        printer << mpiHaloUAttr[iter].cast<mlir::IntegerAttr>().getInt();
        printer << ", ";
    }
    printer << mpiExchangeHaloOp.rank() << ");";
}

namespace mlir {
namespace sw {
#define GET_OP_CLASSES
#include "Dialect/SW/SWOps.cpp.inc"
}
}