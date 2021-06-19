/**
 * @file stenCC.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 主程序
 * @version 0.1
 * @date 2021-02-27
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AsmState.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Parser.h>

#include "Parser/Parser.h"
#include "Parser/MLIRGen.h"

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/SW/SWDialect.h"
#include "Conversion/StencilToSW/Passes.h"

using namespace mlir;
using namespace swsten;

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input SWStenDSL file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
    enum InputType {SWSten, MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(SWSten), cl::desc("Decided the kind of input desired"), 
    cl::values(clEnumValN(SWSten, "swsten", "load the input file as a SWSten source file")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

// 自定义选项
namespace {
enum Action {
    None,
    DumpAST,
    DumpSW
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of outpput desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpSW, "sw", "output the SW dump"))
);

// 解析输入的文件, 并构造抽象语法树, 如果发生错误则返回nullptr
std::unique_ptr<swsten::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }

    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);

    return parser.parseModule();
}

int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a SWSten AST when the input is MLIR\n";
        return -1;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return -1;
    
    dump(*moduleAST);

    return 0;
}

// 加载文件
int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    // 处理'sten'输入
    if (inputType == SWSten &&
        llvm::StringRef(inputFilename).endswith(".dsl")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return -1;
        module = mlirGen(context, *moduleAST);
        return !module ? 1 : 0;
    }

    // 否则, 输入'.mlir'
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }

    // 解析输入的mlir文件
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return -1;
    }

    return 0;
}

// 加载并处理输入文件, 生成相应的MLIR
int loadAndProcessMLIR(mlir::MLIRContext &context,
                        mlir::OwningModuleRef &module) {
    // 加载文件
    if(int error = loadMLIR(context, module))
        return error;
    
    mlir::PassManager pm(&context);
    // 应用命令行传递过来的通用选项
    applyPassManagerCLOptions(pm);

    // 检查是否需要下降到sw Dialect
    bool isLoweringToSW = emitAction == Action::DumpSW;

    if (isLoweringToSW) {
        pm.addPass(mlir::createConvertStencilToSWPass());
        pm.addPass(mlir::createSWOutliningPass());
    }

    if (mlir::failed(pm.run(*module)))
        return -1;
    return 0;
}

int main(int argc, char *argv[])
{
    registerAllDialects();
    registerAllPasses();

    // 注册命令行选项
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "SWSten compiler\n");

    if (emitAction == Action::DumpAST)
        return dumpAST();
    
    // 如果不是打印AST, 则需要将其转换为MLIR
    mlir::MLIRContext context(/*loadAllDialects=*/false);
    // 加载stencil Dialect
    context.getOrLoadDialect<mlir::stencil::StencilDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::sw::SWDialect>();

    mlir::OwningModuleRef module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;

    // 输出sw Dialect, 此部分输出的是类C语言
    module->dump();

    return 0;
}