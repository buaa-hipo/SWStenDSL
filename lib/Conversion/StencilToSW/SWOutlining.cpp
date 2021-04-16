/**
 * @file SWOutlining.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 将转化后的launchOp操作移动到外部使之成为相应的从核函数
 * @version 0.1
 * @date 2021-04-09
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/RegionUtils.h>

#include "Dialect/SW/SWDialect.h"
#include "Dialect/SW/SWOps.h"
#include "Conversion/StencilToSW/Passes.h"
#include "PassDetail.h"

using namespace mlir;

// 将sw::LaunchOp的域移动到一个spe函数中, 并将sw.terminator操作替换为sw.return.
static sw::FuncOp outlineKernelFuncImpl(sw::LaunchOp launchOp, StringRef kernelFnName)
{
    Location loc = launchOp.getLoc();
    ValueRange operands = launchOp.getOperands();
    // 创建builder但不指定插入位置, 插入过程将在符号表的管理下分别执行.
    OpBuilder builder(launchOp.getContext());
    Region &launchOpBody = launchOp.region();

    // 创建sw::func操作
    SmallVector<Type, 10> kernelOperandTypes;
    for (Value operand : operands) {
        kernelOperandTypes.push_back(operand.getType());
    }
    FunctionType type =
        FunctionType::get(kernelOperandTypes, {}, launchOp.getContext());
    auto outlinedFunc = builder.create<sw::FuncOp>(loc, kernelFnName, type);
    // 删除操作自行生成的Block
    outlinedFunc.region().getBlocks().back().erase();

    BlockAndValueMapping map;   // 无需重新映射
    // 将sw.launch操作的域复制到sw.func的域
    Region &outlinedFuncBody = outlinedFunc.region();
    launchOpBody.cloneInto(&outlinedFuncBody, map);
    // 替换终结符
    outlinedFunc.walk([](sw::TerminatorOp op) {
        OpBuilder replacer(op);
        replacer.create<sw::ReturnOp>(op.getLoc());
        op.erase();
    });

    return outlinedFunc;
}

// 在launchOp的位置插入LaunchFuncOp, 并删除launchOp
static void convertToLaunchFuncOp(sw::LaunchOp launchOp,
                                    sw::FuncOp kernelFunc) {
    OpBuilder builder(launchOp);
    ValueRange operands = launchOp.getOperands();
    builder.create<sw::LaunchFuncOp>(
        launchOp.getLoc(), builder.getSymbolRefAttr(kernelFunc.getName()), operands);
    launchOp.erase();
}

namespace {
/**
 * @brief 该pass负责将launchOp的域转化为相应的spe函数, 并将launchOp替换为launchFuncOp
 * 
 */
class SWOutliningPass : public SWOutliningPassBase<SWOutliningPass> {
public:
    void runOnOperation() override {
        SymbolTable symbolTable(getOperation());
        bool modified = false;
        for (auto func : getOperation().getOps<sw::MainFuncOp>()) {
            // 将生成的spe func插入到当前func的前面
            Block::iterator insertPt(func.getOperation());
            auto funcWalkResult = func.walk([&](sw::LaunchOp op) {
                std::string kernelFnName = 
                    Twine(op.getParentOfType<sw::MainFuncOp>().getName(), "_kernel").str();

                SmallVector<Type, 8> cacheReadAttr;
                SmallVector<Type, 8> cacheWriteAttr;
                for (auto elem : op.getCacheReadAttributions()) {
                    cacheReadAttr.push_back(elem.getType());
                }
                for (auto elem : op.getCacheWriteAttributions()) {
                    cacheWriteAttr.push_back(elem.getType());
                }
                sw::FuncOp outlinedFunc = 
                    outlineKernelFuncImpl(op, kernelFnName);
                auto kernelModule = createKernelModule(outlinedFunc, symbolTable, 
                    cacheReadAttr, cacheWriteAttr);
                symbolTable.insert(kernelModule, insertPt);

                // 删除原有的launch操作, 并替换为launch_func操作
                convertToLaunchFuncOp(op, outlinedFunc);

                return WalkResult::advance();
            });
            if (funcWalkResult.wasInterrupted())
                return signalPassFailure();
        }
    }

private:
    // 创建并返回一个包含kernelFunc的sw::ModuleOp
    sw::ModuleOp createKernelModule(sw::FuncOp kernelFunc,
                                        const SymbolTable &parentSymbolTable,
                                        ArrayRef<Type> cacheReadAttr,
                                        ArrayRef<Type> cacheWriteAttr) {
        auto context = getOperation().getContext();
        OpBuilder builder(context);
        OperationState state(kernelFunc.getLoc(), sw::ModuleOp::getOperationName());

        sw::ModuleOp::build(builder, state, kernelFunc.getName(), cacheReadAttr, cacheWriteAttr);
        auto kernelModule = cast<sw::ModuleOp>(Operation::create(state));
        // 在kernelModule的末尾插入终结符号
        builder.setInsertionPointToEnd(kernelModule.getBody());
        builder.create<sw::ModuleEndOp>(kernelFunc.getLoc());
        SymbolTable symbolTable(kernelModule);
        symbolTable.insert(kernelFunc);

        // 用module中的cacheRead和cacheWrite替换kernelFunc中对cacheRead和cacheWrite
        // Argument的引用, 并删除kernelFunc的cacheRead和cacheWrite Argument
        int kernelFuncOperandsNum = kernelFunc.getNumFuncArguments();
        // 标准的func中是没有cacheRead和cacheWrite属性的, 之间拷贝body的时候未移除, 为权宜之计
        int kernelFuncCacheReadWriteAttrNum = 
            kernelFunc.region().front().getNumArguments() - kernelFuncOperandsNum;
        auto kernelModuleArgument = kernelModule.region().front().getArguments();
        auto kernelFuncArgument = kernelFunc.region().front().getArguments();
        for (int i = 0; i < kernelFuncCacheReadWriteAttrNum; i++) {
            Value moduleArg = kernelModuleArgument[i];
            Value funcArg = kernelFuncArgument[kernelFuncOperandsNum + i];
            funcArg.replaceAllUsesWith(moduleArg);
        }
        // 删除func中的cacheRead和cacheWrite属性
        Block *funcBody = &(kernelFunc.region().front());
        for (int i = kernelFuncCacheReadWriteAttrNum-1; i >=0; i--) {
            int index = i + kernelFuncOperandsNum;
            funcBody->eraseArgument(index);
        }

        SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
        while (!symbolDefWorklist.empty()) {
            if (Optional<SymbolTable::UseRange> symbolUses =
                    SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
                for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
                    StringRef symbolName = 
                        symbolUse.getSymbolRef().cast<FlatSymbolRefAttr>().getValue();
                    if (symbolTable.lookup(symbolName))
                        continue;
                    
                    Operation *symbolDefClone = 
                        parentSymbolTable.lookup(symbolName)->clone();
                    symbolDefWorklist.push_back(symbolDefClone);
                    symbolTable.insert(symbolDefClone);
                }
            }
        }

        return kernelModule;
    }
};
} // end of anonymous namespace

std::unique_ptr<Pass> mlir::createSWOutliningPass() {
    return std::make_unique<SWOutliningPass>();
}
