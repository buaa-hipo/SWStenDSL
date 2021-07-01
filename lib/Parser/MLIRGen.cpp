/**
 * @file MLIRGen.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief MLIR dump
 * @version 0.1
 * @date 2021-06-12
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/raw_ostream.h>
#include <numeric>
#include <vector>
#include <set>
#include <map>

#include "Parser/MLIRGen.h"
#include "Parser/AST.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilOps.h"

using namespace mlir;
using namespace swsten;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

// 从SWStenDSL生成stencil dialect方言
class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    // 将AST转化为MLIR
    mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
        // 创建空的MLIR module
        theModule = ModuleOp::create(builder.getUnknownLoc());

        // 由于一个module中只包含一个stencil定义, 此处直接处理
        auto stencilAST = moduleAST.getStencil();
        auto stencil = mlirGen(*stencilAST);
        if (stencil.size() == 0)
            return nullptr;

        // stencil部分可能返回多个func (多次迭代情况下会返回两个func, 而单次迭代则只会返回一个func)
        for (int i= 0; i < stencil.size(); i++)
            theModule.push_back(stencil[i]);

        // 在构建完成后对module进行检查
        if (failed(mlir::verify(theModule))) {
            theModule.emitError("module verfication error");
            return nullptr;
        }

        return theModule;        
    }

private:
    // 一个Module对应一个SWStenDSL文件, 包含一个stencil定义
    mlir::ModuleOp theModule;

    // builder 是创建MLIR的辅助类, 该类包含状态, 保存着新op的插入点
    mlir::OpBuilder builder;

    // 记录输出结果的数组名称
    llvm::StringRef outputArrayName;

    // 记录operation指定的kernel名称
    llvm::StringRef operationKernelName;

    // 记录问题域大小及类型
    swsten::VarType domainType;

    // 记录operation操作的上下界, (stencil.copy操作的上下界与operation指定的kernel相同)
    std::vector<int64_t> operationLB;
    std::vector<int64_t> operationUB;

    // 计算函数符号表
    std::map<StringRef, mlir::Value> symbolTable;
    // 迭代函数符号表
    std::map<StringRef, mlir::Value> iterFuncSymbolTable;
    // kernel符号表, 每进行处理kernel之前必须清理
    std::map<llvm::StringRef, mlir::Value> kernelSymbolTable;

    // 辅助函数, 负责将AST中的location转化为MLIR的location
    mlir::Location loc(swsten::Location loc) {
        return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line, loc.col);
    }

    // 在当前范围内声明一个变量, 如果该变量还未被声明则返回成功, 该函数负责管理计算函数的符号表
    mlir::LogicalResult declare(llvm::StringRef &name, mlir::Value value) {
        auto res = symbolTable.find(name);
        if (res != symbolTable.end())
            return mlir::failure();
        symbolTable[name] = value;
        return mlir::success();
    }

    // 在当前范围内声明一个变量, 如果该变量还未被声明则返回成功, 该函数负责管理迭代函数的符号表
    mlir::LogicalResult declareIterFunc(llvm::StringRef &name, mlir::Value value) {
        auto res = iterFuncSymbolTable.find(name);
        if (res != iterFuncSymbolTable.end())
            return mlir::failure();
        iterFuncSymbolTable[name] = value;
    }

    // 根据给定的维度大小及元素类型构造mlir中类型(数组)
    mlir::Type getType(VarDeclExprAST &varDecl) {
        auto name = varDecl.getName();
        auto type = varDecl.getType();
        auto arrayType = varDecl.getArrayType();
        auto location = varDecl.loc();

        if (type.shape.empty()) {
            emitError(loc(location)) << "expected an array '" << name << "'"; 
            return nullptr;
        }
        // 如果是结构化数组, 则一定是问题域数组, 需要记录其相关信息
        if (arrayType == Type_StructArray) {
            domainType = type;
        }
        if (type.elemType == Type_Double)
            return mlir::stencil::FieldType::get(builder.getF64Type(), type.shape);
        
        return mlir::stencil::FieldType::get(builder.getF32Type(), type.shape);
    }

    // 获取指定kernel数组中的信息, 包含引用的数组变量名及上下界
    void getKernelInputArrayInfo(ExprAST *expr, std::set<std::string> &inputs) {
        auto location = expr->loc();
        switch(expr->getKind()) {
        case ExprAST::Expr_Num:  // 数字直接返回
            break;
        case ExprAST::Expr_Array: // 对于数组类型, 将其名称插入到inputs中, 并根据其偏移量更新该kernel的上下界
            {
                auto arrayExpr = dyn_cast<ArrayExprAST>(expr);
                if (!arrayExpr)
                    emitError(loc(location)) << "Failed to parse kernel Expr";
                inputs.insert(arrayExpr->getName().str());
            }
            break;
        case ExprAST::Expr_BinOp: // 对于二元操作符类型, 需要遍历其左子树和右子树
            {
                auto binExpr = dyn_cast<BinaryExprAST>(expr);
                if (!binExpr)
                    emitError(loc(location)) << "Failed to parse kernel Expr";
                getKernelInputArrayInfo(binExpr->getLHS(), inputs);
                getKernelInputArrayInfo(binExpr->getRHS(), inputs);
            }
            break;
        default:    // Expr_VarDecl 和其他出现的类型均为非法
            emitError(loc(location)) << "unexpected Expr type '" << Twine(expr->getKind()) << "'";
            break;
        }

        return;
    }

    // stencil 层将被转化为函数
    // stencil计算部分将会被生成为一个函数
    // 多次迭代部分将会被生成另外一个函数(如果需要多次迭代的话)
    std::vector<mlir::FuncOp> mlirGen(StencilAST &stencilAST) {
        std::vector<mlir::FuncOp> retFuncList;
        auto location = loc(stencilAST.loc());

        // 保存operation指定的kernel的名称
        operationKernelName = stencilAST.getOperation();

        /************ 生成计算部分的函数 ****************/
        // 处理参数列表, 生成相应类型, 注意, 对于Struct类型的数组(指采用偏移方式访问的数组, 一般而言有且只有一个),
        // 需要复制一份用以表示输出数组
        // argNames保存了计算函数传参的名称顺序, 同时迭代函数中的stencil.iteration的左半部分传参也使用了同样的顺序
        // argNamesForIterationOp记录了stencil.iteration中右半部分传参的名称顺序
        llvm::SmallVector<mlir::Type, 4> argTypes;
        llvm::SmallVector<llvm::StringRef, 4> argNames;
        llvm::SmallVector<mlir::StringRef, 4> argNamesForIterationOp;
        for (int i = 0; i < stencilAST.getArgs().size(); i++) {
            auto argItem = stencilAST.getArgs()[i].get();
            mlir::Type type = getType(*argItem);
            llvm::StringRef name = (*argItem).getName();
            if (!type)
                return retFuncList;

            if ((*argItem).getArrayType() == Type_StructArray) {
                // 如果参数是结构化数组, 需要增加写回数组
                llvm::StringRef outputName = llvm::Twine(name, "_output").str();
                // 输入数组
                argNames.push_back(name);
                argTypes.push_back(type);
                // 写回数组
                argNames.push_back(outputName);
                argTypes.push_back(type);
                outputArrayName = outputName;

                argNamesForIterationOp.push_back(outputName);
                argNamesForIterationOp.push_back(name);
            } else {
                // 如果是参数型数组, 不需要增加写回数组
                argNames.push_back(name);
                argTypes.push_back(type);

                argNamesForIterationOp.push_back(name);
            }
        }

        // 构造该计算函数头
        auto computeFuncType = builder.getFunctionType(argTypes, llvm::None);
        auto stencilProgramAttr = builder.getNamedAttr(mlir::stencil::StencilDialect::getStencilProgarmAttrName(), builder.getUnitAttr());
        llvm::ArrayRef<mlir::NamedAttribute> stencilProgramAttrs(stencilProgramAttr);
        auto computeFuncOp = mlir::FuncOp::create(location, stencilAST.getName(), computeFuncType, stencilProgramAttrs);
        if (!computeFuncOp)
            return retFuncList;
        
        // 构造计算函数体, 在MLIR中, entryBlock比较特殊, 它必须包含函数参数列表
        auto &entryBlock = *computeFuncOp.addEntryBlock();
        // 在符号表中声明所有参数
        for (const auto &name_value : llvm::zip(argNames, entryBlock.getArguments())) {
            if (failed(declare(std::get<0>(name_value), std::get<1>(name_value))))
                return retFuncList;
        }

        // 给computeFuncOp添加终结符号
        builder.setInsertionPointToEnd(&entryBlock);
        builder.create<ReturnOp>(location);
        // 将插入点设置到函数体的开头
        builder.setInsertionPointToStart(&entryBlock);
        // 生成函数体内部(kernel)的内容
        auto kernelList = std::move(stencilAST.getKernelList());
        for (int kernelASTIter = 0; kernelASTIter < kernelList.size(); kernelASTIter++) {
            if (failed(mlirGen(*(kernelList[kernelASTIter])))) {
                computeFuncOp.erase();
                return retFuncList;
            }
            // 将插入点从之前生成的applyOp中移出, 移动到最后的std.return前
            builder.setInsertionPoint(&(entryBlock.back()));
        }

        // 生成stencil.copy操作将operation标定的kernel生成的结果写回到输出数组
        auto outputArray = symbolTable[outputArrayName];
        auto writeBackResult = symbolTable[stencilAST.getOperation()];
        // 根据operation的上下界生成copy的上下界(两者上下界是相同的)
        auto copyLB = builder.getI64ArrayAttr(llvm::ArrayRef<int64_t>(operationLB));
        auto copyUB = builder.getI64ArrayAttr(llvm::ArrayRef<int64_t>(operationUB));
        builder.create<stencil::CopyOp>(location, writeBackResult, outputArray, copyLB, copyUB);

        // 记录计算函数
        retFuncList.push_back(computeFuncOp);

        /************ 生成多次迭代部分的函数 *************/
        // 该部分的函数与计算函数相同, 所以可以直接使用
        // 构造迭代部分计算函数头
        auto iterationFuncType = builder.getFunctionType(argTypes, llvm::None);
        auto stencilIterationAttr = builder.getNamedAttr(mlir::stencil::StencilDialect::getStencilIterationAttrName(), builder.getUnitAttr());
        std::vector<mlir::NamedAttribute> stencilIterationAttrVec;
        stencilIterationAttrVec.push_back(stencilIterationAttr);
        llvm::ArrayRef<mlir::NamedAttribute> stencilIterationAttrs(stencilIterationAttrVec);
        auto iterationFuncOp = mlir::FuncOp::create(location, llvm::Twine(stencilAST.getName(), "_iteration").str(), iterationFuncType, stencilIterationAttrs);
        if (!iterationFuncOp) {
            retFuncList.clear();
            return retFuncList;
        }

        // 构造迭代函数体, 在MLIR中 entryBlock比较特殊, 它必须包含函数参数列表
        auto &iterationFuncEntryBlock = *iterationFuncOp.addEntryBlock();
        // 在符号表中声明所有参数
        for (const auto &name_value : llvm::zip(argNames, iterationFuncEntryBlock.getArguments())) {
            if (failed(declareIterFunc(std::get<0>(name_value), std::get<1>(name_value)))) {
                retFuncList.clear();
                return retFuncList;
            }
        }

        // 给computeFuncOp添加终结符号
        builder.setInsertionPointToEnd(&iterationFuncEntryBlock);
        builder.create<ReturnOp>(location);
        // 将插入点设置到迭代函数的开头
        builder.setInsertionPointToStart(&iterationFuncEntryBlock);
        // 生成迭代函数体内的内容, 实际上只生成一个stencil.iteration操作
        // 准备传入参数
        llvm::SmallVector<mlir::Value, 4> iterFuncOpArg;
        // 首先处理传入参数左半部分
        for (auto iter = argNames.begin(); iter != argNames.end(); iter++) {
            mlir::Value item = iterFuncSymbolTable[*iter];
            iterFuncOpArg.push_back(item);
        }

        // 然后处理传入参数右半部分
        for (auto iter = argNamesForIterationOp.begin(); iter != argNamesForIterationOp.end(); iter++) {
            mlir::Value item = iterFuncSymbolTable[*iter];
            iterFuncOpArg.push_back(item);
        }

        int bindParamNum = argNames.size();
        int iterationNum = stencilAST.getIteration();

        if (iterationNum != 1 && iterationNum%2 != 0) {
            emitError(location) << "expect iteration num is 1 or a multiple of 2";
            retFuncList.clear();
            return retFuncList;
        }
        // 获取mpiTile
        std::vector<int64_t> mpiTile = stencilAST.getMpiTile();
        // 获取mpiHalo, 并将其分成mpiHaloL和mpiHaloU
        // mpiHaloL中存储的是偏移量为负的数字部分
        // mpiHaloU中存储的是偏移量为正的数字部分
        std::vector<int64_t> mpiHaloL;
        std::vector<int64_t> mpiHaloU;
        std::vector<std::pair<int64_t, int64_t>> mpiHaloLAndU = stencilAST.getMpiHalo();
        for (auto iter = mpiHaloLAndU.begin(); iter != mpiHaloLAndU.end(); iter++) {
            mpiHaloL.push_back(iter->first);
            mpiHaloU.push_back(iter->second);
        }
        // 参数转化, 如果对应的参数存在, 则生成相应的传参, 否则对应传参为空
        llvm::Optional<llvm::ArrayRef<int64_t>> mpiTile_param = llvm::Optional<llvm::ArrayRef<int64_t>>();
        llvm::Optional<llvm::ArrayRef<int64_t>> mpiHaloL_param = llvm::Optional<llvm::ArrayRef<int64_t>>();
        llvm::Optional<llvm::ArrayRef<int64_t>> mpiHaloU_param = llvm::Optional<llvm::ArrayRef<int64_t>>();
        if (mpiTile.size() != 0)
            mpiTile_param = llvm::ArrayRef<int64_t>(mpiTile);
        if (mpiHaloL.size() != 0 && mpiHaloU.size() != 0) {
            mpiHaloL_param = llvm::ArrayRef<int64_t>(mpiHaloL);
            mpiHaloU_param = llvm::ArrayRef<int64_t>(mpiHaloU);
        }
        if (iterationNum != 1) {
            // 创建stencil.iteration操作
            builder.create<stencil::IterationOp>(location, 
                            builder.getSymbolRefAttr(computeFuncOp.getName()), 
                            iterFuncOpArg, iterationNum/2, bindParamNum, 
                            mpiTile_param, mpiHaloL_param, mpiHaloU_param);
            // 记录迭代函数
            retFuncList.push_back(iterationFuncOp);
        }
        return retFuncList;
    }

    // 处理计算函数内部kernel部分
    mlir::LogicalResult mlirGen(KernelAST &kernelAST) {
        auto location = kernelAST.loc();
        // 获取kernelName
        auto kernelName = kernelAST.getName();
        // 遍历kernel中的expr项目, 获取其使用的所有输入数组
        std::set<std::string> kernelInputName;
        getKernelInputArrayInfo(kernelAST.getExpr(), kernelInputName);

        // 获取上下界
        std::vector<int64_t> kernelLB;
        std::vector<int64_t> kernelUB;
        std::vector<std::pair<int64_t, int64_t>> lbAndUb = kernelAST.getDomainRange();

        for (auto iter = lbAndUb.begin(); iter != lbAndUb.end(); iter++) {
            kernelLB.push_back(iter->first);
            kernelUB.push_back(iter->second);
        }

        // 如果当前kernel是operation指定的kernel, 则需记录其上界和下界, 以提供给stencil.copy操作
        if (kernelName.equals(operationKernelName)) {
            operationLB = kernelLB;
            operationUB = kernelUB;
        }

        // 根据获得到的输入数组, 构造stencil.apply的输入参数
        // 记录输入数组名称
        llvm::SmallVector<llvm::StringRef, 4> argsName;
        llvm::SmallVector<mlir::Value, 4> args;
        for (auto iter = kernelInputName.begin(); iter != kernelInputName.end(); iter ++) {
            if (auto variable = symbolTable[llvm::StringRef(*iter)]) {
                argsName.push_back(llvm::StringRef(*iter));
                args.push_back(variable);
            } else {
                emitError(loc(location)) << "Array '" << *iter << "' has not been define ";
                return failure();
            }
        }
        
        // 返回结果的类型与(问题域的)输入数组中结构化数组的类型相同
        mlir::Type kernelResultType; 
        if (domainType.elemType == Type_Double)
            kernelResultType = mlir::stencil::FieldType::get(builder.getF64Type(), domainType.shape);
        else 
            kernelResultType = mlir::stencil::FieldType::get(builder.getF32Type(), domainType.shape);

        // 创建apply操作
        auto applyOp = builder.create<stencil::ApplyOp>(loc(location), args, kernelLB, kernelUB, \
                kernelAST.getTile(), kernelAST.getSWCacheAt(), kernelResultType);

        // 建立输入数组名称与block中value的绑定关系, 是kernel部分的符号表
        auto entryBlock = applyOp.getBody();
        kernelSymbolTable.clear();
        for (const auto &name_value : llvm::zip(argsName, entryBlock->getArguments()))
            kernelSymbolTable[std::get<0>(name_value)] = std::get<1>(name_value);

        // 将插入点更改为apply定义体的开头
        builder.setInsertionPointToStart(applyOp.getBody());

        // 继续解析Expr表达式
        auto resultValue = mlirGen(*(kernelAST.getExpr()));
        if (!resultValue)
            return failure();
        
        // 将结果Value与kernel名称写入符号表
        if (failed(declare(kernelName, applyOp.getResult(0))))
            return failure();

        // 生成stencil.store操作
        auto storeResult = builder.create<stencil::StoreOp>(loc(location), resultValue);

        // 生成stencil.return操作
        auto returnOp = builder.create<stencil::ReturnOp>(loc(location), storeResult.getResult(), llvm::Optional<mlir::ArrayAttr>());
        if (!returnOp)
            return failure();
        return success();
    }

    // 生成表达式部分
    mlir::Value mlirGen(ExprAST &expr) {
        switch (expr.getKind()) {
            case swsten::ExprAST::Expr_BinOp:
                return mlirGen(cast<BinaryExprAST>(expr));
            case swsten::ExprAST::Expr_Num:
                return mlirGen(cast<NumberExprAST>(expr));
            case swsten::ExprAST::Expr_Array:
                return mlirGen(cast<ArrayExprAST>(expr));
            default:
                emitError(loc(expr.loc()))
                    << "MLIR codegen encounterd an unexpected expr kind '"
                    << Twine(expr.getKind()) << "'";
                return nullptr;
        }
    }

    // 生成二元运算符号
    mlir::Value mlirGen(BinaryExprAST &binOp) {
        auto location = loc(binOp.loc());

        // 处理左子树表达式
        mlir::Value lhs = mlirGen(*binOp.getLHS());
        if (!lhs)
            return nullptr;

        // 处理右子树表达式
        mlir::Value rhs = mlirGen(*binOp.getRHS());
        if (!rhs)
            return nullptr;
        
        // 根据实际的运算符生成合适的op
        switch(binOp.getOp()) {
        case '+':
            return builder.create<AddFOp>(location, lhs, rhs);
        case '-':
            return builder.create<SubFOp>(location, lhs, rhs);
        case '*':
            return builder.create<MulFOp>(location, lhs, rhs);
        case '/':
            return builder.create<DivFOp>(location, lhs, rhs);
        default:
            // 错误处理
            emitError(location, "invalid binary operator '") << binOp.getOp() << "'";
            return nullptr;
        }
    }

    // 生成数字
    mlir::Value mlirGen(NumberExprAST &num) {
        if (num.getType() == Type_Double)
            return builder.create<ConstantOp>(loc(num.loc()), builder.getF64FloatAttr(num.getValue()));
        
        return builder.create<ConstantOp>(loc(num.loc()), builder.getF32FloatAttr((float)num.getValue()));
    }

    // 生成数组
    mlir::Value mlirGen(ArrayExprAST &array) {
        auto location = loc(array.loc());
        auto index = array.getIndex();
        auto arrayValue= kernelSymbolTable[array.getName()];
        // 结构化数组, 使用stencil.access进行访问
        if (array.getArrayType() == Type_StructArray)
            return builder.create<stencil::AccessOp>(location, arrayValue, llvm::ArrayRef<int64_t>(index));

        // 否则是参数数组, 采用stencil.load进行访问
        return builder.create<stencil::LoadOp>(location, arrayValue, index);
    }

};

} // End of anonymous namespace

namespace swsten {
// 生成MLIR的对外接口
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST) {
    return MLIRGenImpl(context).mlirGen(moduleAST);
}

}