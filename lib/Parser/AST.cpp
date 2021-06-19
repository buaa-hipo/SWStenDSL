/**
 * @file AST.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 抽象语法树相关函数实现
 * @version 0.1
 * @date 2021-06-09
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include "Parser/AST.h"

#include <llvm/ADT/Twine.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>

using namespace swsten;

namespace {

// RAII 辅助结构, 用来在遍历AST时管理缩进
struct Indent {
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
};

// 遍历AST输出语法树节点信息的辅助类
class ASTDumper {
public:
    void dump(ModuleAST *node);

private:
    void dump(const VarType &type);
    void dump(const ElemType &type);
    void dump(ExprAST *expr);
    void dump(VarDeclExprAST *varDecl);
    void dump(NumberExprAST *num);
    void dump(ArrayExprAST *array);
    void dump(BinaryExprAST *binary);
    void dump(KernelAST *kernel);
    void dump(StencilAST *stencil);

    // 输出符合当前缩进的制表符
    void indent() {
        for (int i = 0; i < curIndent; i++)
            llvm::errs() << "\t";
    }

    int curIndent = 0;
};
} // End of anonymous namespace

// 返回一个字符串表示当前节点在源文件中的位置
template<typename T> static std::string loc(T *node) {
    const auto &loc = node->loc();
    return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
            llvm::Twine(loc.col)).str();
}

// 缩进辅助宏, 当进入到一个新的node节点时, 缩进增加并输出缩进, 
// 退出节点时, 缩进减少
#define INDENT()\
    Indent level_(curIndent);\
    indent();

// 输出类型属性
void ASTDumper::dump(const VarType &type) {
    llvm::errs() << "<";

    std::vector<int64_t> shape = type.shape;
    for (int i = 0; i < shape.size(); i++) 
        llvm::errs() << shape[i] << "x";

    dump(type.elemType);

    llvm::errs() << ">";
}

// 输出元素类型(单精度or双精度)
void ASTDumper::dump(const ElemType &type) {
    if (type == Type_Float)
        llvm::errs() << "f32";
    else if (type == Type_Double)
        llvm::errs() << "f64";
    else 
        llvm::errs() << "Error Type";
}

// 将Expr的dump转发到合适的expr子类的dump
void ASTDumper::dump(ExprAST *expr) {
    llvm::TypeSwitch<ExprAST *>(expr)
        .Case<VarDeclExprAST, NumberExprAST, ArrayExprAST, BinaryExprAST>(
            [&](auto *node) { this->dump(node); })
        .Default([&](ExprAST *) {
            // 没有找到匹配的子类, 输出错误信息
            INDENT();
            llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
        });
}

// 输出变量声明节点
void ASTDumper::dump(VarDeclExprAST *varDecl) {
    INDENT();
    llvm::errs() << "VarDecl " << varDecl->getName();
    dump(varDecl->getType());
    std::string arrayType;
    if (varDecl->getArrayType() == Type_StructArray)
        arrayType = "Struct";
    else 
        arrayType = "Param";
    llvm::errs() << "(ArrayType: " << arrayType << ") " << loc(varDecl) << "\n";
}

// 输出数值节点
void ASTDumper::dump(NumberExprAST *number) {
    INDENT();
    llvm::errs()<< "Value: " << number->getValue() 
                << " (Type: ";
    dump(number->getType());
    llvm::errs()<< ") " << loc(number) << "\n";
}

// 输出数组节点
void ASTDumper::dump(ArrayExprAST *array) {
    INDENT();
    llvm::errs() << array->getName();

    std::vector<int64_t> index = array->getIndex();
    for (int i = 0; i < index.size(); i++) {
        llvm::errs() << "[" << index[i] << "]";
    }
    
    std::string arrayType;
    if (array->getArrayType() == Type_ParamArray)
        arrayType = "Param";
    else
        arrayType = "Struct";
    llvm::errs() << " (Type: ";
    dump(array->getType());
    llvm::errs() << ", ArrayType: " << arrayType << ") " 
                << loc(array) << "\n";
}

// 输出二元运算符节点
void ASTDumper::dump(BinaryExprAST *binary) {
    INDENT();
    llvm::errs() << "BinOp: '" << binary->getOp() << "' " << loc(binary) << "\n";

    dump(binary->getLHS());
    dump(binary->getRHS());
}

// 输出kernel节点
void ASTDumper::dump(KernelAST *kernel) {
    // 输出名称
    INDENT();
    llvm::errs() << "kernel: " << kernel->getName() << loc(kernel) << "\n";

    // 输出Tile
    indent();
    llvm::errs() << "-->Tile: [";
    std::vector<int64_t> tile;
    tile = kernel->getTile();
    for (int i=0; i < tile.size(); i++) {
        llvm::errs() << tile[i];
        
        if (i+1 != tile.size())
            llvm::errs() << ",";
    }
    llvm::errs() << "]\n";

    // 输出swCacheAt
    indent();
    llvm::errs() << "-->swCacheAt: " << kernel->getSWCacheAt() << "\n";

    // 输出问题域范围
    indent();
    llvm::errs() << "-->domain: ";
    std::vector<std::pair<int64_t, int64_t>> domainRange = kernel->getDomainRange();
    for (auto iter = domainRange.begin(); iter != domainRange.end(); iter++)
        llvm::errs() << "[" << iter->first << "," << iter->second << "]";
    llvm::errs() << "\n";

    // 输出Expr
    indent();
    llvm::errs() << "-->Expr:\n";
    dump(kernel->getExpr());
}

// 输出stencil节点
void ASTDumper::dump(StencilAST *stencil) {
    // 输出名称
    INDENT();
    llvm::errs() << "Stencil:" << stencil->getName() << " " << loc(stencil) << "\n";

    // 输出参数列表
    indent();
    llvm::errs() << "-->Input: \n";
    llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> args = std::move(stencil->getArgs());
    for (int i = 0; i < args.size(); i++) {
        dump(args[i].get());
    }

    // 输出迭代次数
    indent();
    llvm::errs() << "-->Iteration: " << stencil->getIteration() << "\n";

    // 输出mpi划分
    indent();
    std::vector<int> mpiTile = stencil->getMpiTile();
    llvm::errs() << "-->mpiTile: [";
    for (int i = 0; i < mpiTile.size(); i++) {
        llvm::errs() << mpiTile[i];

        if (i+1 != mpiTile.size())
            llvm::errs() << ",";
    }
    llvm::errs() << "]\n";

    // 输出operation
    indent();
    llvm::errs() << "-->operation: " << stencil->getOperation() << "\n";

    // 输出kernel
    indent();
    llvm::errs() << "-->Kernels:\n";
    llvm::ArrayRef<std::unique_ptr<KernelAST>> kernelList = std::move(stencil->getKernelList());
    for (int i = 0; i < kernelList.size(); i++) {
        dump(kernelList[i].get());
    }

}

// 解析ModuleAST
void ASTDumper::dump(ModuleAST *node) {
    llvm::errs() << "Module:\n";
    dump(node->getStencil());
}

namespace swsten {
    // 对外提供的API
    void dump(ModuleAST &module)  {
        ASTDumper().dump(&module);
    }
} // End of swsten