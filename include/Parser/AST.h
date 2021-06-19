/**
 * @file AST.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 抽象语法树
 * @version 0.1
 * @date 2021-06-07
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTENDSL_AST_H_
#define _SWSTENDSL_AST_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <vector>

#include "Lexer.h"

namespace swsten {

// 元素属性
enum ElemType : int {
    Type_Float,
    Type_Double,
};

// 类型属性
struct VarType {
    std::vector<int64_t> shape;
    ElemType elemType;
};

// 数组类型
// 两种输入数组的类型, 分为参数型数组和结构化数组
// 参数型数组使用绝对地址进行访问
// 结构化数组使用偏移地址进行访问
enum ArrayType :int {
    Type_ParamArray,
    Type_StructArray,
};

// 所有表达式节点的基类
class ExprAST {
public:
    enum ExprASTKind : int{
        Expr_VarDecl,
        Expr_Num,
        Expr_Array,
        Expr_BinOp
    };

    ExprAST(ExprASTKind kind, Location location)
        : kind(kind), location(location) {}
    virtual ~ExprAST() = default;

    ExprASTKind getKind() const { return kind; }

    const Location &loc() { return location; }

private:
    const ExprASTKind kind;
    Location location;
};

// 数字表达式节点
class NumberExprAST : public ExprAST {
private:
    double Val;
    ElemType Type;

public:
    NumberExprAST(Location loc, double val, ElemType type) : ExprAST(Expr_Num, loc) {
        this->Val = val;
        this->Type = type;
    }

    double getValue() { return Val; }
    ElemType getType() {return Type; }

    // LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

// 定义变量表达式节点, 用于定义变量
class VarDeclExprAST : public ExprAST {
private:
    std::string name;
    VarType type;
    ArrayType arrayType;

public:
    VarDeclExprAST(Location loc, llvm::StringRef name, VarType type) : ExprAST(Expr_VarDecl, loc) {
        this->name = name.str();
        this->type = type;
    }

    llvm::StringRef getName() { return name; }
    const VarType &getType() { return type; }
    const ArrayType &getArrayType() { return arrayType; }
    void setArrayType(ArrayType arrayType) {
        this->arrayType = arrayType;
    }

    // LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

// 定义数组变量, 为变量名+相应的访问位置
class ArrayExprAST : public ExprAST {
private:
    std::string name;
    std::vector<int64_t> index;
    ElemType type;
    ArrayType arrayType;

public:
    ArrayExprAST(Location loc, llvm::StringRef name, std::vector<int64_t> index, ElemType type, ArrayType arrayType)
                    : ExprAST(Expr_Array, loc) {
        this->name = name.str();
        this->index = index;
        this->type = type;
        this->arrayType = arrayType;
    }

    llvm::StringRef getName() { return name; }
    std::vector<int64_t> getIndex() { return index; }
    ElemType getType() { return type; }
    ArrayType getArrayType() {return arrayType; }

    // LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Array; }
};

// 定义二元运算
class BinaryExprAST : public ExprAST {
private:
    char op;
    std::unique_ptr<ExprAST> lhs, rhs;
    ElemType type;

public:
    BinaryExprAST(Location loc, char Op, std::unique_ptr<ExprAST> lhs, 
                    std::unique_ptr<ExprAST> rhs, ElemType type)
        : ExprAST(Expr_BinOp, loc) {
        this->op = Op;
        this->lhs = std::move(lhs);
        this->rhs = std::move(rhs);
    }
    char getOp() { return op; }
    ExprAST *getLHS() { return lhs.get(); }
    ExprAST *getRHS() { return rhs.get(); }

    // LLVM style RTTI
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

class KernelAST {
private:
    std::string name;               // kernel名称
    std::vector<int64_t> tile;      // kernel划分
    int swCacheAt;                  // 并行位置, 申威专用
    std::vector<std::pair<int64_t, int64_t>> domainRange; // 问题域计算范围
    std::unique_ptr<ExprAST> expr;

    Location location;

public:
    KernelAST(Location loc, std::unique_ptr<ExprAST> expr, std::string name, 
        std::vector<int64_t> tile, int64_t swCacheAt, std::vector<std::pair<int64_t, int64_t>> domainRange) {
        this->location = loc;
        this->name = name;
        this->tile = tile;
        this->swCacheAt = swCacheAt;
        this->domainRange = domainRange;
        this->expr = std::move(expr);
    }

    const Location &loc() { return location; }
    llvm::StringRef getName() { return name; }
    std::vector<int64_t> getTile() { return tile; }
    int64_t getSWCacheAt() { return swCacheAt; }
    std::vector<std::pair<int64_t, int64_t>> getDomainRange() { return domainRange; }
    ExprAST *getExpr() { return expr.get(); }
};

// stencil模块抽象语法书节点, 包含多个kernel的定义以及其他一些属性的定义
class StencilAST {
private:
    std::string name;                                   // stencil名称
    std::vector<std::unique_ptr<VarDeclExprAST>> args;  // 参数列表
    int iteration;                                      // 迭代次数
    std::vector<int> mpiTile;                           // mpiTile划分
    std::string operation;                              // 输出结果的kernel名称

    std::vector<std::unique_ptr<KernelAST>> kernelList;

    Location location;

public:
    StencilAST(Location loc, std::vector<std::unique_ptr<KernelAST>> kernelList, std::string name,
            std::vector<std::unique_ptr<VarDeclExprAST>> args, int iteration, 
            std::vector<int> mpiTile, std::string operation) {
        this->location = loc;
        this->name = name;
        this->args = std::move(args);
        this->iteration = iteration;
        this->mpiTile = mpiTile;
        this->operation = operation;
        this->kernelList = std::move(kernelList);
    }

    const Location &loc() { return location; }
    llvm::StringRef getName() { return name; }
    llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getArgs() { return args; }
    int getIteration() { return iteration; }
    std::vector<int> getMpiTile() { return mpiTile; }
    llvm::StringRef getOperation() { return operation; }
    llvm::ArrayRef<std::unique_ptr<KernelAST>> getKernelList() { return kernelList; }
};

// 顶层模块抽象语法树节点, 包含一个stencil定义
class ModuleAST {
    std::unique_ptr<StencilAST> stencilList;

public:
    ModuleAST(std::unique_ptr<StencilAST> stencilList)
        : stencilList(std::move(stencilList)) {}

    StencilAST *getStencil() { return stencilList.get(); }
};

void dump(ModuleAST &);

} /* End of namespace swsten */

#endif /* End of _SWSTENDSL_AST_H_ */