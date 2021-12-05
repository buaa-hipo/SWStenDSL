/**
 * @file Parser.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 语法解析器
 * @version 0.1
 * @date 2021-06-07
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTENDSL_PARSER_PARSER_H_
#define _SWSTENDSL_PARSER_PARSER_H_

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

#include <map>
#include <utility>
#include <vector>

#include "AST.h"
#include "Lexer.h"

namespace swsten {

// 这只是一个简单的语法解析器, 根据词法分析器返回的Token构造抽象语法树(AST), 并未进行严格的
// 语法检查及符号解析, 比如代码可以引用未定义的变量名及符号名
class Parser {
public:
    // 利用提供的词法解析器构造语法解析器
    Parser(Lexer &lexer) : lexer(lexer) {}

    // 解析整个模块, 模块由一个stencil定义构成
    std::unique_ptr<ModuleAST> parseModule() {
        // 初始化词法解析器
        lexer.getNextToken();

        std::unique_ptr<StencilAST> record;
        switch (lexer.getCurToken()) {
            case tok_eof:
                break;
            case tok_stencil:
                record = parseStencil();
                break;
            default:
                return parseError<ModuleAST> ("stencil", "when parsing top level module.");
        }

        // 如果没能解析到文件末尾EOF, 说明解析构成中出现错误
        if (lexer.getCurToken() != tok_eof)
            return parseError<ModuleAST>("nothing", "at end of module");

        return std::make_unique<ModuleAST>(std::move(record));
    }

private:
    Lexer &lexer;   // 词法解析器
    ElemType domainType; // 问题域计算类型, 单精度or双精度
    // 记录数组的访问类型, 根据kernel中具体的使用情况来记录, 从而在stencilAST中设置相关数组的访问类型
    std::map<std::string, swsten::ArrayType> arrayNameAndAccessPatternMapping;

    // 解析stencil定义
    std::unique_ptr<StencilAST> parseStencil() {
        if (lexer.getCurToken() != tok_stencil)
            return parseError<StencilAST>("stencil", "in stencil define");
        lexer.consume(tok_stencil);

        // 解析stencil计算名称
        auto loc = lexer.getLastLocation();
        if (lexer.getCurToken() != tok_identifier)
            return parseError<StencilAST>("stencil name", "in stencil define");
        std::string stencilName(lexer.getId());
        lexer.consume(tok_identifier);

        if (lexer.getCurToken() != '(')
            return parseError<StencilAST>("(", "in stencil define");
        lexer.consume(Token('('));

        // 解析名称后面跟随的列表
        std::vector<std::unique_ptr<VarDeclExprAST>> args;
        if (lexer.getCurToken() != ')') {
            // 解析问题域类型
            if (lexer.getCurToken() == tok_float) {
                domainType = Type_Float;
                lexer.consume(tok_float);
            } else if (lexer.getCurToken() == tok_double) {
                domainType = Type_Double;
                lexer.consume(tok_double);
            } else {
                return parseError<StencilAST>("float or double", "in stencil define");
            }

            // 解析参数列表
            do {
                auto argLoc = lexer.getLastLocation();
                std::string name;
                VarType type;

                // 解析变量名称
                if (lexer.getCurToken() != tok_identifier)
                    return parseError<StencilAST>("variable name", "in stencil define");
                name = lexer.getId().str();
                lexer.consume(tok_identifier);

                // 解析参数维度信息
                type.elemType = domainType;     // 参数元素类型与问题域类型相同
                do {
                    if (lexer.getCurToken() != '[')
                        return parseError<StencilAST>("[", "in stencil define");
                    lexer.consume(Token('['));
                    
                    // 解析维度大小
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<StencilAST>("Integer", "in stencil define");
                    type.shape.push_back((int64_t)lexer.getValue());
                    lexer.consume(tok_number);
                    
                    if (lexer.getCurToken() != ']')
                        return parseError<StencilAST>("]", "in stencil define");
                    lexer.consume(Token(']'));
                } while (lexer.getCurToken() == '[');   // 如果还有'['则证明维度未结束
                
                assert(type.shape.size() <= 3 && "domain dimension is excepted not greater than 3");
                // 添加参数到参数列表中
                args.push_back(std::make_unique<VarDeclExprAST>(argLoc, name, type));

                // 如果下一个字符为',', 则说明还有参数, 否则代表参数列表终止
                if (lexer.getCurToken() != ',')
                    break;
                lexer.consume(Token(','));
            } while(true);

            // 如果上述解析维度信息过程结束时当前符号不是')'则说明出错
            if (lexer.getCurToken() != ')')
                return parseError<StencilAST>(")", "in stencil define");
            lexer.consume(Token(')'));
        }

        // 解析定义体
        int iteration=0;
        std::vector<int64_t> mpiTile;
        std::vector<std::pair<int64_t, int64_t>> mpiHalo;
        std::string operation;
        std::vector<std::unique_ptr<KernelAST>> kernelList;

        // 解析左大括号
        if (lexer.getCurToken() != '{')
            return parseError<StencilAST>("{", "in stencil define body");
        lexer.consume(Token('{'));

        while (lexer.getCurToken() != '}') {
            if (lexer.getCurToken() == tok_iteration) {
                lexer.consume(tok_iteration);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<StencilAST>("(", "in stencil define body");
                lexer.consume(Token('('));

                // 解析数字
                if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                    return parseError<StencilAST>("Integer", "in stencil define body");
                iteration = (int)lexer.getValue();
                lexer.consume(tok_number);
                
                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<StencilAST>(")", "in stencil define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_mpiTile) {
                lexer.consume(tok_mpiTile);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<StencilAST>("(", "in stencil define body");
                lexer.consume(Token('('));

                while (true) {
                    // 解析数字
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<StencilAST>("Integer", "in stencil define body");
                    mpiTile.push_back((int64_t)lexer.getValue());
                    lexer.consume(tok_number);

                    // 解析数字之间的逗号, 此处跳出只能是右括号, 后续解析右括号时会做相应检查
                    if (lexer.getCurToken() != ',')
                        break;
                    lexer.consume(Token(','));
                }

                assert(mpiTile.size() <= 3 && "domain dimension is excepted not greater than 3");

                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<StencilAST>(")" , "in stencil define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_mpiHalo) {
                lexer.consume(tok_mpiHalo);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<StencilAST>("(", "in stencil define body");
                lexer.consume(Token('('));
                // 解析mpiHalo信息
                do {
                    if (lexer.getCurToken() != '[')
                        return parseError<StencilAST>("]", "in stencil define body");
                    lexer.consume(Token('['));

                    // 解析对应维度的halo
                    // 指代某一维度的halo, halo_a代表的是与坐标轴反方向的halo,
                    // 而halo_b则表示的是与坐标轴方向相同的halo
                    // 简而言之就是halo_a指代的偏移是负的, 以正值进行保存
                    // halo_b指代的偏移是正的, 以正值保存
                    int64_t halo_a, halo_b;
                    // halo_a
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<StencilAST>("Integer", "in stencil define body");
                    halo_a = (int64_t)lexer.getValue();
                    lexer.consume(tok_number);

                    // 解析数字之间的逗号
                    if (lexer.getCurToken() != ',')
                        return parseError<StencilAST>(",", "in stencil define body");
                    lexer.consume(Token(','));

                    // halo_b
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<StencilAST>("Integer", "in stencil define body");
                    halo_b = (int64_t)lexer.getValue();
                    lexer.consume(tok_number);

                    std::pair<int64_t, int64_t> halo_ab(halo_a, halo_b);
                    mpiHalo.push_back(halo_ab);

                    if (lexer.getCurToken() != ']')
                        return parseError<StencilAST>("]", "in stencil define body");
                    lexer.consume(Token(']'));
                } while (lexer.getCurToken() == '['); // 如果还有'['则证明解析还未结束
                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<StencilAST>(")", "in stencil define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_operation) {
                lexer.consume(tok_operation);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<StencilAST>("(", "in stencil define body");
                lexer.consume(Token('('));

                // 解析kernel名称
                if (lexer.getCurToken() != tok_identifier)
                    return parseError<StencilAST>("kernel name", "in stencil define body");
                operation = lexer.getId().str();
                lexer.consume(tok_identifier);

                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<StencilAST>(")", "in stencil define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_kernel) {
                kernelList.push_back(parseKernel());
            } else {
                return parseError<StencilAST>("iteration, mpiTile, operation or kernel", "in stencil define body");
            }
        }

        // 解析右大括号
        if (lexer.getCurToken() != '}')
            return parseError<StencilAST>("}", "in stencil define body");
        lexer.consume(Token('}'));

        // 遍历整个参数列表, 根据解析得到的访问模式, 设置参数列表中的相应数组的访问模式
        for (unsigned argIter = 0; argIter < args.size(); argIter++) {
            args[argIter]->setArrayType(arrayNameAndAccessPatternMapping[args[argIter]->getName().str()]);
        }
        return std::make_unique<StencilAST>(loc, std::move(kernelList), stencilName, std::move(args), iteration, mpiTile, mpiHalo, operation);
    }

    std::unique_ptr<KernelAST> parseKernel() {
        // 解析kernel关键字
        if (lexer.getCurToken() != tok_kernel)
            return parseError<KernelAST>("kernel", "in kernel define");
        lexer.consume(tok_kernel);

        // 解析kernel名称
        auto loc = lexer.getLastLocation();
        if (lexer.getCurToken() != tok_identifier)
            return parseError<KernelAST>("kernel name", "in kernel define");
        std::string kernelName(lexer.getId());
        lexer.consume(tok_identifier);

        // 解析定义体
        std::vector<int64_t> tile;
        int swCacheAt = -1;
        std::vector<std::pair<int64_t, int64_t>> domainRange;
        std::unique_ptr<ExprAST> expr;
        // 解析左大括号
        if (lexer.getCurToken() != '{')
            return parseError<KernelAST>("{", "in kernel define body");
        lexer.consume(Token('{'));

        while (lexer.getCurToken() != '}') {
            if (lexer.getCurToken() == tok_tile) {
                lexer.consume(tok_tile);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<KernelAST>("(", "in kernel define body");
                lexer.consume(Token('('));

                // 解析数字部分
                while (true) {
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<KernelAST>("Integer", "in stencil define body");
                    tile.push_back((int)lexer.getValue());
                    lexer.consume(tok_number);

                    // 解析数字之间的逗号,此处跳出只能是右括号, 后续解析右括号时会做相应检查
                    if (lexer.getCurToken() != ',')
                        break;
                    lexer.consume(Token(','));
                }
                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<KernelAST>(")", "in kernel define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_swCacheAt) {
                lexer.consume(tok_swCacheAt);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<KernelAST>("(", "in kernel define body");
                lexer.consume(Token('('));

                // 解析数字
                if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                    return parseError<KernelAST>("Integer", "in stencil define body");
                swCacheAt = (int)lexer.getValue();
                lexer.consume(tok_number);

                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<KernelAST>(")", "in kernel define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_domain) {
                lexer.consume(tok_domain);
                // 解析左括号
                if (lexer.getCurToken() != '(')
                    return parseError<KernelAST>("(", "in kernel define body");
                lexer.consume(Token('('));
                // 解析domain信息
                do {
                    if (lexer.getCurToken() != '[')
                        return parseError<KernelAST>("[", "in kernel define body");
                    lexer.consume(Token('['));

                    // 解析某维度的上下界
                    int64_t lb, ub;
                    // 下界
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<KernelAST>("Integer", "in kernel define body");
                    lb = (int64_t)lexer.getValue();
                    lexer.consume(tok_number);
                    // 解析数字之间的逗号
                    if (lexer.getCurToken() != ',')
                        return parseError<KernelAST>(",", "in kernel define body");
                    lexer.consume(Token(','));

                    // 上界
                    if (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer)
                        return parseError<KernelAST>("Integer", "in kernel define body");
                    ub = (int64_t)lexer.getValue();
                    lexer.consume(tok_number);

                    std::pair<int64_t, int64_t> lbAndub(lb, ub);
                    domainRange.push_back(lbAndub);

                    if (lexer.getCurToken() != ']')
                        return parseError<KernelAST>("]", "in kernel define body");
                    lexer.consume(Token(']'));
                } while (lexer.getCurToken() == '['); // 如果还有'['则证明解析还未结束
                // 解析右括号
                if (lexer.getCurToken() != ')')
                    return parseError<KernelAST>(")", "in kernel define body");
                lexer.consume(Token(')'));
            } else if (lexer.getCurToken() == tok_Expr) {
                lexer.consume(tok_Expr);
                // 解析左大括号
                if (lexer.getCurToken() != '{')
                    return parseError<KernelAST>("{", "in kernel define body");
                lexer.consume(Token('{'));                

                expr = parseExpression();
                // 解析右大括号
                if (lexer.getCurToken() != '}')
                    return parseError<KernelAST>("}", "in kernel define body");
                lexer.consume(Token('}'));
            } else {
                return parseError<KernelAST>("tile, swCacheAt, domain, expr", "in kernel define body");
            }
        }

        // 解析右大括号
        if (lexer.getCurToken() != '}') 
            return parseError<KernelAST>("}", "in kernel define body");
        lexer.consume(Token('}'));

        // 构造并返回
        return std::make_unique<KernelAST>(loc, std::move(expr), kernelName, tile, swCacheAt, domainRange);
    }

    // 解析表达式
    std::unique_ptr<ExprAST> parseExpression() {
        auto lhs = parsePrimary();
        if (!lhs)
            return nullptr;
        return parseBinOpRHS(0, std::move(lhs));
    }

    // 解析某值, 该值可能是数字, 指定位置的输入数组, kernel指定位置的数组, 亦或是括号表达式
    std::unique_ptr<ExprAST> parsePrimary() {
        switch (lexer.getCurToken()) {
            case '-':           // '-'号可能跟随数字, 用以表示负数
            case tok_number:
                return parseNumberExpr();
            case tok_identifier:
                return parseIdentifierExpr();
            case '(':
                return parseParentExpr();
            default:
                llvm::errs() << "unknown token '" << lexer.getCurToken() 
                             << "' when expecting an expression";
                return nullptr;
        }
    }

    // 解析表达式RHS
    std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec, 
                                            std::unique_ptr<ExprAST> lhs) {
        while (true) {
            // 如果当前是二元运算符, 首先获取其优先级
            int tokPrec = getTokPrecedence();

            // 如果之前的操作已经处理的操作符优先级高于现在未处理的操作符优先级, 则返回, 不继续处理
            if (tokPrec < exprPrec)
                return lhs;
            
            // 否则继续处理
            int binOp = lexer.getCurToken();
            auto loc = lexer.getLastLocation();
            lexer.consume(Token(binOp));

            // 处理当前二元操作符号右边的值
            auto rhs = parsePrimary();
            if (!rhs)
                return parseError<ExprAST>("expression", "to complete binary operator");
            
            // 如果当前操作符的优先级低于后面操作符的优先级, 则说明rhs还未处理完, 需要继续进行处理
            int nextPrec = getTokPrecedence();
            if (tokPrec < nextPrec) {
                rhs = parseBinOpRHS(tokPrec+1, std::move(rhs));
                if (!rhs)
                    return nullptr;
            }

            // 合并lhs和rhs
            lhs = std::make_unique<BinaryExprAST>(loc, binOp, std::move(lhs), std::move(rhs), domainType);
        }
    }

    // 解析数值(浮点数)
    std::unique_ptr<ExprAST> parseNumberExpr() {
        auto loc = lexer.getLastLocation();
        int flag = 1;

        // 检查开头是否有'-'号
        if (lexer.getCurToken() == '-') {
            flag = -1;
            lexer.consume(Token('-'));
        }

        // 处理完可能存在的'-'号后, 检查是否为浮点数
        if (lexer.getCurToken() != tok_number)
            return parseError<BinaryExprAST>("float point number", "in expr define");

        auto result = std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue()*flag, domainType);
        lexer.consume(tok_number);

        return std::move(result);
    }

    // 解析括号表达式
    std::unique_ptr<ExprAST> parseParentExpr() {
        lexer.consume(Token('('));
        auto v = parseExpression();
        if (!v)
            return nullptr;
        
        if (lexer.getCurToken() != ')')
            return parseError<ExprAST>(")", "to close expression with parentheses");
        lexer.consume(Token(')'));
        return v;
    }

    // 解析变量名(包括指定位置的输入数组以及kernel数组)
    std::unique_ptr<ExprAST> parseIdentifierExpr() {
        auto loc = lexer.getLastLocation();
        std::string name(lexer.getId());
        lexer.consume(tok_identifier);

        // 解析指定的位置
        std::vector<int64_t> offset;
        ArrayType arrayType = Type_ParamArray;  // 默认是参数数组
        int iter = 0;
        if (lexer.getCurToken() != '[')
            return parseError<ExprAST>("[", "in expr define");
        // 开始解析具体位置
        while (true) {
            // 不再以'['开头, 说明解析结束
            if (lexer.getCurToken() != '[')
                break;
            lexer.consume(Token('['));

            int flag = 1;

            // 这里有两种情况:
            // 第一种是相对偏移访问, 比如a[x+1][y+1][z+1]
            // 第二种是直接采用数字访问, 用于参数型数组, 参数型数组的访问座标始终是正值
            // 解析基准位置, 这里只是一些x,y,z, 
            // 这些字母仅用于方便用户编写程序, 实际上并不起任何作用, 也不做任何检查
            if (lexer.getCurToken() != tok_identifier && lexer.getCurToken() != tok_number)
                return parseError<ExprAST>("x, y, z or Integer", "in expr define");

            // 这个变量用于应对第一种情况下没有偏移量的情况, 如[x]
            // 由于该变量仅在第一种情况下修改, 因此对第二种情况无影响
            bool has_offset = true;
            // 处理第一种情况, 一二两种情况的区别仅在于是否对符号进行处理
            if (lexer.getCurToken() == tok_identifier) {
                lexer.consume(tok_identifier);
                // 解析偏移位置
                // 首先是一个'+'或'-'
                // 当然, 在偏移量为0时也可能没有'+'或'-', 此时要求后面必须为']',
                // 且此时不需要处理偏移量数字(数字不存在), 偏移量为0;
                if (lexer.getCurToken() == '+') {
                    flag = 1;
                    lexer.consume(Token('+'));
                } else if (lexer.getCurToken() == '-') {
                    flag = -1;
                    lexer.consume(Token('-'));
                } else if (lexer.getCurToken() == ']')
                    has_offset = false;
                else
                    return parseError<ExprAST>("+, -, ]", "in expr define");

                // 检查访问模式是否正确
                if (iter == 0)
                    arrayType = Type_StructArray;
                else if (arrayType != Type_StructArray)
                    return parseError<ExprAST>("same index pattern", "in expr define");
            } else {
                // 检查访问模式是否正确
                if (iter != 0 && arrayType != Type_ParamArray)
                    return parseError<ExprAST>("same index pattern", "in expr define");
            }

            // 解析数字
            if (has_offset && (lexer.getCurToken() != tok_number || lexer.getValueType() != type_Integer))
                return parseError<ExprAST>("Integer", "in expr define");

            int val = 0;
            // 如果有数字则进行处理
            if (has_offset) {
                val = (int)lexer.getValue()*flag;
                lexer.consume(tok_number);
            }
            offset.push_back(val);

            // 解析']'
            if (lexer.getCurToken() != ']')
                return parseError<ExprAST>("]", "in expr define");
            lexer.consume(Token(']'));
            iter ++;
        }

        // 记录该数组类型的访问模式
        arrayNameAndAccessPatternMapping[name]=arrayType;

        return std::make_unique<ArrayExprAST>(loc, name, offset, domainType, arrayType);
    }

    // 获取指定二元操作符的优先级
    int getTokPrecedence() {
        if (!isascii(lexer.getCurToken()))
            return -1;
        
        // 1是最低的优先级
        switch(static_cast<char>(lexer.getCurToken())) {
            case '+':
            case '-':
                return 20;
            case '*':
            case '/':
                return 40;
            default:
                return -1;
        }
    }

    // 当解析出现错误时, 用以报错的辅助函数, 该函数传入参数为期望的token, 另外一个参数则提供了
    // 更多的信息
    template <typename R, typename T, typename U = const char *>
    std::unique_ptr<R> parseError(T &&expected, U &&context="") {
        auto curToken = lexer.getCurToken();
        llvm::errs() << "Parse error (" << 
            *(lexer.getLastLocation().file) << "@" <<
            lexer.getLastLocation().line << ":" << 
            lexer.getLastLocation().col << "): expected '" << expected <<
            "' " << context << " but has Token " << curToken;
        if (isprint(curToken))
            llvm::errs() << " '" << (char) curToken << "'";
        
        llvm:: errs() << "\n";
        return nullptr;
    }
};
} /* End of namespace swsten */

#endif // End of _SWSTENDSL_PARSER_PARSER_H_