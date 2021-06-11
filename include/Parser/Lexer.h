/**
 * @file Lexer.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 词法分析器
 * @version 0.1
 * @date 2021-06-07
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTENDSL_PARSER_LEXER_H_
#define _SWSTENDSL_PARSER_LEXER_H_

#include <llvm/ADT/StringRef.h>

#include <assert.h>
#include <memory>
#include <string>

namespace swsten {
// 描述当前所在的文件位置
struct Location {
    std::shared_ptr<std::string> file;  // 文件名
    int line;                           // 当前行号
    int col;                            // 当前列号
};

// 词法分析器返回的符号列表
enum Token : int {
    tok_semicolon = ';', 
    tok_parenthese_open = '(',
    tok_parenthese_close =')',
    tok_bracket_open = '{',
    tok_bracket_close = '}',
    tok_sbracket_open = '[',
    tok_sbracket_close = ']',

    tok_eof = -1,

    // commands
    tok_stencil = -2,
    tok_float = -3,
    tok_double = -4,
    tok_iteration = -5,
    tok_mpiTile = -6,
    tok_operation = -7,
    tok_kernel = -8,
    tok_tile = -9,
    tok_swCacheAt = -10,
    tok_Expr = -11,

    // primary
    tok_identifier = -12,
    tok_number = -13,
};

enum ValueType : int{
    type_Integer,
    type_FloatPoint,
};

// 词法解析器
class Lexer {
public:
    // 使用给定的文件名创建一个词法解析器, 用来跟踪解析位置
    Lexer(std::string filename) : lastLocation(
                {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}
    virtual ~Lexer() = default;

    // 获取当前的token
    Token getCurToken() { return curTok; }

    // 移动到下一个token, 更新当前的token并返回
    Token getNextToken() { return curTok = getTok(); }

    // 移动到下一个token, 假定当前token等于希望消耗的token
    void consume(Token tok) {
        assert(tok == curTok && "consumer Token mismatch expectation");
        getNextToken();
    }

    // 返回当前的标志符identifier (前置条件: getCurToken() == tok_identifier)
    llvm::StringRef getId() {
        assert(curTok == tok_identifier);
        return identifierStr;
    }

    // 返回当前的数字 (前置条件: getCurToken() == tok_number)
    double getValue() {
        assert(curTok == tok_number);
        return numVal;
    }

    // 返回当前数字的类型 (前置条件: getCurToken() == tok_number)
    ValueType getValueType() {
        assert(curTok == tok_number);
        return numType;
    }

    // 返回当前token的位置
    Location getLastLocation() { return lastLocation; }

    // 返回当前位置在文件中的行号
    int getLine() { return curLineNum; }

    // 返回当前位置在文件中的列号
    int getCol()  { return curCol; }

private:
    // 派生类负责实现下一行的获取, 如果到达了文件的末尾则返回EOF, 通常情况下以'\n'表示一行的结束
    virtual llvm::StringRef readNextLine() = 0;

    // 返回下一个字符, 如果当前行读完, 还将使用readNextLine函数获取下一行
    int getNextChar() {
        // 当前行应该不为空, 否则表示到达了文件的结尾
        if (curLineBuffer.empty())
            return EOF;
        ++curCol;

        auto nextchar = curLineBuffer.front();
        curLineBuffer = curLineBuffer.drop_front();
        if (curLineBuffer.empty())
            curLineBuffer = readNextLine();
        if (nextchar == '\n') {
            ++curLineNum;
            curCol = 0;
        }

        return nextchar;
    }

    // 返回下一个token
    Token getTok() {
        // 跳过空格
        while (isspace(lastChar))
            lastChar = Token(getNextChar());

        // 保存当前位置
        lastLocation.line = curLineNum;
        lastLocation.col = curCol;

        // Identifier: [a-zA-Z][a-zA-Z0-9_]*
        if (isalpha(lastChar)) {
            identifierStr = (char)lastChar;
            while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
                identifierStr += (char)lastChar;
            
            // 如果是关键字则返回相应的token
            if (identifierStr == "stencil")
                return tok_stencil;
            if (identifierStr == "double")
                return tok_double;
            if (identifierStr == "float")
                return tok_float;
            if (identifierStr == "iteration")
                return tok_iteration;
            if (identifierStr == "mpiTile")
                return tok_mpiTile;
            if (identifierStr == "operation")
                return tok_operation;
            if (identifierStr == "kernel")
                return tok_kernel;
            if (identifierStr == "tile")
                return tok_tile;
            if (identifierStr == "swCacheAt")
                return tok_swCacheAt;
            if (identifierStr == "expr")
                return tok_Expr;
            
            return tok_identifier;
        }

        // 数字, 此处需要负责处理整型, 双精度浮点, 单精度浮点
        // 整型不包含小数点, 而浮点必须包含小数点
        // ([1-9][0-9]*|0)([.][0-9])?
        if (isdigit(lastChar)) {
            // 默认数字为整型
            numType = type_Integer;
            std::string numStr;
            do {
                if (lastChar == '.') 
                    numType = type_FloatPoint;
                numStr += lastChar;

                // 读取下一字符
                lastChar = Token(getNextChar());
            } while (isdigit(lastChar) || lastChar == '.');

            numVal = strtod(numStr.c_str(), nullptr);

            return tok_number;
        }

        // 注释, 直到本行行尾
        if (lastChar == '#') {
            do {
                lastChar = Token(getNextChar());
            } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

            if (lastChar != EOF)
                return getTok();
        }

        // 检查文件末尾, 不要消耗EOF
        if (lastChar == EOF)
            return tok_eof;

        // 否则, 将符号值以ascii值返回
        Token thisChar = Token(lastChar);
        lastChar = Token(getNextChar());
        return thisChar;
    }

    // 从输入读到的最后一个token
    Token curTok = tok_eof;

    // curTok 所处的位置
    Location lastLocation;

    // 如果当前Token是关键字, 则以下变量保存这个关键字
    std::string identifierStr;

    // 如果当前的Token是数字, 则以下变量分别存储其类型及数值
    ValueType numType = type_Integer;
    double numVal = 0;

    // 保存getNextChar返回的最后一个值, 词法解析器需要根据这个预先读取的一个字符决定是否终结
    // 一个token, 此处不能将其读完后再放回到输入流中
    Token lastChar = Token(' ');

    // 跟踪当前行号
    int curLineNum = 0;

    // 跟踪列号
    int curCol = 0;

    // 供派生类函数readNextline函数使用的缓冲区
    llvm::StringRef curLineBuffer = "\n";
};

// 词法分析器的派生类实现, 实现具体的readNextline函数
class LexerBuffer final : public Lexer {
public:
    LexerBuffer(const char *begin, const char *end, std::string filename)
        : Lexer(std::move(filename)), current(begin), end(end) {}
private:
    // 每次提供一行给词法分析器, 如果到达了文件末尾则返回空
    llvm::StringRef readNextLine() override {
        auto *begin = current;
        while (current <= end && *current && *current != '\n')
            ++current;
        // 包含进'\n'
        if (current <= end && *current)
            ++current;

        llvm::StringRef result{begin, static_cast<size_t>(current-begin)};

        return result;
    }

    const char *current, *end;
};

} /* End of namespace swsten */

#endif /* End of _SWSTENDSL_PARSER_LEXER_H_ */