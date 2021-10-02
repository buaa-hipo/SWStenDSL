/**
 * @file ConvertStencilToSW.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 转化pass实现
 * @version 0.1
 * @date 2021-03-22
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>

#include "Conversion/StencilToSW/ConvertStencilToSW.h"
#include "Conversion/StencilToSW/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/SW/SWDialect.h"
#include "Dialect/SW/SWOps.h"
#include "Dialect/SW/SWTypes.h"
#include "PassDetail.h"

using namespace mlir;
using namespace stencil;
using namespace scf;

namespace {
//============================================================================//
// Rewriting模式
//============================================================================//
class FuncOpLowering : public StencilOpToSWPattern<FuncOp> {
public:
    using StencilOpToSWPattern<FuncOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto funcOp = cast<FuncOp>(operation);

        // 转化原始func操作的参数
        TypeConverter::SignatureConversion result(funcOp.getNumArguments());
        for (auto &en : llvm::enumerate(funcOp.getType().getInputs())) {
            result.addInputs(en.index(), typeConverter.convertType(en.value()));
        }

        // 创建新函数类型
        auto funcType =
            FunctionType::get(result.getConvertedTypes(),
                            funcOp.getType().getResults(), funcOp.getContext());

        // 对于描述stencil计算的func部分
        if (StencilDialect::isStencilProgram(funcOp)) {
            auto newFuncOp = rewriter.create<sw::MainFuncOp>(loc, funcOp.getName(), funcType);
            // 删除自己创建的block
            rewriter.eraseBlock(&newFuncOp.getBody().getBlocks().back());

            // 复制域
            rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                        newFuncOp.end());        
            // 删除原有std终结符并在函数域中插入终结符
            auto stdReturnOp = newFuncOp.getBody().front().getTerminator();
            rewriter.eraseOp(stdReturnOp);
            rewriter.setInsertionPointToEnd(&newFuncOp.getBody().front());
            rewriter.create<sw::MainReturnOp>(loc);

            // 转换签名
            rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
        } else if (StencilDialect::isStencilIteration(funcOp)) {
            // 对于迭代定义的func部分
            auto newFuncOp = rewriter.create<sw::MainIterationFuncOp>(loc, funcOp.getName(), funcType);
            // 删除自己创建的block
            rewriter.eraseBlock(&newFuncOp.getBody().getBlocks().back());

            // 复制域
            rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                        newFuncOp.end());        
            // 删除原有std终结符并在函数域中插入终结符
            auto stdReturnOp = newFuncOp.getBody().front().getTerminator();
            rewriter.eraseOp(stdReturnOp);
            rewriter.setInsertionPointToEnd(&newFuncOp.getBody().front());
            rewriter.create<sw::MainIterationReturnOp>(loc);


            // 转换签名
            rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
        }
        // 删除原操作
        rewriter.eraseOp(funcOp);

        return success();
    }
};

class ApplyOpLowering : public StencilOpToSWPattern<stencil::ApplyOp> {
public:
    using StencilOpToSWPattern<stencil::ApplyOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands, 
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto applyOp = cast<stencil::ApplyOp>(operation);
        auto shapeOp = cast<ShapeOp>(operation);

        // 判断是否需要启用了向量化, 如果启用了, 则最内层循环的增量设置为向量宽度
        bool is_enable_vector = false;
        unsigned int vectorWidth = 1;
        auto vectorMaskedLoadOps = applyOp.getOps<vector::MaskedLoadOp>();
        if (!vectorMaskedLoadOps.empty()) {
            is_enable_vector = true;
            vectorWidth = (*(vectorMaskedLoadOps.begin())).getResult().getType().cast<VectorType>().getShape()[0];
        }

        /****************** 计算cacheRead和cacheWrite的大小 *********************/
        // 获取cache所在循环位置
        int64_t cacheAt = applyOp.getCacheAtAttr().cast<IntegerAttr>().getValue().getSExtValue();
        // 计算cacheRead的大小(此种包含需要tile的输入域, 还包含需要全部加载的参数域)
        SmallVector<unsigned int, 3> cacheRead;
        SmallVector<unsigned int, 3> parameter;
        SmallVector<Type, 3> cacheReadAttrType;
        for (auto &en : llvm::enumerate(operands)) {
            auto elem = en.value();
            auto elemShape = elem.getType().cast<sw::GridType>().getShape();
            auto elemType = elem.getType().cast<sw::GridType>().getElementType();
            SmallVector<int64_t, 3> newTypeShape;
            newTypeShape.clear();
            bool is_parameter = false;
            if (getUserOp<stencil::LoadOp>(applyOp.region().getArgument(en.index()))) {
                is_parameter = true;
            }

            for (int64_t i = 0; i < elemShape.size(); i++) {
                int64_t halo_size_l = shapeOp.getLB()[i];
                int64_t halo_size_u = elemShape[i] - shapeOp.getUB()[i];
                int64_t tile = applyOp.getTile()[i];
                if (!is_parameter && i <= cacheAt) {
                    newTypeShape.push_back(tile + halo_size_l + halo_size_u);

                } else {
                    newTypeShape.push_back(elemShape[i]);
                }
            }

            auto newType = mlir::sw::MemRefType::get(elemType, newTypeShape);
            if (is_parameter)
                parameter.push_back(en.index());
            else
                cacheRead.push_back(en.index());
            cacheReadAttrType.push_back(newType);
        }
        // 计算cacheWrite的大小
        SmallVector<Type, 3> cacheWriteAttrType;
        for (int i = 0; i < applyOp.getNumResults(); i++) {
            auto applyOpResultShape = applyOp.getResult(i).getType().cast<stencil::GridType>().getShape();
            auto cacheWriteType = applyOp.getResult(i).getType().cast<stencil::GridType>().getElementType();
            SmallVector<int64_t, 3> cacheWriteShape;
            cacheWriteShape.clear();
            for (int64_t j = 0; j < applyOpResultShape.size(); j++) {
                int64_t halo_size_l = shapeOp.getLB()[j];
                int64_t halo_size_u = applyOpResultShape[j] - shapeOp.getUB()[j];
                int64_t tile = applyOp.getTile()[j];
                if (j <= cacheAt)
                    cacheWriteShape.push_back(tile);
                else
                    cacheWriteShape.push_back(applyOpResultShape[j] - halo_size_l - halo_size_u);
            }
            auto newType = mlir::sw::MemRefType::get(cacheWriteType, cacheWriteShape);
            cacheWriteAttrType.push_back(newType);
        }

        /*************** 为每个stencil计算的结果申请临时的存储空间 ******************/
        SmallVector<Value, 10> newResults;
        for (int i = 0; i < applyOp.getNumResults(); i++) {
            auto allocType = typeConverter.convertType(applyOp.getResult(i).getType());
            auto allocOp = rewriter.create<sw::AllocOp>(loc, allocType);
            newResults.push_back(allocOp.getResult());
        }

        /********************* 创建launch操作并插入终结符号 ***********************/
        SmallVector<Value, 10> launchOpOperands;
        launchOpOperands.insert(launchOpOperands.end(), operands.begin(), operands.end());
        launchOpOperands.insert(launchOpOperands.end(), newResults.begin(), newResults.end());
        auto launchOp = rewriter.create<sw::LaunchOp>(loc, launchOpOperands, cacheReadAttrType, cacheWriteAttrType);
        rewriter.setInsertionPointToEnd(&launchOp.region().front());
        rewriter.create<sw::TerminatorOp>(loc);
        auto launchOpArg = launchOp.getBody()->getArguments();

        /******************* 创建sw.launchOp操作中嵌套sw.forOp ******************/
        rewriter.setInsertionPointToStart(&launchOp.region().front());
        // 计算每层循环的起点, 终点以及跨步, 注意此处将对多层循环进行tiling
        SmallVector<Value, 6> lbs, ubs, steps;
        auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
        // 外层循环
        for (int i = 0; i < shapeOp.getRank(); i++) {
            int64_t lb = shapeOp.getLB()[i];
            int64_t ub = shapeOp.getUB()[i];
            int64_t tile = applyOp.getTile()[i];
            int64_t size = (ub - lb) / tile;
            
            // 最外层循环起始位置为从核号, 跨步为64
            if (i == 0) {
                lbs.push_back(rewriter.create<sw::GetIDOp>(loc, rewriter.getI64Type()));
                steps.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(64), rewriter.getI64Type()));
            } else {
                lbs.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(0), rewriter.getI64Type()));
                steps.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(1), rewriter.getI64Type()));
            }
            ubs.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(size), rewriter.getI64Type()));
        }
        // 内层循环
        for (int i = 0; i < shapeOp.getRank(); i++) {
            int64_t tile = applyOp.getTile()[i];
            lbs.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(0), rewriter.getI64Type()));
            ubs.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(tile), rewriter.getI64Type()));
            int64_t step = (i == shapeOp.getRank() - 1 && is_enable_vector) ? vectorWidth : (returnOp.unroll().hasValue() ? returnOp.getUnroll()[i] : 1);
            steps.push_back(rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(step), rewriter.getI64Type()));
        }

        // 如果有参数数组, 还要在创建嵌套循环前创建加载参数数组的操作
        // 参数数组需要全部加载
        Value value_0 = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(0), rewriter.getI64Type());
        if (parameter.size() != 0) {
            auto attr_0 = rewriter.getI64IntegerAttr(0);
            auto attr_1 = rewriter.getI64IntegerAttr(1);
            SmallVector<Value, 3> index_0;
            index_0.clear();
            for (auto en : parameter) {
                // 获取并计算各个参数
                // 原输入信息
                auto elem = applyOp.region().getArgument(en);
                auto elemShape = elem.getType().cast<stencil::GridType>().getShape();
                // 对应的cacheRead参数信息
                auto elem_parameter = launchOp.getCacheReadAttributions()[en];
                int64_t dma_size = 1;
                for (int i = 0; i < elemShape.size(); i++) {
                    dma_size *= elemShape[i];
                    index_0.push_back(value_0);
                }
                auto attr_dma_size = rewriter.getI64IntegerAttr(dma_size);
                // DMA参数数组
                rewriter.create<sw::MemcpyToLDMOp>(loc, launchOpArg[en], elem_parameter, index_0, attr_1, attr_dma_size, attr_0, attr_0);
            }
        }

        // 创建sw.for嵌套循环
        SmallVector<Value, 6> inductionVars; ///< 记录嵌套循环的迭代变量
        SmallVector<Value, 3> innerLoopBasePos; ///< 记录内层循环访问cacheRead数组的出发位置, 将来加上内层循环索引即可得基准位置
        sw::ForOp forOp = rewriter.create<sw::ForOp>(loc, lbs[0], ubs[0], steps[0]);
        inductionVars.push_back(forOp.getInductionVar());
        auto loop_tile = applyOp.getTile();
        for (int64_t i = 1; i < shapeOp.getRank()*2; i++) {
            // 创建嵌套for操作并插入终结符号
            rewriter.setInsertionPointToEnd(forOp.getBody());
            // 插入DMA到LDM操作
            if (i-1 == cacheAt) {
                for (auto en : cacheRead) {
                    // 获取并计算各个参数
                    // 原输入参数信息
                    auto elem = applyOp.region().getArgument(en);
                    auto elemShape = elem.getType().cast<stencil::GridType>().getShape();
                    // 对应的cacheRead参数信息
                    auto elem_cacheRead = launchOp.getCacheReadAttributions()[en];
                    auto elem_cacheReadShape = elem_cacheRead.getType().cast<sw::GridType>().getShape();

                    // 计算索引
                    SmallVector<Value, 3> indexArray;
                    indexArray.clear();
                    for (int iter = 0; iter < elemShape.size(); iter++) {
                        if (iter <= cacheAt) {
                            Value tile_size =  rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(loop_tile[iter]), rewriter.getI64Type());
                            Value index = rewriter.create<sw::MuliOp>(loc, rewriter.getI64Type(), inductionVars[iter], tile_size);
                            indexArray.push_back(index);
                        } else {
                            indexArray.push_back(value_0);
                        }
                    }

                    // 计算z_dim参数
                    int64_t zDim = (elemShape.size() == 3) ? elem_cacheReadShape[0] : 1;
                    auto zDimAttr = rewriter.getI64IntegerAttr(zDim);
                    // 计算cnt参数, 按面加载, 不计算三维情况的最高维度
                    int64_t cnt = 1;
                    int iter = 0;
                    if (elemShape.size() == 3) {
                        iter = 1;
                    }
                    for (; iter < elemShape.size(); iter++) {
                        cnt *= elem_cacheReadShape[iter];
                    }
                    auto cntAttr = rewriter.getI64IntegerAttr(cnt);
                    // 计算bsize参数
                    int64_t bsize = elem_cacheReadShape[elem_cacheReadShape.size()-1];
                    auto bsizeAttr = rewriter.getI64IntegerAttr(bsize);
                    // 计算stride参数
                    int64_t stride = elemShape[elemShape.size()-1]  - bsize;
                    auto strideAttr = rewriter.getI64IntegerAttr(stride);
                    // 创建操作
                    rewriter.create<sw::MemcpyToLDMOp>(loc, launchOpArg[en], elem_cacheRead, indexArray, zDimAttr, cntAttr, strideAttr, bsizeAttr);
                }
            }

            // 插入该层循环的出发位置
            if (i-1 < shapeOp.getRank()) {
                Value halo_l = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(shapeOp.getLB()[i-1]), rewriter.getI64Type());
                if (i-1 <= cacheAt) {
                    // 对于cacheAt之外的外层循环, 对应的出发位置为相应的halo_left
                    innerLoopBasePos.push_back(halo_l);
                } else {
                    // 对于cacheAt之内的最外层循环, 对应的出发位置为外层循环的索引*相应的tile值+halo_left
                    Value tile = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(loop_tile[i-1]), rewriter.getI64Type());
                    Value multiResult = rewriter.create<sw::MuliOp>(loc, rewriter.getI64Type(), inductionVars[i-1], tile);
                    Value multiAddResult = rewriter.create<sw::AddiOp>(loc, rewriter.getI64Type(), multiResult, halo_l);
                    innerLoopBasePos.push_back(multiAddResult);
                }
            }
            forOp = rewriter.create<sw::ForOp>(loc, lbs[i], ubs[i], steps[i]);
            inductionVars.push_back(forOp.getInductionVar());
            // 插入DMA到MEM操作
            if (i-1 == cacheAt) {
                for (auto en : llvm::enumerate(launchOp.getCacheWriteAttributions())) {
                    // 获取并计算各个参数
                    // 原结果信息
                    auto elem = applyOp.getResult(en.index());
                    auto elemShape = elem.getType().cast<stencil::GridType>().getShape();
                    // 对应的cacheWrite信息
                    auto elem_cacheWrite = en.value();
                    auto elem_cacheWriteShape = elem_cacheWrite.getType().cast<sw::GridType>().getShape();

                    // 计算索引, 写回时要加上偏移量, 确保位置正确
                    SmallVector<Value, 3> indexArray;
                    indexArray.clear();
                    for (int iter = 0; iter < elemShape.size(); iter++) {
                        Value halo_l = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(shapeOp.getLB()[iter]), rewriter.getI64Type());
                        if (iter <= cacheAt) {
                            Value tile_size = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(loop_tile[iter]), rewriter.getI64Type());
                            Value base = rewriter.create<sw::MuliOp>(loc, rewriter.getI64Type(), inductionVars[iter], tile_size);
                            Value index = rewriter.create<sw::AddiOp>(loc, rewriter.getI64Type(), base, halo_l);
                            indexArray.push_back(index);
                        } else {
                            indexArray.push_back(halo_l);
                        }
                    }

                    // 计算z_dim参数
                    int64_t zDim = (elemShape.size() == 3) ? elem_cacheWriteShape[0] : 1;
                    auto zDimAttr = rewriter.getI64IntegerAttr(zDim);
                    // 计算cnt参数, 按面加载, 不计算三维情况的最高维度
                    int64_t cnt = 1;
                    int iter = 0;
                    if (elemShape.size() == 3) {
                        iter = 1;
                    }
                    for (; iter < elemShape.size(); iter++) {
                        cnt *= elem_cacheWriteShape[iter];
                    }
                    auto cntAttr = rewriter.getI64IntegerAttr(cnt);
                    // 计算bsize参数
                    int64_t bsize = elem_cacheWriteShape[elem_cacheWriteShape.size()-1];
                    auto bsizeAttr = rewriter.getI64IntegerAttr(bsize);
                    // 计算stride参数
                    int64_t stride = elemShape[elemShape.size()-1] - bsize;
                    auto strideAttr = rewriter.getI64IntegerAttr(stride);

                    // 创建操作
                    rewriter.create<sw::MemcpyToMEMOp>(loc, elem_cacheWrite, launchOpArg[operands.size() + en.index()], indexArray, zDimAttr, cntAttr, strideAttr, bsizeAttr);
                }
            }
            rewriter.create<sw::YieldOp>(loc);
        }

        // 在最内层循环插入终结符号
        rewriter.setInsertionPointToEnd(forOp.getBody());
        rewriter.create<sw::YieldOp>(loc);
        // NOTICE: 此处强制约定最内层循环的前2-3个AddiOp操作对应的为基准位置, 方便后面的操作查找基准位置
        // 在最内层循环开头插入索引基点计算, 基准位置为出发位置加上相应的内层循环索引
        rewriter.setInsertionPointToStart(forOp.getBody());
        for (int i = 0; i < shapeOp.getLB().size(); i++) {
            rewriter.create<sw::AddiOp>(loc, rewriter.getI64Type(), innerLoopBasePos[i], inductionVars[i+shapeOp.getLB().size()]);
        }

        // 转化apply操作签名
        TypeConverter::SignatureConversion result(applyOp.getNumOperands());
        for (auto &en : llvm::enumerate(applyOp.getOperands())) {
            result.remapInput(en.index(), operands[en.index()]);
        }
        rewriter.applySignatureConversion(&applyOp.region(), result);
        // 复制域
        rewriter.mergeBlockBefore(
            applyOp.getBody(),
            forOp.getBody()->getTerminator());

        /************************ 释放之前申请的临时空间 **************************/
        rewriter.setInsertionPoint(
            applyOp.getParentRegion()->back().getTerminator());
        for (int i = 0; i < newResults.size(); i++) {
            rewriter.create<sw::DeAllocOp>(loc, newResults[i]);
        }

        // 替换applyOp
        rewriter.replaceOp(applyOp, newResults);
        return success();
    }
};

class AccessOpLowering : public StencilOpToSWPattern<stencil::AccessOp> {
public:
    using StencilOpToSWPattern<stencil::AccessOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation , ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto accessOp = cast<stencil::AccessOp>(operation);
        auto offsetOp = cast<OffsetOp>(accessOp.getOperation());

        // 获取基准位置, 按照约定, 收集3个即可
        SmallVector<Value, 3> basePos;
        auto forOp = operation->getParentOfType<sw::ForOp>();
        forOp.walk([&](sw::AddiOp addiOp) {
            if (basePos.size() > 3)
                return;
            basePos.push_back(addiOp.getResult());
        });

        // 计算相应的偏移量
        SmallVector<Value, 3> loadOffset;
        auto offset = offsetOp.getOffset();
        for (auto elem : llvm::enumerate(offset)) {
            Value offset_value = 
                rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(elem.value()), rewriter.getI64Type());
            Value offset_index = 
                rewriter.create<sw::AddiOp>(loc, rewriter.getI64Type(), basePos[elem.index()], offset_value);
            loadOffset.push_back(offset_index);
        }
        // 找到对应的cacheRead数组
        auto launchOp = operation->getParentOfType<sw::LaunchOp>();
        auto launchOpOperands = launchOp.getOperands();
        auto launchOpCacheRead = launchOp.getCacheReadAttributions();
        int cacheReadIndex;
        for (auto elem : llvm::enumerate(launchOp.getOperands())) {
            if (elem.value() == operands[0])
                cacheReadIndex = elem.index();
        }
        // 替换Op
        rewriter.replaceOpWithNewOp<sw::LoadOp>(operation, launchOpCacheRead[cacheReadIndex], loadOffset);
        return success();
    }
};

class LoadOpLowering : public StencilOpToSWPattern<stencil::LoadOp> {
public:
    using StencilOpToSWPattern<stencil::LoadOp>::StencilOpToSWPattern;
    
    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto loadOp = cast<stencil::LoadOp>(operation);
        auto offsetOp = cast<OffsetOp>(operation);
        // 计算位置
        SmallVector<Value, 3> loadOffset;
        for (auto elem : offsetOp.getOffset()) {
            auto offset_index = 
                rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(elem), rewriter.getI64Type());
            loadOffset.push_back(offset_index);
        }
        // 找到对应的cacheRead数组
        auto launchOp = operation->getParentOfType<sw::LaunchOp>();
        auto launchOpOperands = launchOp.getOperands();
        auto launchOpCacheRead = launchOp.getCacheReadAttributions();
        int cacheReadIndex;
        for (auto elem : llvm::enumerate(launchOp.getOperands())) {
            if (elem.value() == operands[0]) {
                cacheReadIndex = elem.index();
                break;
            }
        }

        // 替换Op
        rewriter.replaceOpWithNewOp<sw::LoadOp>(operation, launchOpCacheRead[cacheReadIndex], loadOffset);
        return success();
    }
};

class StoreOpLowering : public StencilOpToSWPattern<stencil::StoreOp> {
public:
    using StencilOpToSWPattern<stencil::StoreOp>::StencilOpToSWPattern;
    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto storeOp = cast<stencil::StoreOp>(operation);

        bool is_enable_vector = false;
        
        // 获取对应的returnOp
        OpOperand *operand = valueToOperand[storeOp.res()];
        auto returnOp = cast<stencil::ReturnOp>(operand->getOwner());

        // 获取相应的launchOp
        auto launchOp = operation->getParentOfType<sw::LaunchOp>();

        // 如果store操作存在一个输入(循环展开时有的store可能没有相应待存储的数值, 
        // 特别是循环维度无法被展开层数整除时)
        if (storeOp.operands().size() == 1) {
            if (operands[0].getType().isa<VectorType>())
                is_enable_vector = true;

            // 获取问题域维度
            auto haloLArray = valueToLB[returnOp.operands()[0]];
            auto domainDim = haloLArray.size();
            // 获取相应的展开参数和相应的展开维度
            unsigned int unrollFac = returnOp.getUnrollFactor();
            size_t unrollDim = returnOp.getUnrollDimension();

            // 计算要写回到哪一个cacheWrite数组
            // NOTICE:
            // return操作支持循环展开和循环融合下的数据返回,
            // 我们强制约定两个applyOp循环展开后的returnOp的参数分别为(%0, %1)和(%2, %3)
            // 那么两个apply循环融合之后原来的两个returnOp将合并为一个returnOp, 且参数顺序
            // 为(%0, %2, %1, %3), 且融合后的apply操作有两个结果输出, 下面的表达式将计算出
            // return操作的某个参数应当写回到哪个结果输出中.
            unsigned int bufferCount = operand->getOperandNumber() % unrollFac;

            // 获取待写入的cacheWrite数组
            Value cacheWriteArray = launchOp.getCacheWriteAttributions()[bufferCount];

            // 计算写回的偏移量
            // 获取基准位置, 按照约定, 收集3个即可
            SmallVector<Value, 3> basePos;
            auto forOp = operation->getParentOfType<sw::ForOp>();
            forOp.walk([&](sw::AddiOp addiOp) {
                if (basePos.size() > 3)
                    return;
                basePos.push_back(addiOp.getResult());
            });

            // 判断循环展开位置, 计算偏移
            unsigned int unroll_offset = operand->getOperandNumber() / unrollFac;
            Value unrollOffsetValue = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(unroll_offset), rewriter.getI64Type());
            SmallVector<Value, 3> storePos;

            for (int i = 0; i < domainDim; i++) {
                Value halo_l = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(haloLArray[i]), rewriter.getI64Type());
                Value base = rewriter.create<sw::SubiOp>(loc, rewriter.getI64Type(), basePos[i], halo_l);
                if (i == unrollDim) {
                    Value pos = rewriter.create<sw::AddiOp>(loc, rewriter.getI64Type(), base, unrollOffsetValue);
                    storePos.push_back(pos);
                } else {
                    storePos.push_back(base);
                }
            }
            // 创建写回操作
            if (is_enable_vector)
                rewriter.create<sw::VectorStoreUOp>(loc, operands[0], cacheWriteArray, storePos);
            else
                rewriter.create<sw::StoreOp>(loc, operands[0], cacheWriteArray, storePos);
        }
        // 删除原来的Op
        rewriter.eraseOp(operation);
        return success();
    }
};

class ReturnOpLowering : public StencilOpToSWPattern<stencil::ReturnOp> {
public:
    using StencilOpToSWPattern<stencil::ReturnOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        // 直接删除
        rewriter.eraseOp(operation);
        return success();
    }
};

class CopyOpLowering : public StencilOpToSWPattern<stencil::CopyOp> {
public:
    using StencilOpToSWPattern<stencil::CopyOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto copyOp = cast<stencil::CopyOp>(operation);
        // 获取参数
        auto result = operands[0];
        auto output = operands[1];

        // 获取使用result的MemcpyToMEM操作, 并将更改其写回数组
        // 获取使用result的launchOp, 并更改其参数
        for (auto user : result.getUsers()) {
            if (auto op = dyn_cast<sw::MemcpyToMEMOp>(user)) {
                op.setOperand(1, output);
            }
            if (auto op = dyn_cast<sw::LaunchOp>(user)) {
                for (auto elem : llvm::enumerate(op.getOperands())) {
                    if (elem.value() == result) {
                        op.setOperand(elem.index(), output);
                    }
                }
            }
        }

        // 删除调对应的Alloc操作
        auto allocOp = result.getDefiningOp();
        rewriter.eraseOp(allocOp);
        // 删除掉对应的deAlloc操作
        auto deAllocOp = getUserOp<sw::DeAllocOp>(result);
        rewriter.eraseOp(deAllocOp);
        // 删除copy操作
        rewriter.eraseOp(operation);
        return success();
    }
};

class ConstantOpLowering : public StencilOpToSWPattern<ConstantOp> {
public:
    using StencilOpToSWPattern<ConstantOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto constantOp = cast<ConstantOp>(operation);
        auto value = constantOp.value();
        
        // 直接替换
        Value constantValue = rewriter.create<sw::ConstantOp>(loc, value, value.getType());
        rewriter.replaceOp(constantOp, constantValue);
        return success();
    }
};

class AddFOpLowering : public StencilOpToSWPattern<AddFOp> {
public:
    using StencilOpToSWPattern<AddFOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto addfOp = cast<AddFOp>(operation);

        // 直接替换
        auto result = addfOp.getResult();
        Value addfValue = rewriter.create<sw::AddfOp>(loc, result.getType(), operands[0], operands[1]);
        rewriter.replaceOp(addfOp, addfValue);
        return success();
    }
};

class SubFOpLowering : public StencilOpToSWPattern<SubFOp> {
public:
    using StencilOpToSWPattern<SubFOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto subfOp = cast<SubFOp>(operation);

        // 直接替换
        auto result = subfOp.getResult();
        Value subfValue = rewriter.create<sw::SubfOp>(loc, result.getType(), operands[0], operands[1]);
        rewriter.replaceOp(subfOp, subfValue);
        return success();
    }
};

class MulFOpLowering : public StencilOpToSWPattern<MulFOp> {
public:
    using StencilOpToSWPattern<MulFOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto mulfOp = cast<MulFOp>(operation);

        // 直接替换
        auto result = mulfOp.getResult();
        Value mulfValue = rewriter.create<sw::MulfOp>(loc, result.getType(), operands[0], operands[1]);
        rewriter.replaceOp(mulfOp, mulfValue);
        return success();
    }
};

class DivFOpLowering : public StencilOpToSWPattern<DivFOp> {
public:
    using StencilOpToSWPattern<DivFOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto divfOp = cast<DivFOp>(operation);

        // 直接替换
        auto result = divfOp.getResult();
        Value divfValue = rewriter.create<sw::DivfOp>(loc, result.getType(), operands[0], operands[1]);
        rewriter.replaceOp(divfOp, divfValue);
        return success();
    }
};

class IterationOpLowering : public StencilOpToSWPattern<IterationOp> {
public:
    using StencilOpToSWPattern<IterationOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto iterationOp = cast<stencil::IterationOp>(operation);
        
        auto bindParamNum = iterationOp.getBindParamNum();
        auto iterNum = iterationOp.getIterNum();
        auto funcName = iterationOp.getStencilFuncName();

        SmallVector<Value, 8> operands1, operands2;
        for (int i = 0; i < bindParamNum; i++) {
            operands1.push_back(operands[i]);
            operands2.push_back(operands[i+bindParamNum]);
        }

        // 创建for循环
        auto lb = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(0), rewriter.getI64Type());
        auto ub = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(iterNum), rewriter.getI64Type());
        auto step = rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(1), rewriter.getI64Type());
        auto forOp = rewriter.create<sw::ForOp>(loc, lb, ub, step);
        // 在嵌套循环中创建sw.launch_main_func, 如果启用了mpi还需要创建相关的函数
        rewriter.setInsertionPointToStart(forOp.getBody());
        // 如果启用了mpi则获取当前进程在通信域中的id
        Value mpiRank;
        if (iterationOp.isEnableMPI()) 
            mpiRank = rewriter.create<sw::GetMpiRankOp>(loc);
        // 计算一次
        rewriter.create<sw::LaunchMainFuncOp>(loc, rewriter.getSymbolRefAttr(funcName), operands1);
        // 如果启用了mpi, 则交换该次计算的结果, 该结果将作为下一次计算的输入
        if (iterationOp.isEnableMPI()) {
            rewriter.create<sw::MpiExchangeHaloOp>(loc, operands1[1], mpiRank,\
                iterationOp.mpiTile().getValue(), iterationOp.mpiHaloL().getValue(), iterationOp.mpiHaloU().getValue());
        }
        // 计算一次
        rewriter.create<sw::LaunchMainFuncOp>(loc, rewriter.getSymbolRefAttr(funcName), operands2);
        // 如果启用了mpi, 则交换该次计算的结果, 该结果将作为下一次计算的输入
        if (iterationOp.isEnableMPI()) {
            rewriter.create<sw::MpiExchangeHaloOp>(loc, operands2[1], mpiRank,\
                iterationOp.mpiTile().getValue(), iterationOp.mpiHaloL().getValue(), iterationOp.mpiHaloU().getValue());
        }
        // 插入forOp终结符号
        rewriter.create<sw::YieldOp>(loc);

        // 删除iteration操作
        rewriter.eraseOp(operation);
        return success();
    }
};

class VectorMaskedLoadOpLowering : public StencilOpToSWPattern<vector::MaskedLoadOp> {
public:
    using StencilOpToSWPattern<vector::MaskedLoadOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto vectorMaskedLoadOp = cast<vector::MaskedLoadOp>(operation);

        // 获取对应的stencil.castToMemRef, constant_mask, 以及该op使用的constant
        // 这三者只是为了适配接口, 可以删除
        auto castToMemRefOp = operands[0].getDefiningOp();
        auto constantMaskOp = operands[1].getDefiningOp();
        auto constantOp = operands[2].getDefiningOp();
        auto offsetOp = cast<OffsetOp>(castToMemRefOp);
        auto resType = vectorMaskedLoadOp.getResult().getType().cast<VectorType>();
        unsigned int vectorWidth = resType.getShape()[0];
        auto elementType = resType.getElementType();

        // 获取基准位置, 按照约定, 收集3个即可
        SmallVector<Value, 3> basePos;
        auto forOp = operation->getParentOfType<sw::ForOp>();
        forOp.walk([&](sw::AddiOp addiOp) {
            if (basePos.size() > 3)
                return;
            basePos.push_back(addiOp.getResult());
        });

        // 计算位置
        SmallVector<Value, 3> vectorLoadOffset;
        for (auto elem : llvm::enumerate(offsetOp.getOffset())) {
            auto offset_value = 
                rewriter.create<sw::ConstantOp>(loc, rewriter.getI64IntegerAttr(elem.value()), rewriter.getI64Type());
            Value offset_index = 
                rewriter.create<sw::AddiOp>(loc, rewriter.getI64Type(), basePos[elem.index()], offset_value);
            vectorLoadOffset.push_back(offset_index);
        }

        // 找到对应的cacheRead数组
        auto launchOp = operation->getParentOfType<sw::LaunchOp>();
        auto launchOperands = launchOp.getOperands();
        // getOperand返回的是未修改之前的op定义的值, 此处的castToMemReefOperand为applyOp
        // 定义的value, 而lowering函数(即本下降函数)的operands为修改之后的op定义的value,
        // 因此,此处不能loadlowring或accesslowering那样, 通过launchOp去寻找对应的cache数组
        // 位置下标(主要是此处无法通过castToMemRefpOp的getOperand函数获取到更改前面依赖op后的value)
        // 考虑到applyOp和launchOp在读数组上顺序相同, 故在进行所有下降之前先建立apply的operands与位置
        // 的绑定关系, 此处即可直接利用该绑定关系获取cacheRead的位置索引
        Value castToMemRefOperand = castToMemRefOp->getOperand(0);
        auto launchOpCacheRead = launchOp.getCacheReadAttributions();
        int cacheReadIndex = -1;
        if (valueToApplyOpIndex.find(castToMemRefOperand) != valueToApplyOpIndex.end()) {
            cacheReadIndex = valueToApplyOpIndex[castToMemRefOperand];
        }

        // 创建一个constant op 用来vector 变量声明
        SmallVector<APFloat, 4> attrValue;
        bool is_double = (elementType.cast<FloatType>().getWidth() == 64);
        for (int i = 0; i < vectorWidth; i++) {
            if (is_double)
                attrValue.push_back(APFloat((double)0));
            else
                attrValue.push_back(APFloat((float)0));
        }

        auto newConstantOpAttr = DenseFPElementsAttr::get(VectorType::get(vectorWidth, elementType), attrValue);
        Value constantValue = rewriter.create<sw::ConstantOp>(loc, newConstantOpAttr, VectorType::get(vectorWidth, elementType));

        // 替换Op
        rewriter.create<sw::VectorLoadUOp>(loc, constantValue, launchOpCacheRead[cacheReadIndex], vectorLoadOffset);

        // 删除无用的Op
        rewriter.eraseOp(castToMemRefOp);
        rewriter.eraseOp(constantMaskOp);
        rewriter.eraseOp(constantOp);
        rewriter.replaceOp(vectorMaskedLoadOp, constantValue);
        return success();
    }
};

class VectorBroadCastOpLowering : public StencilOpToSWPattern<vector::BroadcastOp> {
public:
    using StencilOpToSWPattern<vector::BroadcastOp>::StencilOpToSWPattern;

    LogicalResult
    matchAndRewrite(Operation *operation, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        auto loc = operation->getLoc();
        auto vectorBroadCastOp = cast<vector::BroadcastOp>(operation);
        auto resType = vectorBroadCastOp.getResult().getType().cast<VectorType>();
        unsigned int vectorWidth = resType.getShape()[0];

        // 直接替换为sw::vectorBroadCastOp
        Value newResult = rewriter.create<sw::VectorBroadCastOp>(loc, operands[0], vectorWidth);
        rewriter.replaceOp(operation, newResult);
        return success();
    }
};

//============================================================================//
// 转换目标
//============================================================================//
class StencilToSWTarget : public ConversionTarget {
public:
    explicit StencilToSWTarget(MLIRContext &context)
        : ConversionTarget(context) {}

    bool isDynamicallyLegal(Operation *op) const override {
        if (auto funcOp = dyn_cast<FuncOp>(op)) {
            return !StencilDialect::isStencilProgram(funcOp) && !StencilDialect::isStencilIteration(funcOp);
        }
    }
};

//============================================================================//
// Rewriting pass
//============================================================================//
struct StencilToSWPass : public StencilToSWPassBase<StencilToSWPass> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<AffineDialect>();
    }
    void runOnOperation() override;
};

void StencilToSWPass::runOnOperation() {
    OwningRewritePatternList patterns;
    auto module = getOperation();
    
    // 记录每个apply操作的下界, apply每个输入以及其中的store操作的参数与下界的绑定, 这样在变换相关操作时可以
    // 直接使用这个绑定关系
    DenseMap<Value, Index> valueToLB;
    DenseMap<Value, unsigned int> valueToApplyOpIndex;
    module.walk([&](stencil::ApplyOp applyOp) {
        auto lb = cast<ShapeOp>(applyOp.getOperation()).getLB();
        // 建立参数和下界的绑定关系
        for (auto en : llvm::enumerate(applyOp.getOperands())) {
            valueToLB[applyOp.getBody()->getArgument(en.index())] = lb;
            valueToApplyOpIndex[applyOp.getBody()->getArgument(en.index())] = en.index();
        }
        // 建立return操作的参数与下界的绑定关系
        for (auto value : applyOp.getBody()->getTerminator()->getOperands()) {
            valueToLB[value] = lb;
        }
    });

    DenseMap<Value, OpOperand *> valueToOperand;
    module.walk([&](stencil::StoreOp storeOp) {
        valueToOperand[storeOp.res()] = storeOp.getReturnOpOperand();
    });

    StencilTypeConverter typeConverter(module.getContext());
    populateStencilToSWConversionPatterns(typeConverter, valueToLB, 
        valueToOperand, valueToApplyOpIndex, patterns);

    StencilToSWTarget target(*(module.getContext()));
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<SCFDialect>();
    target.addLegalDialect<mlir::sw::SWDialect>();
    target.addDynamicallyLegalOp<FuncOp>();
    target.addLegalOp<vector::ConstantMaskOp>();
    target.addLegalOp<mlir::stencil::CastToMemRefOp>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    if (failed(applyFullConversion(module, target, patterns))) {
        signalPassFailure();
    }
}

} // end of anonymous namespace

namespace mlir {
namespace stencil {

// 填充转换模式列表
void populateStencilToSWConversionPatterns(
    StencilTypeConverter &typeConverter, DenseMap<Value, Index> &valueToLB,
        DenseMap<Value, OpOperand *> &valueToOperand, 
        DenseMap<Value, unsigned int> &valueToApplyOpIndex, 
        mlir::OwningRewritePatternList &patterns) {
    patterns.insert<FuncOpLowering, ApplyOpLowering, AccessOpLowering,
                    LoadOpLowering, StoreOpLowering, ReturnOpLowering,
                    CopyOpLowering, ConstantOpLowering, AddFOpLowering,
                    SubFOpLowering, MulFOpLowering, DivFOpLowering, 
                    IterationOpLowering, VectorMaskedLoadOpLowering, 
                    VectorBroadCastOpLowering>(typeConverter, valueToLB, 
                    valueToOperand, valueToApplyOpIndex);
}

//============================================================================//
// Stencil类型转换器
//============================================================================//
StencilTypeConverter::StencilTypeConverter(MLIRContext *context_)
    : context(context_) {
    // 给field类型添加类型转换
    addConversion([&](stencil::GridType type) {
        return mlir::sw::MemRefType::get(type.getElementType(), type.getShape());
    });

    addConversion([&](Type type) -> Optional<Type> {
        if (auto gridType = type.dyn_cast<stencil::GridType>())
            return llvm::None;
        return type;
    });
}

//============================================================================//
// Stencil模式基类
//============================================================================//
StencilToSWPattern::StencilToSWPattern(
    StringRef rootOpName, StencilTypeConverter &typeConverter,
    DenseMap<Value, Index> &valueToLB,
    DenseMap<Value, OpOperand *> &valueToOperand, 
    DenseMap<Value, unsigned int> &valueToApplyOpIndex,
    PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, typeConverter.getContext()),
    typeConverter(typeConverter), valueToLB(valueToLB),
    valueToOperand(valueToOperand), valueToApplyOpIndex(valueToApplyOpIndex) {}

// 获取嵌套循环的迭代变量
SmallVector<Value, 6>
StencilToSWPattern::getInductionVars(Operation *operation) const {
    SmallVector<Value, 6> inductionVariables;

    // 获取包含本op的launchOp
    auto launchOp = operation->getParentOfType<sw::LaunchOp>();

    // 遍历其中的sw::forOp, 获取循环变量
    launchOp.walk([&](sw::ForOp forOp) {
        inductionVariables.push_back(forOp.getInductionVar());
    });

    return inductionVariables;
}

} // end of namespace stencil
} // end of namespace mlir

std::unique_ptr<Pass> mlir::createConvertStencilToSWPass() {
    return std::make_unique<StencilToSWPass>();
}