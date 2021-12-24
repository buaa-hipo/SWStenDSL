/**
 * @file StencilKernelFusion.cpp
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief kernel融合pass实现
 * @version 0.1
 * @date 2021-08-14
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/UseDefLists.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/Passes.h"
#include "PassDetail.h"

using namespace mlir;
using namespace stencil;

namespace {
//============================================================================//
// base class for kernel fusion patterns
//============================================================================//
struct StencilKernelFusionPattern : public ApplyOpPattern {
    StencilKernelFusionPattern(MLIRContext *context, PatternBenefit benefit = 1)
        : ApplyOpPattern(context, benefit) {};
    
    // 检查当前的apply操作是否是指定apply操作唯一的消费者
    bool hasSingleConsumer(stencil::ApplyOp producerOp,
                            stencil::ApplyOp applyOp) const {
        return llvm::all_of(producerOp.getOperation()->getUsers(),
                            [&](Operation *op) {return op == applyOp; });
    }
};

//============================================================================//
// pattern for inlining producer into consumer
// the producer should have only a single consumer
//============================================================================//
struct StencilKernelFusionInliningRewrite : public StencilKernelFusionPattern {
    using StencilKernelFusionPattern::StencilKernelFusionPattern;

    // 将producer内连到consumer中
    LogicalResult inlineProducer(stencil::ApplyOp producerOp,
                                 stencil::ApplyOp consumerOp,
                                 ValueRange producerResults,
                                 PatternRewriter &rewriter) const {
        // 合并生产者和消费者的参数
        SmallVector<Value, 10> buildOperands = producerOp.getOperands();
        buildOperands.insert(buildOperands.end(), consumerOp.getOperands().begin(),
                             consumerOp.getOperands().end());
        
        // 创建一个新的apply op
        auto loc = consumerOp.getLoc();
        auto shapeOp = cast<ShapeOp>(consumerOp.getOperation());
        auto consumerOpLB = shapeOp.getLB();
        auto consumerOpUB = shapeOp.getUB();
        auto consumerOpTile = consumerOp.getTile();
        int64_t consumerOpCacheAt = consumerOp.getCacheAtAttr().cast<IntegerAttr>().getValue().getSExtValue();
        auto buildOp = rewriter.create<stencil::ApplyOp>(
            loc, buildOperands, consumerOpLB, consumerOpUB, consumerOpTile,
            consumerOpCacheAt, consumerOp.getResultTypes());
        rewriter.mergeBlocks(consumerOp.getBody(), buildOp.getBody(),
                             buildOp.getBody()->getArguments().take_back(
                             consumerOp.getNumOperands()));
        
        // 计算生产者结果的在新apply op中的替换索引
        DenseMap<Value,size_t>replacementIndex;
        for (auto en : llvm::enumerate(buildOperands)) {
            auto pos = std::find(producerOp.getResults().begin(),
                                 producerOp.getResults().end(), en.value());
            
            if (pos != producerOp.getResults().end()) {
                replacementIndex[buildOp.getBody()->getArgument(en.index())] = 
                    std::distance(producerOp.getResults().begin(), pos);
            }
        }

        // 删除producer op中的store op
        producerOp.walk([&](Operation *op) {
            if (auto storeOp = dyn_cast<stencil::StoreOp>(op)) {
                assert(storeOp.operands().size() == 1 &&
                       "expected store result ops to store a value");
                rewriter.replaceOp(storeOp, storeOp.operands());
            }
        });

        // 遍例访问生产者结果的access操作, 并用实际的计算进行替换
        DenseMap<Value, SmallVector<std::tuple<Index, Value>, 10>> inliningCache;
        rewriter.setInsertionPoint(buildOp);
        buildOp.walk([&](stencil::AccessOp accessOp) {
            if (replacementIndex.count(accessOp.field()) != 0) {
                // 获取该操作的偏移量
                Index offset = cast<OffsetOp>(accessOp.getOperation()).getOffset();
                // 检查生产者结果访问中该偏移量的访问是否已经内联, 如果内联则直接使用内联后的值
                if (inliningCache.count(accessOp.field()) != 0) {
                    for (auto it : inliningCache[accessOp.field()]) {
                        if (std::get<0>(it) == offset &&
                            std::get<1>(it).getParentRegion()->isAncestor(
                                accessOp->getParentRegion())) {
                            rewriter.replaceOp(accessOp, std::get<1>(it));
                            return;
                        }
                    }
                }
                // 如果还没有内联则需要将producer拷贝进去
                auto cloneOp = cast<stencil::ApplyOp>(rewriter.clone(*producerOp));
                cloneOp.walk(
                    [&](stencil::ShiftOp shiftOp) {shiftOp.shiftByOffset(offset); });
                rewriter.mergeBlockBefore(cloneOp.getBody(), accessOp, 
                                            buildOp.getBody()->getArguments().take_front(
                                            producerOp.getNumOperands()));
                rewriter.eraseOp(cloneOp);
                // 将消费者access操作替换为相应的生产者的计算的值
                // 向上寻找returnOp
                auto returnOp = accessOp.getOperation()->getPrevNode();
                while (returnOp && !isa<stencil::ReturnOp>(returnOp))
                    returnOp = returnOp->getPrevNode();
                assert(returnOp && "expected to find a return op in producer kernel");

                auto operand = returnOp->getOperand(replacementIndex[accessOp.field()]);
                rewriter.replaceOp(accessOp, operand);
                rewriter.eraseOp(returnOp);
                // 记录内联的计算结果
                inliningCache[accessOp.field()].push_back(std::make_tuple(offset, operand));
            }
        });

        // 清理未使用的或者重复的参数
        auto newOp = cleanupOpArguments(buildOp, rewriter);
        assert(newOp && "expected op to have unused producer consumer edges");

        // 更新
        rewriter.replaceOp(consumerOp, newOp.getResults());
        rewriter.eraseOp(buildOp);
        rewriter.eraseOp(producerOp);
        return success();
    }

    LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                  PatternRewriter &rewriter) const override {
        // 寻找该apply操作的对应apply生产者
        for (auto operand : applyOp.operands()) {
            if (auto producerOp = 
                    dyn_cast_or_null<stencil::ApplyOp>(operand.getDefiningOp())) {
                if (hasSingleConsumer(producerOp, applyOp)) {
                    return inlineProducer(producerOp, applyOp, producerOp.getResults(),
                            rewriter);
                }
            }
        }
        return failure();
    }
};

//============================================================================//
// Rewriting pass
//============================================================================//
struct StencilKernelFusionPass : public StencilKernelFusionPassBase<StencilKernelFusionPass> {
    void runOnOperation() override;
};

void StencilKernelFusionPass::runOnOperation() {
    auto moduleOp = getOperation();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<StencilKernelFusionInliningRewrite>(&getContext());

    moduleOp.walk([&](FuncOp funcOp) {
        if (StencilDialect::isStencilProgram(funcOp)) {
            // 执行kernel fusion之前不能进行循环展开操作
            bool hasUnrolledStencils = false;
            funcOp.walk([&](stencil::ReturnOp returnOp) {
                if (returnOp.unroll().hasValue()) {
                    returnOp.emitOpError("execute stencil kernel fusion after stencil unrolling");
                    hasUnrolledStencils = true;
                }
            });
            if (hasUnrolledStencils) {
                signalPassFailure();
                return;
            }
            (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
        }
    });
}

} // end of anonymous namespace

std::unique_ptr<Pass> mlir::createStencilKernelFusionPass() {
    return std::make_unique<StencilKernelFusionPass>();
}
