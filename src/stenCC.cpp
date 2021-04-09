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

#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/SW/SWDialect.h"
#include "Conversion/StencilToSW/Passes.h"

using namespace mlir;

int main(int argc, char *argv[])
{
    registerAllDialects();
    registerAllPasses();
    
    // register the stencil passes
    registerStencilConversionPasses();

    mlir::DialectRegistry registry;
    registry.insert<stencil::StencilDialect>();
    registry.insert<sw::SWDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<scf::SCFDialect>();

    return failed(mlir::MlirOptMain(argc, argv, "StencilDSL Compiler\n", registry));
}