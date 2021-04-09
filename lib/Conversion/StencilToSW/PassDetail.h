/**
 * @file PassDetail.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 自定义pass生成文件包装
 * @version 0.1
 * @date 2021-03-22
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _CONVERSION_STENCILTOSW_PASSDETAIL_H_
#define _CONVERSION_STENCILTOSW_PASSDETAIL_H_

#include <mlir/Pass/Pass.h>

namespace mlir {

#define GEN_PASS_CLASSES
#include "Conversion/StencilToSW/Passes.h.inc"

} // end of namespace mlir

#endif // _CONVERSION_STENCILTOSW_PASSDETAIL_H_