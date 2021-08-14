/**
 * @file PassDetail.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief stencil 方言优化生成文件包装
 * @version 0.1
 * @date 2021-08-14
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */
#ifndef _CONVERSION_STENCIL_PASSDETAIL_H_
#define _CONVERSION_STENCIL_PASSDETAIL_H_

#include <mlir/Pass/Pass.h>

namespace mlir {

#define GEN_PASS_CLASSES
#include "Dialect/Stencil/Passes.h.inc"

} // end of namespace mlir

#endif // _CONVERSION_STENCIL_PASSDETAIL_H_