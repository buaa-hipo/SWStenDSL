#ifndef _DIALECT_STENCIL_STENCIL_INTERFACES_TD_
#define _DIALECT_STENCIL_STENCIL_INTERFACES_TD_

include "mlir/IR/OpBase.td"

//============================================================================//
// 本部分定义stencil op中共用的接口
//============================================================================//

// 域的边界相关
def ShapeOp : OpInterface<"ShapeOp"> {
    let description = [{
        Interfaces to get the operation bounds.
    }];

    let methods = [
        InterfaceMethod<"/* Get the lower bound of the operation */",
            "Index", "getLB", (ins), [{
                Index result;
                ArrayAttr lb = $_op.lb();
                for (auto &elem : lb.getValue())
                    result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
                return result;
            }]>,
        InterfaceMethod<"/* Get the upper bound of the operation */",
            "Index", "getUB", (ins), [{
                Index result;
                ArrayAttr ub = $_op.ub();
                for (auto &elem : ub.getValue())
                    result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
                return result;
            }]>,
        InterfaceMethod<"/* Set the lower bound of the operation */",
            "void", "setLB", (ins "ArrayRef<int64_t>":$lb), [{
                SmallVector<Attribute, kIndexSize> lbAttr;
                size_t lbSize = lb.size();
                for (size_t i = 0; i < lbSize; i++) {
                    lbAttr.push_back(IntegerAttr::get(IntegerType::get(64, $_op.getContext()), lb[i]));
                }
                $_op.lbAttr(ArrayAttr::get(lbAttr, $_op.getContext()));
            }]>,
        InterfaceMethod<"/* Set the upper bound of the operation */",
            "void", "setUB", (ins "ArrayRef<int64_t>":$ub), [{
                SmallVector<Attribute, kIndexSize> ubAttr;
                size_t upSize = ub.size();
                for (size_t i = 0; i < upSize; i++) {
                    ubAttr.push_back(IntegerAttr::get(IntegerType::get(64, $_op.getContext()), ub[i]));
                }
                $_op.ubAttr(ArrayAttr::get(ubAttr, $_op.getContext()));
            }]>,
        InterfaceMethod<"/* Get the rank of the operation */",
            "int64_t", "getRank", (ins), [{
                ArrayAttr lb = $_op.lb();
                ArrayAttr ub = $_op.ub();
                assert(lb.getValue().size() == ub.getValue().size() &&
                        "expected lower and upper bound to have the same rank");
                return lb.getValue().size();
            }]>
    ];
}

#endif // end of _DIALECT_STENCIL_STENCIL_INTERFACES_TD_