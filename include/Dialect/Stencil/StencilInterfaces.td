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

def OffsetOp : OpInterface<"OffsetOp"> {
    let description = [{
        Interface to get the operation offset.
    }];

    let methods = [
        InterfaceMethod<"/* Get the offset of the operation */",
        "Index", "getOffset", (ins), [{
            Index result;
            ArrayAttr offset = $_op.offset();
            for (auto &elem : offset)
                result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
            return result;
        }]>
    ];
}

def ShiftOp : OpInterface<"ShiftOp"> {
  let description = [{
    Interface to shift the operation offset.
  }];

  let methods = [
    InterfaceMethod<"/*Shift operation by a constant offset*/",
      "void", "shiftByOffset", (ins "ArrayRef<int64_t>":$offset), [{}],
        /*defaultImplementation=*/[{
        SmallVector<Attribute, kIndexSize> offsetAttr;
        for (auto elem : llvm::zip(offset, $_op.offset())) {
          offsetAttr.push_back(IntegerAttr::get(
              IntegerType::get(64, $_op.getContext()),
              std::get<1>(elem).template cast<IntegerAttr>().getValue().getSExtValue() + 
              std::get<0>(elem)));
        }
        $_op.offsetAttr(ArrayAttr::get(offsetAttr, $_op.getContext()));
    }]>,
  ];
}

#endif // end of _DIALECT_STENCIL_STENCIL_INTERFACES_TD_