import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)
sys.path.insert(0, par_dir)
csrc_dir = os.path.dirname(par_dir)
sys.path.insert(0, csrc_dir)


import torch
from pytorch.builder.x86_kernel_builder import X86KernelBuilder
from util import TestDecorator

x86_kernel_module = X86KernelBuilder().load()

kt = TestDecorator()


@kt.case(dtypes=[torch.float])
def test_gemm_case():
    inpA = kt.rand([128, 128])
    inpB = kt.rand([128, 128])

    cus_inpA = inpA.clone().cpu().numpy()
    cus_inpB = inpB.clone().cpu().numpy()

    def custom():
        cus_out = x86_kernel_module.test_gemm(cus_inpA, cus_inpB)
        return [cus_out]

    base_inpA = inpA.clone()
    base_inpB = inpB.clone()

    def baseline():
        base_out = torch.mm(base_inpA, base_inpB)
        return [base_out]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cpu", nhead=16)
    kernel_list = ["test_gemm_case"]
    kt.run(kernel_list)
