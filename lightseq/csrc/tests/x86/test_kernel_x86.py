import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)
sys.path.insert(0, par_dir)

import torch
import lightseq.kernel as lsk
from util import TestDecorator

kt = TestDecorator()


@kt.case(dtypes=[torch.float])
def test_gemm_case():
    inpA = kt.rand([128, 128])
    inpB = kt.rand([128, 128])

    def custom():
        cus_inpA = inpA.clone().cpu().numpy()
        cus_inpB = inpB.clone().cpu().numpy()
        cus_out = lsk.test_gemm(cus_inpA, cus_inpB)
        return [torch.from_numpy(cus_out).contiguous()]

    def baseline():
        base_inpA = inpA.clone()
        base_inpB = inpB.clone()
        base_out = torch.mm(base_inpA, base_inpB)
        return [base_out.contiguous().cpu()]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kernel_list = ["test_gemm_case"]
    kt.run(kernel_list)
