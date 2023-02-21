import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)
sys.path.insert(0, par_dir)
csrc_dir = os.path.dirname(par_dir)
sys.path.insert(0, csrc_dir)


import torch
import numpy as np

from pytorch.builder.x86_kernel_builder import X86KernelBuilder
from util import TestDecorator

x86_kernel_module = X86KernelBuilder().load()

kt = TestDecorator()


@kt.case(dtypes=[torch.float])
def test_gemm_case():
    hidden_size = kt.hidden_dim
    batch, seq_len = kt.bs_sl()
    bs = batch * seq_len
    inpA = kt.rand([bs, hidden_size])
    inpB = kt.rand([hidden_size, hidden_size])
    outC = kt.zeros([bs, hidden_size]).clone().detach()

    cus_inpA = inpA.clone().detach().cpu().numpy()
    cus_inpB = inpB.clone().detach().cpu().numpy()
    print(f"torch threads: {torch.get_num_threads()}")

    def custom():
        x86_kernel_module.test_simple_gemm(cus_inpA, cus_inpB, outC)
        return [outC.clone().detach()]

    base_inpA = inpA.clone()
    base_inpB = inpB.clone()

    def baseline():
        base_out = torch.mm(base_inpA, base_inpB)
        return [base_out]

    return custom, baseline


@kt.case(dtypes=[torch.int8])
def test_gemm_u8s8s32():
    # TODO: may fail when cpu not support vnni
    # https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html#processors-with-the-intel-avx2-or-intel-avx-512-support
    hidden_size = kt.hidden_dim
    batch, seq_len = kt.bs_sl()
    bs = batch * seq_len
    inpA = kt.randint8([bs, hidden_size])
    inpB = kt.randint8([hidden_size, hidden_size])
    outC = kt.zeros([bs, hidden_size]).to(dtype=torch.int32)

    cus_inpA = inpA.to(torch.int32).add(128).to(torch.uint8).contiguous().numpy()
    cus_inpB = inpB.clone().cpu().contiguous().numpy()
    C_compensation = (
        inpB.T.to(torch.int32)
        .sum(dim=0)
        .mul(-128)
        .contiguous()
        .clone()
        .detach()
        .numpy()
    )
    cus_outC = outC.clone().detach().cpu().contiguous().numpy()

    def custom():
        x86_kernel_module.test_gemm_u8s8s32(
            cus_inpA, cus_inpB, C_compensation, cus_outC, False, True
        )
        res = np.array(cus_outC)
        return [res]

    base_inpA = inpA.clone().detach()
    base_inpB = inpB.clone().detach()

    def baseline():
        base_out = torch.nn.functional.linear(base_inpA.float(), base_inpB.float())
        return [base_out.to(torch.int32)]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cpu", nhead=16)
    kernel_list = ["test_gemm_case", "test_gemm_u8s8s32"]
    kt.run(kernel_list)
