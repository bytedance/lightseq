#include "kernel_headers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace lightseq {
namespace x86 {

void test_simple_gemm(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& inpA,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& inpB,
    py::array_t<float, py::array::c_style | py::array::forcecast>& outC) {
  auto inpA_mutable = inpA.unchecked<2>();
  const float* inpA_ptr = inpA_mutable.data(0, 0);
  int _m = inpA_mutable.shape(0);
  int _k = inpA_mutable.shape(1);

  auto inpB_mutable = inpB.unchecked<2>();
  const float* inpB_ptr = inpB_mutable.data(0, 0);
  if (_k != inpB_mutable.shape(0)) {
    printf("Error! inpA.shape(1) not equal to inpB.shape(0)\n");
  }
  int _n = inpB_mutable.shape(1);

  auto outC_mutable = outC.mutable_unchecked<2>();
  float* outC_ptr = outC_mutable.mutable_data(0, 0);
  if (_m != outC_mutable.shape(0) && _n != outC_mutable.shape(1)) {
    printf("Error! outC.shape(0) should be (%d, %d) buf found (%d, %d)\n", _m,
           _n, outC_mutable.shape(0), outC_mutable.shape(1));
  }

  matrix_gemm(inpA_ptr, inpB_ptr, outC_ptr, _m, _n, _k);

  return;
}

void test_gemm_u8s8s32(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& inpA,
    const py::array_t<int8_t, py::array::c_style | py::array::forcecast>& inpB,
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>&
        C_compensation,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>& outC,
    bool trans_a, bool trans_b) {
  auto inpA_mutable = inpA.unchecked<2>();
  const uint8_t* inpA_ptr = inpA_mutable.data(0, 0);
  int _m = inpA_mutable.shape(trans_a ? 1 : 0);
  int _k = inpA_mutable.shape(trans_a ? 0 : 1);

  auto inpB_mutable = inpB.unchecked<2>();
  const int8_t* inpB_ptr = inpB_mutable.data(0, 0);
  int _n = inpB_mutable.shape(trans_b ? 0 : 1);
  if (_k != inpB_mutable.shape(trans_b ? 1 : 0)) {
    std::printf(
        "Error! inpB_mutable.shape() should be (%d, %d) buf found (%d, %d)\n",
        _n, _k, inpB_mutable.shape(0), inpB_mutable.shape(1));
    throw std::runtime_error("wrong shape of b");
  }

  auto outC_mutable = outC.mutable_unchecked<2>();
  int32_t* outC_ptr = outC_mutable.mutable_data(0, 0);
  if (_m != outC_mutable.shape(0) || _n != outC_mutable.shape(1)) {
    std::printf("Error! outC.shape() should be (%d, %d) buf found (%d, %d)\n",
                _m, _n, outC_mutable.shape(0), outC_mutable.shape(1));
  }
  auto C_compensation_mutable = C_compensation.unchecked<1>();
  const int32_t* C_compensation_ptr = C_compensation_mutable.data(0);
  if (_n != C_compensation_mutable.shape(0)) {
    std::printf(
        "Error! C_compensation.shape() should be (%d, ) buf found (%d, )\n", _n,
        C_compensation_mutable.shape(0));
    throw std::runtime_error("error shape");
  }

  const int64_t lda = trans_a ? _m : _k;
  const int64_t ldb = trans_b ? _k : _n;

  gemm(false, false, trans_a, trans_b, _m, _n, _k, 1, inpA_ptr, lda, inpB_ptr,
       ldb, 0, outC_ptr, _n, C_compensation_ptr);

  return;
}

}  // namespace x86
}  // namespace lightseq

#ifdef PYBIND_INTERFACE
#define PYBIND_MODULE_NAME TORCH_EXTENSION_NAME
#else
#define PYBIND_MODULE_NAME inference
#endif

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
  m.def("test_simple_gemm", &lightseq::x86::test_simple_gemm,
        "LightSeq test gemm with fp32 (x86 CPU)");
  m.def("test_gemm_u8s8s32", &lightseq::x86::test_gemm_u8s8s32,
        "LightSeq test gemm with int8 (x86 CPU)");
}
