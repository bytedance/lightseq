#include "context.h"
#include "kernel_headers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace lightseq {
namespace x86 {

py::array_t<float> test_gemm(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& inpA,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& inpB) {
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

  std::vector<int> outC_shape{_m, _n};
  auto outC = py::array_t<float>(outC_shape);
  float* output_data = outC.mutable_data(0, 0);

  int ret = matrix_gemm(inpA_ptr, inpB_ptr, output_data, _m, _n, _k);

  return outC;
}

}  // namespace x86
}  // namespace lightseq

PYBIND11_MODULE(inference, m) {
  // m.attr("__name__") = "lightseq.inference";

  m.def("test_gemm", &lightseq::x86::test_gemm,
        "LightSeq test gemm with fp32 (x86 CPU)");
}
