#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pywrapper/transformer.cc.cu"
#include "pywrapper/transformer_decoder.cc.cu"

namespace py = pybind11;

PYBIND11_MODULE(lightseq, m) {
  py::class_<byseqlib::cuda::TransformerDecoder>(m, "TransformerDecoder")
      .def(py::init<const std::string, const int>())
      .def("infer", &byseqlib::cuda::TransformerDecoder::infer);

  py::class_<byseqlib::cuda::Transformer>(m, "Transformer")
      .def(py::init<const std::string, const int>())
      .def("infer", &byseqlib::cuda::Transformer::infer);
}
