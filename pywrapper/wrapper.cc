#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pywrapper/gpt.cc.cu"
#include "pywrapper/transformer.cc.cu"
#include "pywrapper/transformer_decoder.cc.cu"

namespace py = pybind11;

PYBIND11_MODULE(inference, m) {
  m.attr("__name__") = "lightseq.inference";
  py::class_<lightseq::cuda::TransformerDecoder>(m, "TransformerDecoder")
      .def(py::init<const std::string, const int>())
      .def("infer", &lightseq::cuda::TransformerDecoder::infer);

  py::class_<lightseq::cuda::Transformer>(m, "Transformer")
      .def(py::init<const std::string, const int>())
      .def("infer", &lightseq::cuda::Transformer::infer,
           py::return_value_policy::reference_internal, py::arg("input_seq"),
           py::arg("multiple_output") = false, py::arg("sampling_method") = "",
           py::arg("beam_size") = -1, py::arg("length_penalty") = -1.0f,
           py::arg("topp") = -1.0f, py::arg("topk") = -1.0f,
           py::arg("diverse_lambda") = -1.0f);

  py::class_<lightseq::cuda::Gpt>(m, "Gpt")
      .def(py::init<const std::string, const int, const int>(),
           py::arg("weight_path"), py::arg("max_batch_size"),
           py::arg("max_step"))
      .def("ppl", &lightseq::cuda::Gpt::ppl,
           py::return_value_policy::reference_internal, py::arg("input_seq"))
      .def("sample", &lightseq::cuda::Gpt::sample,
           py::return_value_policy::reference_internal, py::arg("input_seq"),
           py::arg("sampling_method") = "topk", py::arg("topk") = 1,
           py::arg("topp") = 0.75);
}
