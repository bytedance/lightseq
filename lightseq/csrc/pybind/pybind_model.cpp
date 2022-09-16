#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "model_base.h"
#include "util.h"

namespace py = pybind11;

class PyTransformer {
 private:
  lightseq::cuda::LSModel *model_;
  int *d_input_;
  std::vector<void *> d_outputs_;

 public:
  PyTransformer(std::string weight_path, int max_batch_size) {
    model_ = lightseq::cuda::LSModelFactory::GetInstance().CreateModel(
        "Transformer", weight_path, max_batch_size);
    std::vector<int> max_input_shape = model_->get_input_max_shape(0);
    int max_size =
        std::accumulate(max_input_shape.begin(), max_input_shape.end(), 1,
                        std::multiplies<int>());
    CHECK_GPU_ERROR(cudaMalloc(&d_input_, sizeof(int) * max_size));

    for (int i = 0; i < model_->get_output_size(); i++) {
      void *d_output;
      std::vector<int> shape = model_->get_output_max_shape(i);
      int output_size = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<int>());
      CHECK_GPU_ERROR(cudaMalloc(&d_output, output_size * sizeof(int)));
      model_->set_output_ptr(i, d_output);
      d_outputs_.push_back(d_output);
    }
  }
  ~PyTransformer() {
    delete model_;
    CHECK_GPU_ERROR(cudaFree(d_input_));
    for (auto d_output : d_outputs_) {
      CHECK_GPU_ERROR(cudaFree(d_output));
    }
  }

  std::tuple<py::array_t<int>, py::array_t<float>> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq) {
    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int *input_seq_data = input_seq_out.data(0, 0);
    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);

    CHECK_GPU_ERROR(cudaMemcpy(d_input_, input_seq_data,
                               sizeof(int) * input_seq_out.size(),
                               cudaMemcpyHostToDevice));

    model_->set_input_ptr(0, d_input_);
    model_->set_input_shape(0, {batch_size, batch_seq_len});

    model_->Infer();

    std::vector<int> output_shape = model_->get_output_shape(0);
    auto tokens = py::array_t<int>(output_shape);
    int *tokens_data = tokens.mutable_data(0, 0);
    const int *d_output = static_cast<const int *>(model_->get_output_ptr(0));
    CHECK_GPU_ERROR(cudaMemcpy(tokens_data, d_output,
                               sizeof(int) * tokens.size(),
                               cudaMemcpyDeviceToHost));

    std::vector<int> score_shape = model_->get_output_shape(1);
    auto scores = py::array_t<float>(score_shape);
    float *scores_data = scores.mutable_data(0, 0);
    const float *d_scores =
        static_cast<const float *>(model_->get_output_ptr(1));

    CHECK_GPU_ERROR(cudaMemcpy(scores_data, d_scores,
                               sizeof(float) * scores.size(),
                               cudaMemcpyDeviceToHost));
    return std::make_tuple(tokens, scores);
  }
};

class PyBert {
 private:
  lightseq::cuda::LSModel *model_;
  int *d_input_;
  std::vector<void *> d_outputs_;

 public:
  PyBert(std::string weight_path, int max_batch_size) {
    model_ = lightseq::cuda::LSModelFactory::GetInstance().CreateModel(
        "Bert", weight_path, max_batch_size);
    std::vector<int> max_input_shape = model_->get_input_max_shape(0);
    int max_size =
        std::accumulate(max_input_shape.begin(), max_input_shape.end(), 1,
                        std::multiplies<int>());
    CHECK_GPU_ERROR(cudaMalloc(&d_input_, sizeof(int) * max_size));

    for (int i = 0; i < model_->get_output_size(); i++) {
      void *d_output;
      std::vector<int> shape = model_->get_output_max_shape(i);
      int output_size = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<int>());
      CHECK_GPU_ERROR(cudaMalloc(&d_output, output_size * sizeof(int)));
      model_->set_output_ptr(i, d_output);
      d_outputs_.push_back(d_output);
    }
  }
  ~PyBert() {
    delete model_;
    CHECK_GPU_ERROR(cudaFree(d_input_));
    for (auto d_output : d_outputs_) {
      CHECK_GPU_ERROR(cudaFree(d_output));
    }
  }

  py::array_t<float> infer(
      py::array_t<int, py::array::c_style | py::array::forcecast> input_seq) {
    auto input_seq_out = input_seq.mutable_unchecked<2>();
    const int *input_seq_data = input_seq_out.data(0, 0);
    int batch_size = input_seq_out.shape(0);
    int batch_seq_len = input_seq_out.shape(1);

    CHECK_GPU_ERROR(cudaMemcpy(d_input_, input_seq_data,
                               sizeof(int) * input_seq_out.size(),
                               cudaMemcpyHostToDevice));

    model_->set_input_ptr(0, d_input_);
    model_->set_input_shape(0, {batch_size, batch_seq_len});

    model_->Infer();

    std::vector<int> output_shape = model_->get_output_shape(0);
    auto output = py::array_t<float>(output_shape);
    float *output_data = output.mutable_data(0, 0);
    lightseq::cuda::DataType output_type = model_->get_output_dtype(0);
    if (output_type == lightseq::cuda::kFloat32) {
      const float *d_output =
          static_cast<const float *>(model_->get_output_ptr(0));

      CHECK_GPU_ERROR(cudaMemcpy(output_data, d_output,
                                 sizeof(float) * output.size(),
                                 cudaMemcpyDeviceToHost));
    } else if (output_type == lightseq::cuda::kFloat16) {
      const half *d_output =
          static_cast<const half *>(model_->get_output_ptr(0));
      std::vector<half> h_bert_out(output.size());
      CHECK_GPU_ERROR(cudaMemcpy(h_bert_out.data(), d_output,
                                 sizeof(half) * output.size(),
                                 cudaMemcpyDeviceToHost));
      for (auto i = 0; i < h_bert_out.size(); i++) {
        float f_data = __half2float(h_bert_out[i]);
        output_data[i] = f_data;
      }
    } else {
      throw std::runtime_error("Not supported output type");
    }

    return output;
  }
};

PYBIND11_MODULE(inference, m) {
  m.attr("__name__") = "lightseq.inference";

  // py::class_<PyTransformer>(m, "Transformer")
  //     .def(py::init<const std::string, const int>(), py::arg("weight_path"),
  //          py::arg("max_batch_size"))
  //     .def("infer", &PyTransformer::infer,
  //          py::return_value_policy::reference_internal,
  //          py::arg("input_seq"));

  py::class_<PyBert>(m, "Bert")
      .def(py::init<const std::string, const int>(), py::arg("weight_path"),
           py::arg("max_batch_size"))
      .def("infer", &PyBert::infer, py::return_value_policy::reference_internal,
           py::arg("input_seq"));
}
