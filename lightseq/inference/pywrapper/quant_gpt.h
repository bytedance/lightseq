
#include "model_base.h"
#include "../model/quant_gpt_encoder.h"
#include "../proto/quant_gpt_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType gpt_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType gpt_optype =
    lightseq::cuda::OperationType::FP32;
#endif

namespace lightseq {
namespace cuda {
class QuantGpt : public LSModel {
 private:
  typedef lightseq::cuda::OperationTypeTraits<gpt_optype> optraits;
  std::shared_ptr<lightseq::cuda::QuantGptEncoder<gpt_optype>> encoder_;

  int* d_input_;
  int* d_sample_id;
  float* d_ppl;

  int _max_batch_size;
  cudaStream_t stream_;
  cudaStream_t cache_stream_;
  cublasHandle_t hd_;
  lightseq::cuda::QuantGptWeight<gpt_optype> tw_;
  std::set<std::string> available_sampling_methods = {"topk", "topp"};

 public:
  QuantGpt(const std::string weight_path, const int max_batch_size);

  ~QuantGpt();

  const int* get_result_ptr();
  const float* get_score_ptr();
  const int get_max_step() { return tw_._max_step; }

  void Infer() override;
  void set_input_ptr(int index, void* input_ptr) override;
  void set_output_ptr(int index, void* output_ptr) override;
  const void* get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
};

LSMODEL_REGISTER(QuantGpt);

}  // namespace cuda
}  // namespace lightseq
