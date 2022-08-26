#include "model_base.h"
#include "../model/t5_decoder.h"
#include "../model/t5_encoder.h"
#include "../proto/t5_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType t5_optype =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType t5_optype =
    lightseq::cuda::OperationType::FP32;
#endif

namespace lightseq {
namespace cuda {
class T5 : public LSModel {
 private:
  typedef OperationTypeTraits<t5_optype> optraits;
  std::shared_ptr<T5Encoder<t5_optype>> encoder_;
  std::shared_ptr<T5Decoder<t5_optype>> decoder_;

  optraits::DataType *d_encoder_output_;
  int *d_input_;
  int *d_src_lang_id_;
  int *d_trg_lang_id_;
  int *d_output_;
  int *d_padding_mask_;
  void *d_buf_;
  int _max_batch_size;
  cudaStream_t stream_;
  cublasHandle_t hd_;
  T5Weight<t5_optype> tw_;

  int get_output_seq_len();

  const int *get_result_ptr();
  const float *get_score_ptr();
  int get_max_step() { return tw_._max_step; }
  int get_beam_size() { return tw_._beam_size; }

 public:
  T5(const std::string weight_path, const int max_batch_size);
  ~T5();

  void Infer() override;
  void set_input_ptr(int index, void *input_ptr) override;
  void set_output_ptr(int index, void *output_ptr) override;
  const void *get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
  void benchmark_mode(bool is_benchmark) override{};
};

LSMODEL_REGISTER(T5);
}  // namespace cuda
}  // namespace lightseq
