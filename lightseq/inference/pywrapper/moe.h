

#include "model_base.h"
#include "../model/moe_decoder.h"
#include "../model/moe_encoder.h"
#include "../proto/moe_weight.h"
#include "../tools/util.h"

#ifdef FP16_MODE
const lightseq::cuda::OperationType moe_optytpe =
    lightseq::cuda::OperationType::FP16;
#else
const lightseq::cuda::OperationType moe_optytpe =
    lightseq::cuda::OperationType::FP32;
#endif

namespace lightseq {
namespace cuda {
class Moe : public LSModel {
 private:
  typedef OperationTypeTraits<moe_optytpe> optraits;
  std::shared_ptr<MoeEncoder<moe_optytpe>> encoder_;
  std::shared_ptr<MoeDecoder<moe_optytpe>> decoder_;

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
  MoeWeight<moe_optytpe> tw_;

  int get_output_seq_len();

  const int *get_result_ptr();
  const float *get_score_ptr();
  const int get_max_step() { return tw_._max_step; }
  const int get_beam_size() { return tw_._beam_size; }

 public:
  Moe(const std::string weight_path, const int max_batch_size);
  ~Moe();

  void Infer() override;
  void set_input_ptr(int index, void *input_ptr) override;
  void set_output_ptr(int index, void *output_ptr) override;
  const void *get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
};

LSMODEL_REGISTER(Moe);
}  // namespace cuda
}  // namespace lightseq
