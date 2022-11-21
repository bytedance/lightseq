

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

  // for hard gates
  int *_p_d_hard_gates;
  int _batch_size;
  /**
    @param: h_hard_gates, the merge of three vector,each has the size of _max_batch_size: [hard_gates,gate_sizes,reorder_indexs]
    @shape: [_max_batch_size*3]
    for example:
        h_hard_gates: [15, 15, 13, 11, 11, 10, 9, 8, 1, 1, 1, 2, 1, 2, 0, 0, 7, 6, 5, 3, 4, 2, 0, 1]
        the merge of:
        hard_gates: [15, 15, 13, 11, 11, 10, 9, 8]
        gate_sizes: [1, 1, 1, 2, 1, 2, 0, 0]
        reorder_indexs: [7, 6, 5, 3, 4, 2, 0, 1]
  */
  std::vector<int> h_hard_gates;
  std::set<int> h_gate_sets;
  std::vector<int> h_lang_id;

  int get_output_seq_len();
  void init_hard_gates();
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
  void benchmark_mode(bool is_benchmark) override{};
};

LSMODEL_REGISTER(Moe);
}  // namespace cuda
}  // namespace lightseq
