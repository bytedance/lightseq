#pragma once
#include "model_base.h"
#include "bert_weight.h"
#include "layer.h"
#include "launch_enc_emb_layer.h"
#include "transformer_encoder_layer.h"
#include "launch_enc_emb.h"
#include "lyr_normalize_layer.h"

#ifdef FP16_MODE
typedef __half OpType_;
#else
typedef float OpType_;
#endif

namespace lightseq {
namespace cuda {

class Bert : public LSModel {
 private:
  BertWeight<OpType_> tw_;

  LaunchEncEmbLayerPtr<OpType_> launch_enc_emb_layer;
  std::vector<TransformerEncoderLayerPtr<OpType_, OpType_> > enc_layer_vec;
  LyrNormalizeLayerPtr<OpType_, OpType_> lyr_norm_layer;

  ContextPtr context_ptr;

  Variable* inp_tokens;  // need to allocate
  Variable* token_emb;
  Variable* pos_emb;
  Variable* pad_mask;  // need to allocate
  Variable* lang_emb;
  Variable* lang_id;

  Variable* bert_out;

  int _max_batch_size;

  int* pad_mask_ptr;

 public:
  Bert(const std::string weight_path, const int max_batch_size);
  ~Bert();

  void before_forward(int batch_size, int seq_len);

  void Infer() override;
  void set_input_ptr(int index, void* input_ptr) override;
  void set_output_ptr(int index, void* output_ptr) override;
  const void* get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
};

LSMODEL_REGISTER(Bert);

}  // namespace cuda
}  // namespace lightseq
