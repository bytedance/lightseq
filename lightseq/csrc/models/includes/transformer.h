#pragma once
#include "model_base.h"

#include "transformer_weight.h"

#include "launch_enc_emb_layer.h"
#include "launch_dec_emb_layer.h"
#include "transformer_encoder_layer.h"
#include "transformer_decoder_layer.h"
#include "lyr_normalize_layer.h"
#include "linear_layer.h"

#ifdef FP16_MODE
typedef __half OpType_;
#else
typedef float OpType_;
#endif

namespace lightseq {
namespace cuda {

class Transformer : public LSModel {
 private:
  TransformerWeight<OpType_> tw_;
  std::shared_ptr<Context> _context_ptr;

  LaunchEncEmbLayerPtr<OpType_> launch_enc_emb_layer;
  std::vector<TransformerEncoderLayerPtr<OpType_, OpType_>> enc_layer_vec;
  LyrNormalizeLayerPtr<OpType_, OpType_> enc_norm_layer;

  LaunchDecEmbLayerPtr<OpType_> launch_dec_emb_layer;
  std::vector<TransformerDecoderLayerPtr<OpType_, OpType_>> dec_layer_vec;
  LyrNormalizeLayerPtr<OpType_, OpType_> dec_norm_layer;
  LinearLayerPtr<OpType_, OpType_> linear_layer;

  ContextPtr context_ptr;

  Variable* inp_tokens;  // need to allocate
  Variable* token_emb;
  Variable* pos_emb;
  Variable* lang_emb;
  Variable* lang_id;

  std::vector<Variable*> cache_k_vec;
  std::vector<Variable*> new_k_vec;
  std::vector<Variable*> cache_v_vec;
  std::vector<Variable*> new_v_vec;

  int _max_batch_size;

 public:
  Transformer(const std::string weight_path, const int max_batch_size);
  ~Transformer();

  void before_forward(int batch_size, int seq_len, int cur_step);

  void Infer() override;
  void set_input_ptr(int index, void* input_ptr) override;
  void set_output_ptr(int index, void* output_ptr) override;
  const void* get_output_ptr(int index) override;
  std::vector<int> get_input_max_shape(int index) override;
  std::vector<int> get_output_max_shape(int index) override;
  DataType get_input_dtype(int index) override;
  DataType get_output_dtype(int index) override;
};

LSMODEL_REGISTER(Transformer);

}  // namespace cuda
}  // namespace lightseq
