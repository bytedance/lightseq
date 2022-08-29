#pragma once
#include "layer_normalize.h"
#include "launch_enc_emb.h"
#include "layer.h"

namespace lightseq {

class LaunchEncEmbLayerWeight {
 public:
  LaunchEncEmbLayerWeight() {}
  char* _token_emb_ptr;
  char* _pos_emb_ptr;
  char* _lang_emb_ptr;
  char* _lang_id_ptr;

  template <typename T>
  int load_params(const std::vector<const T*>& para_vec, int offset) {
    _token_emb_ptr = (char*)para_vec[offset];
    _pos_emb_ptr = (char*)para_vec[offset + 1];
    _lang_emb_ptr = (char*)para_vec[offset + 4];
    _lang_id_ptr = (char*)nullptr;
    return 0;
  }
};

template int LaunchEncEmbLayerWeight::load_params(const std::vector<const float*>& para_vec, int offset);
template int LaunchEncEmbLayerWeight::load_params(const std::vector<const __half*>& para_vec, int offset);

using LaunchEncEmbLayerWeightPtr = std::shared_ptr<LaunchEncEmbLayerWeight>;

template <typename T>
class LaunchEncEmbLayer : public Layer {
 private:
  // operators
  LaunchEncEmbOp<T>* _launch_enc_op = nullptr;

  // parameters
  Variable* _token_emb;
  Variable* _pos_emb;
  Variable* _lang_emb;
  Variable* _lang_id;

 public:
  LaunchEncEmbLayer(LaunchEncEmbLayerWeightPtr enc_emb_wt, 
                  int max_batch_tokens, int pad_id, int hidden_dim, int multilg_type):
    Layer("LaunchEncEmbLayer"),
    _launch_enc_op(new LaunchEncEmbOp<T>(max_batch_tokens, pad_id, hidden_dim, multilg_type)) {
        
        _token_emb = new Variable("token_emb", enc_emb_wt->_token_emb_ptr);

        _pos_emb = new Variable("pos_emb", enc_emb_wt->_pos_emb_ptr);

        _lang_emb = new Variable("lang_emb", enc_emb_wt->_lang_emb_ptr);

        _lang_id = new Variable("lang_id", enc_emb_wt->_lang_id_ptr);

        this->_context_ptr->exit_layer();  // necessary
    }

  virtual ~LaunchEncEmbLayer() {}

  Variable* operator()(Variable* inp, Variable* pad_mask)  {
    this->set_inputs({inp, pad_mask});
    Variable* out = (*_launch_enc_op)(inp, _token_emb, _pos_emb, pad_mask, _lang_emb, _lang_id);
    this->set_outputs({out});
    return out;
  }

  void before_forward(int batch_size, int seq_len) { _launch_enc_op->before_forward(batch_size, seq_len); }

  void before_backward() {}
};

template class LaunchEncEmbLayer<__half>;
template class LaunchEncEmbLayer<float>;

template <class T>
using LaunchEncEmbLayerPtr = std::shared_ptr<LaunchEncEmbLayer<T>>;

}  // namespace lightseq
