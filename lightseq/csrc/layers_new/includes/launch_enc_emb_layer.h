#pragma once
#include "layer_normalize.h"
#include "launch_enc_emb.h"
#include "layer.h"

namespace lightseq {

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
  LaunchEncEmbLayer(int max_batch_tokens, int pad_id, int hidden_dim,
                    int multilg_type)
      : Layer("LaunchEncEmbLayer"),
        _launch_enc_op(new LaunchEncEmbOp<T>(max_batch_tokens, pad_id,
                                             hidden_dim, multilg_type)) {
    _token_emb = new Variable("token_emb");
    _pos_emb = new Variable("pos_emb");
    _lang_emb = new Variable("lang_emb");
    _lang_id = new Variable("lang_id");

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~LaunchEncEmbLayer() {}

  std::tuple<Variable*, Variable*> operator()(Variable* inp) {
    set_inputs({inp});

    std::tuple<Variable*, Variable*> out =
        (*_launch_enc_op)(inp, _token_emb, _pos_emb, _lang_emb, _lang_id);

    set_outputs({std::get<0>(out), std::get<1>(out)});
    return out;
  }

  void before_forward(int batch_size, int seq_len) {
    _launch_enc_op->before_forward(batch_size, seq_len);
  }

  void before_backward() {}

  int load_params(const std::vector<const T*>& para_vec, int offset) {
    _token_emb->set_value((char*)para_vec[offset]);
    _pos_emb->set_value((char*)para_vec[offset + 1]);
    // _lang_emb->set_value((char*)para_vec[offset + 4]);
    return 0;
  }
};

template class LaunchEncEmbLayer<float>;
#ifdef LIGHTSEQ_cuda
template class LaunchEncEmbLayer<__half>;
#endif

template <class T>
using LaunchEncEmbLayerPtr = std::shared_ptr<LaunchEncEmbLayer<T>>;

}  // namespace lightseq
