#pragma once
#include "launch_gpt_emb.h"
#include "layer.h"

namespace lightseq {

template <typename T>
class LaunchGptEmbLayer : public Layer {
 private:
  // operators
  LaunchGptEmbOp<T>* _launch_gpt_op = nullptr;

  // parameters
  Variable* _token_emb;
  Variable* _pos_emb;

 public:
  LaunchGptEmbLayer(int max_batch_tokens, int max_step, int beam_size,
                    int pad_id, int hidden_dim)
      : Layer("LaunchGptEmbLayer"),
        _launch_gpt_op(new LaunchGptEmbOp<T>(max_batch_tokens, max_step,
                                             beam_size, pad_id, hidden_dim)) {
    _token_emb = new Variable("token_emb", g_dtype<T>());
    _pos_emb = new Variable("pos_emb", g_dtype<T>());

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~LaunchGptEmbLayer() {}

  std::tuple<Variable*, Variable*> operator()(Variable* inp) {
    set_inputs({inp});

    std::tuple<Variable*, Variable*> out =
        (*_launch_gpt_op)(inp, _token_emb, _pos_emb);

    set_outputs({std::get<0>(out), std::get<1>(out)});
    return out;
  }

  void before_forward(int batch_size, int seq_len, int offset) {
    _launch_gpt_op->before_forward(batch_size, seq_len, offset);
  }

  void before_backward() {}

  int load_params(const std::vector<const T*>& para_vec, int offset) {
    _token_emb->set_value((char*)para_vec[offset]);
    // _token_emb->set_shape({});
    _pos_emb->set_value((char*)para_vec[offset + 1]);
    // _pos_emb->set_shape();
    return 0;
  }
};

template class LaunchGptEmbLayer<float>;
#ifdef LIGHTSEQ_cuda
template class LaunchGptEmbLayer<__half>;
#endif

template <class T>
using LaunchGptEmbLayerPtr = std::shared_ptr<LaunchGptEmbLayer<T>>;

}  // namespace lightseq
