#include "encdec_kv_layer.h"

namespace lightseq {

template <typename T1, typename T2>
EncDecKvLayer<T1, T2>::EncDecKvLayer(int nshared_layer, int max_batch_tokens,
                                     int hidden_size,
                                     int num_heads)
    : Layer("EncDecKvLayer"),  // necessary
      _nshared_layer(nshared_layer),
      _max_batch_tokens(max_batch_tokens),
      _hidden_size(hidden_size),
      _heads(num_heads),
      // operators
      _kv_linear(new LinearOp<T1, T2>(
          max_batch_tokens, nshared_layer * 2 * hidden_size, hidden_size)),
      _bias_add_transform_20314(new BiasAddTrans20314<T1, T2>(
          max_batch_tokens, num_heads, hidden_size, 2 * nshared_layer)) {
  // parameters
  _enc_kvw = new Variable("_enc_kvw", g_dtype<T1>(), g_dtype<T2>());
  _enc_kvb = new Variable("_enc_kvb", g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* EncDecKvLayer<T1, T2>::operator()(Variable* enc_out) {
  set_inputs({enc_out});

  Variable* kv_out = (*_kv_linear)(enc_out, _enc_kvw);

  Variable* transform_20314_out =
      (*_bias_add_transform_20314)(kv_out, _enc_kvb);

  set_outputs({transform_20314_out});

  return transform_20314_out;
}

template <typename T1, typename T2>
void EncDecKvLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
  _batch_tokens = batch_size * seq_len;

  _kv_linear->before_forward(_batch_tokens);

  _bias_add_transform_20314->before_forward(batch_size, seq_len);
}

template <typename T1, typename T2>
void EncDecKvLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
int EncDecKvLayer<T1, T2>::load_para_and_grad(const T1* para_ptr,
                                              T2* grad_ptr) {  // for training
  int offset = 0;
  _enc_kvw->set_value((char*)(para_ptr + offset));
  _enc_kvw->set_grad((char*)(grad_ptr + offset));
  offset += _nshared_layer * _hidden_size * _hidden_size * 2;

  _enc_kvb->set_value((char*)(para_ptr + offset));
  _enc_kvb->set_grad((char*)(grad_ptr + offset));
  offset += _nshared_layer * _hidden_size * 2;

  return offset;
}

template <typename T1, typename T2>
int EncDecKvLayer<T1, T2>::load_params(const std::vector<const T1*>& para_vec,
                                       int offset) {  // for inference
  int size = 0;
  _enc_kvw->set_value((char*)para_vec[offset + size]), size++;
  _enc_kvb->set_value((char*)para_vec[offset + size]), size++;
  return size;
}

}  // namespace lightseq
