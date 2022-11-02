#include "transformer.h"

namespace lightseq {
namespace cuda {

Transformer::Transformer(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"encoder_output"}),
      _max_batch_size(max_batch_size) {
  /* --- step.1 initial context --- */
  Context::create_global_context();
  _context_ptr = Context::global_instance();

  /* --- step.2 load model weights into GPU memory --- */
  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }
  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */
  inp_tokens = new Variable("inp_tokens");
  /* --- step.4 inital operator & layer --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;

  // initial LaunchEncEmb layer
  launch_enc_emb_layer.reset(new LaunchEncEmbLayer<OpType_>(
      max_batch_tokens, tw_._padding_id, tw_._hidden_size, tw_._multilg_type));
  launch_enc_emb_layer->load_params(tw_.get_src_emb_wei(), 0);

  // initial TransformerEncoder layers
  float attn_prob_dropout_ratio = 0.0;
  float activation_dropout_ratio = 0.0;
  float hidden_dropout_ratio = 0.0;
  int enc_wei_offset = 0;
  for (int idx = 0; idx < tw_._n_enc_layer; idx++) {
    TransformerEncoderLayerPtr<OpType_, OpType_> enc_layer_(
        new TransformerEncoderLayer<OpType_, OpType_>(
            idx, max_batch_tokens, tw_._max_step, tw_._hidden_size,
            tw_._head_num, tw_._inner_size, attn_prob_dropout_ratio,
            activation_dropout_ratio, hidden_dropout_ratio, true,
            tw_._use_gelu ? "gelu" : "relu", false, tw_._is_post_ln));
    enc_wei_offset +=
        enc_layer_->load_params(tw_.get_enc_wei(), enc_wei_offset);
    enc_layer_vec.push_back(enc_layer_);
  }

  // initial LayerNormalize layer
  lyr_norm_layer.reset(new LyrNormalizeLayer<OpType_, OpType_>(
      max_batch_tokens, tw_._hidden_size));
  lyr_norm_layer->load_params(tw_.get_src_emb_wei(), 2);

  // initial 


  /* --- step.5 construct network --- */
  // std::tuple<Variable*, Variable*> enc_emb_outs = (*launch_enc_emb_layer)(inp_tokens);
  // Variable *enc_emb = std::get<0>(enc_emb_outs);
  // Variable *pad_mask = std::get<1>(enc_emb_outs);
  // for (auto iter : enc_layer_vec) {
  //   enc_emb = (*iter)(enc_emb, pad_mask);
  // }
  // bert_out = (*lyr_norm_layer)(enc_emb);
}

Transformer::~Transformer() {  }

void Transformer::before_forward(int batch_size, int seq_len) {
  // launch_enc_emb_layer->before_forward(batch_size, seq_len);

  // for (auto iter : enc_layer_vec) {
  //   iter->before_forward(batch_size, seq_len);
  // }

  // lyr_norm_layer->before_forward(batch_size * seq_len);
}

void Transformer::Infer() {
  int batch_size = input_shapes_[0][0], seq_len = input_shapes_[0][1];

  before_forward(batch_size, seq_len);

  /* --- notice that the order of forward should be the same with network --- */
  launch_enc_emb_layer->forward();
  for (auto iter : enc_layer_vec) {
    iter->forward();
  }
  lyr_norm_layer->forward();

  CHECK_GPU_ERROR(cudaStreamSynchronize(_context_ptr->get_stream()));

  set_output_shape(0, {batch_size, seq_len, tw_._hidden_size});
}

void Transformer::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      inp_tokens->set_value((char *)input_ptr);
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Transformer::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      bert_out->set_value((char *)output_ptr);
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *Transformer::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(bert_out->value());
    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Transformer::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
std::vector<int> Transformer::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step, tw_._hidden_size};

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Transformer::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Transformer::get_output_dtype(int index) {
  switch (index) {
    case 0:
#ifdef FP16_MODE
      return DataType::kFloat16;
#else
      return DataType::kFloat32;
#endif

      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

}  // namespace cuda
}  // namespace lightseq
