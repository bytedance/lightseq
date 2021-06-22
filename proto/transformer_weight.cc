#include "transformer_weight.h"

#include <fstream>
/**
@file
Load the model weights which stored in custom proto file into GPU memory.
Currently, fp16 and fp32 versions are provided.
Weights in proto file will always be in fp32. For fp16, the weights
  will be casted from fp32 into fp16
*/

namespace lightseq {
namespace cuda {
/**
Cast weights into required datatype.
The datatype of weights in custom proto file will always be in fp32.
*/
template <>
float TransformerWeight<OperationType::FP32>::float2required(float value) {
  return value;
}

/**
fp16 version, cast fp32 into fp16
*/
template <>
__half TransformerWeight<OperationType::FP16>::float2required(float value) {
  return __float2half_rn(value);
}

/**
Read model config stored in custom proto file.
*/
template <OperationType OpType_>
void TransformerWeight<OpType_>::get_model_config(
    const Transformer &transformer, bool only_decoder) {
  _hidden_size = transformer.trg_embedding().norm_scale_size();
  _inner_size =
      transformer.decoder_stack()[0].ffn_first_kernel_size() / _hidden_size;
  _max_step =
      transformer.trg_embedding().position_embedding_size() / _hidden_size;
  if (!only_decoder) {
    _src_vocab_size =
        transformer.src_embedding().token_embedding_size() / _hidden_size;
  }
  _trg_vocab_size =
      transformer.trg_embedding().token_embedding_size() / _hidden_size;
  if (!only_decoder) {
    _n_enc_layer = transformer.encoder_stack_size();
  }
  _n_dec_layer = transformer.decoder_stack_size();
  _head_num = transformer.model_conf().head_num();
  _dim_per_head = _hidden_size / _head_num;
  _weight_per_enc_layer = 12;
  _weight_per_dec_layer = 18;
  _beam_size = transformer.model_conf().beam_size();
  _extra_decode_length = transformer.model_conf().extra_decode_length();
  _length_penalty = transformer.model_conf().length_penalty();
  _padding_id = transformer.model_conf().src_padding_id();
  _start_id = transformer.model_conf().trg_start_id();
  _end_id = transformer.model_conf().trg_end_id();
  if (_end_id == 0) {
    _end_id = _trg_vocab_size - 1;
  }
  _diverse_lambda = transformer.model_conf().diverse_lambda();
  _sampling_method = transformer.model_conf().sampling_method();
  if (_sampling_method == "") {
    _sampling_method = "beam_search";
  }
  _topk = transformer.model_conf().topk();
  _topp = transformer.model_conf().topp();
  _is_post_ln = transformer.model_conf().is_post_ln();
  _no_scale_embedding = transformer.model_conf().no_scale_embedding();
  _use_gelu = transformer.model_conf().use_gelu();
  _is_multilingual = transformer.model_conf().is_multilingual();
}

/**
Load the weights of embedding layer into GPU memory.
Compared with the encoder, the decoder has more
  encoder output project weights, encoder output project bias,
  logits bias. So we need an "source" parameter to
  distinguish between encoder and decoder
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::parse_emb_wei(
    const EmbeddingLayer &layer, std::string source) {
  int vocab_size = (source == "src") ? _src_vocab_size : _trg_vocab_size;

  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  offset.push_back(idx);
  if (layer.token_embedding_size() != vocab_size * _hidden_size)
    return "Wrong token_embedding_size !";
  for (float ele : layer.token_embedding()) value.push_back(ele);
  idx += vocab_size * _hidden_size;

  offset.push_back(idx);
  if (layer.position_embedding_size() != _max_step * _hidden_size)
    return "Wrong position_embedding_size !";
  for (float ele : layer.position_embedding()) value.push_back(ele);
  idx += _max_step * _hidden_size;

  offset.push_back(idx);
  if (layer.norm_scale_size() != _hidden_size) return "Wrong norm_scale_size !";
  for (float ele : layer.norm_scale()) value.push_back(ele);
  idx += _hidden_size;

  offset.push_back(idx);
  if (layer.norm_bias_size() != _hidden_size) return "Wrong norm_bias_size !";
  for (float ele : layer.norm_bias()) value.push_back(ele);
  idx += _hidden_size;

  if (source == "src") {
    std::vector<_DataType> raw_value;
    for (float e : value) raw_value.push_back(float2required(e));
    _d_src_emb_wei = raw_value;
    for (int e : offset)
      _p_d_src_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_src_emb_wei.data()) + e);
  } else {
    // for trg, encdec_kv_kernel, encdec_kv_bias, logit_bias

    offset.push_back(idx);
    if (layer.encode_output_project_kernel_kv_size() !=
        _hidden_size * _hidden_size * 2 * _n_dec_layer)
      return "Wrong encode_output_project_kernel_kv_size !";
    for (float ele : layer.encode_output_project_kernel_kv())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size * 2 * _n_dec_layer;

    offset.push_back(idx);
    if (layer.encode_output_project_bias_kv_size() !=
        _hidden_size * 2 * _n_dec_layer)
      return "Wrong encode_output_project_bias_kv_size !";
    for (float ele : layer.encode_output_project_bias_kv())
      value.push_back(ele);
    idx += _hidden_size * 2 * _n_dec_layer;

    offset.push_back(idx);
    if (layer.shared_bias_size() != vocab_size)
      return "Wrong shared_bias_size !";
    for (float ele : layer.shared_bias()) value.push_back(ele);
    idx += vocab_size;

    std::vector<_DataType> raw_value;
    for (float e : value) raw_value.push_back(float2required(e));
    _d_trg_emb_wei = raw_value;
    for (int e : offset) {
      _p_d_trg_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_trg_emb_wei.data()) + e);
    }
  }  // trg

  if (_is_multilingual) {
    // fill in language embedding
    std::vector<_DataType> raw_value;
    for (float e : layer.lang_emb()) {
      raw_value.push_back(float2required(e));
    }

    if (source == "src") {
      _d_src_lang_emb = raw_value;
      _p_d_src_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_src_lang_emb.data()));
    } else {
      if (layer.lang_emb_size() / _hidden_size !=
          layer.trg_vocab_mask_size() / _trg_vocab_size) {
        return "Wrong trg_lang_emb_size or trg_vocab_mask_size !";
      }
      _d_trg_lang_emb = raw_value;
      _p_d_trg_emb_wei.push_back(
          thrust::raw_pointer_cast(_d_trg_lang_emb.data()));
      // fill in target vocab mask
      std::vector<int> h_mask;
      for (int ele : layer.trg_vocab_mask()) h_mask.push_back(ele);
      _d_trg_vocab_mask = h_mask;
      _p_d_trg_vocab_mask = thrust::raw_pointer_cast(_d_trg_vocab_mask.data());
    }

    std::cout << "Finish loading multi lingual weights from host to device"
              << std::endl;
  }

  std::cout << "Finish loading " << source << "_emb_wei from host to device"
            << std::endl;
  return "";
}

/**
Load the weights of encoder into GPU memory.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::parse_enc_wei(
    const Transformer &transformer) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  for (auto enc_layer : transformer.encoder_stack()) {
    offset.push_back(idx);
    if (enc_layer.multihead_norm_scale_size() != _hidden_size)
      return "Wrong multihead_norm_scale_size !";
    for (float ele : enc_layer.multihead_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_norm_bias_size() != _hidden_size)
      return "Wrong multihead_norm_bias_size !";
    for (float ele : enc_layer.multihead_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_qkv_size() !=
        _hidden_size * _hidden_size * 3)
      return "Wrong multihead_project_kernel_qkv_size !";
    for (float ele : enc_layer.multihead_project_kernel_qkv())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_qkv_size() != _hidden_size * 3)
      return "Wrong multihead_project_bias_qkv_size !";
    for (float ele : enc_layer.multihead_project_bias_qkv())
      value.push_back(ele);
    idx += _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "Wrong multihead_project_kernel_output_size !";
    for (float ele : enc_layer.multihead_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_output_size() != _hidden_size)
      return "Wrong multihead_project_bias_output_size !";
    for (float ele : enc_layer.multihead_project_bias_output())
      value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_scale_size() != _hidden_size)
      return "Wrong ffn_norm_scale_size !";
    for (float ele : enc_layer.ffn_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_bias_size() != _hidden_size)
      return "Wrong ffn_norm_bias_size !";
    for (float ele : enc_layer.ffn_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_first_kernel_size !";
    for (float ele : enc_layer.ffn_first_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_bias_size() != _inner_size)
      return "Wrong ffn_first_bias_size !";
    for (float ele : enc_layer.ffn_first_bias()) value.push_back(ele);
    idx += _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_second_kernel_size !";
    for (float ele : enc_layer.ffn_second_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_bias_size() != _hidden_size)
      return "Wrong ffn_second_bias_size !";
    for (float ele : enc_layer.ffn_second_bias()) value.push_back(ele);
    idx += _hidden_size;

  }  // for

  std::vector<_DataType> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_enc_wei = raw_value;

  for (int e : offset)
    _p_d_enc_wei.push_back(thrust::raw_pointer_cast(_d_enc_wei.data()) + e);
  std::cout << "Finish loading enc_wei from host to device" << std::endl;
  return "";
}

/**
Load the weights of decoder into GPU memory.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::parse_dec_wei(
    const Transformer &transformer) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  for (auto dec_layer : transformer.decoder_stack()) {
    offset.push_back(idx);
    if (dec_layer.self_norm_scale_size() != _hidden_size)
      return "Wrong self_norm_scale size !";
    for (float ele : dec_layer.self_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.self_norm_bias_size() != _hidden_size)
      return "Wrong self_norm_bias_size !";
    for (float ele : dec_layer.self_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.self_project_kernel_qkv_size() !=
        _hidden_size * _hidden_size * 3)
      return "Wrong self_project_kernel_qkv size !";
    for (float ele : dec_layer.self_project_kernel_qkv()) value.push_back(ele);
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    if (dec_layer.self_project_bias_qkv_size() != _hidden_size * 3)
      return "Wrong self_project_bias_qkv size !";
    for (float ele : dec_layer.self_project_bias_qkv()) value.push_back(ele);
    idx += _hidden_size * 3;

    offset.push_back(idx);
    if (dec_layer.self_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "Wrong self_project_kernel_output size !";
    for (float ele : dec_layer.self_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (dec_layer.self_project_bias_output_size() != _hidden_size)
      return "Wrong self_project_bias_output size !";
    for (float ele : dec_layer.self_project_bias_output()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_norm_scale_size() != _hidden_size)
      return "Wrong encdec_norm_scale size !";
    for (float ele : dec_layer.encdec_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_norm_bias_size() != _hidden_size)
      return "Wrong encdec_norm_bias_size !";
    for (float ele : dec_layer.encdec_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_kernel_q_size() != _hidden_size * _hidden_size)
      return "Wrong encdec_project_kernel_q size !";
    for (float ele : dec_layer.encdec_project_kernel_q()) value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_bias_q_size() != _hidden_size)
      return "Wrong encdec_project_bias_q size !";
    for (float ele : dec_layer.encdec_project_bias_q()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_kernel_output_size() !=
        _hidden_size * _hidden_size)
      return "Wrong encdec_project_kernel_output size !";
    for (float ele : dec_layer.encdec_project_kernel_output())
      value.push_back(ele);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (dec_layer.encdec_project_bias_output_size() != _hidden_size)
      return "Wrong encdec_project_bias_output size !";
    for (float ele : dec_layer.encdec_project_bias_output())
      value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.ffn_norm_scale_size() != _hidden_size)
      return "Wrong ffn_norm_scale_size !";
    for (float ele : dec_layer.ffn_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.ffn_norm_bias_size() != _hidden_size)
      return "Wrong ffn_norm_bias_size !";
    for (float ele : dec_layer.ffn_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (dec_layer.ffn_first_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_first_kernel_size !";
    for (float ele : dec_layer.ffn_first_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (dec_layer.ffn_first_bias_size() != _inner_size)
      return "Wrong ffn_first_bias_size !";
    for (float ele : dec_layer.ffn_first_bias()) value.push_back(ele);
    idx += _inner_size;

    offset.push_back(idx);
    if (dec_layer.ffn_second_kernel_size() != _hidden_size * _inner_size)
      return "Wrong ffn_second_kernel_size !";
    for (float ele : dec_layer.ffn_second_kernel()) value.push_back(ele);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (dec_layer.ffn_second_bias_size() != _hidden_size)
      return "Wrong ffn_second_bias_size !";
    for (float ele : dec_layer.ffn_second_bias()) value.push_back(ele);
    idx += _hidden_size;

  }  // for

  std::vector<_DataType> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_dec_wei = raw_value;

  for (int e : offset)
    _p_d_dec_wei.push_back(thrust::raw_pointer_cast(_d_dec_wei.data()) + e);
  std::cout << "Finish loading dec_wei from host to device" << std::endl;
  return "";
}

/**
Load the proto file into CPU memory and parse it.
*/
template <OperationType OpType_>
std::string TransformerWeight<OpType_>::initializing(std::string proto_path,
                                                     bool only_decoder) {
  Transformer transformer;
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::fstream raw_input(proto_path, std::ios::in | std::ios::binary);
  if (!transformer.ParseFromIstream(&raw_input)) {
    return "Parse weights from [" + proto_path + "] failed.";
  }

  get_model_config(transformer, only_decoder);

  if (_hidden_size % 4 != 0) {
    return "hidden_size should be a multiple of 4 to avoid misaligned address "
           "in CUDA";
  }

  std::string res;
  if (!only_decoder) {
    res = parse_emb_wei(transformer.src_embedding(), "src");
    if (!res.empty()) return res;
  }

  res = parse_emb_wei(transformer.trg_embedding(), "trg");
  if (!res.empty()) return res;

  if (!only_decoder) {
    res = parse_enc_wei(transformer);
    if (!res.empty()) return res;
  }

  res = parse_dec_wei(transformer);
  if (!res.empty()) return res;

  std::cout << "Finish loading all weight from host to device" << std::endl;
  // Optional:  Delete all global objects allocated by libprotobuf.
  // google::protobuf::ShutdownProtobufLibrary();
  return "";
}

template class TransformerWeight<OperationType::FP16>;
template class TransformerWeight<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
