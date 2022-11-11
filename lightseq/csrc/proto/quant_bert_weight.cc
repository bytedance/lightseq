#include "quant_bert_weight.h"

#include <fstream>

/**
@file
Load the model weights which stored in custom proto file into GPU memory.
Currently, fp16 and fp32 versions are provided.
Weights in proto file will always be in fp32. For fp16, the weights
  will be casted from fp32 into fp16
*/

namespace lightseq {

/**
Cast weights into required datatype.
The datatype of weights in custom proto file will always be in fp32.
*/
template <>
float QuantBertWeight<float>::float2required(float value) {
  return value;
}

/**
fp16 version, cast fp32 into fp16
*/
template <>
__half QuantBertWeight<__half>::float2required(float value) {
  return __float2half_rn(value);
}

/**
Read model config stored in custom proto file.
*/
template <typename T>
void QuantBertWeight<T>::proto_get_model_config(const QuantBert &bert) {
  _hidden_size = bert.src_embedding().norm_scale_size();
  _inner_size =
      bert.encoder_stack()[0].ffn_first_kernel().size() / _hidden_size;
  _max_step = bert.src_embedding().position_embedding_size() / _hidden_size;
  _src_vocab_size =
      bert.src_embedding().token_embedding().size() / _hidden_size;
  _n_enc_layer = bert.encoder_stack_size();
  _head_num = bert.model_conf().head_num();
  _dim_per_head = _hidden_size / _head_num;
  _weight_per_enc_layer = 12;
  _padding_id = bert.model_conf().src_padding_id();
  _is_post_ln = bert.model_conf().is_post_ln();
  _use_gelu = bert.model_conf().use_gelu();
  _multilg_type = bert.model_conf().multilg_type();
}

/**
Load the weights of embedding layer into GPU memory.
*/
template <typename T>
std::string QuantBertWeight<T>::proto_parse_emb_wei(
    const QuantBertEmbeddingLayer &layer) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  offset.push_back(idx);
  if (layer.token_embedding().size() != _src_vocab_size * _hidden_size)
    return "wrong token_embedding_size !";
  for (unsigned char ele : layer.token_embedding())
    value.push_back(dequantize(ele, _quant_range, layer.emb_clip_max()));
  idx += _src_vocab_size * _hidden_size;
  _src_emb_clip_max = layer.emb_clip_max();

  offset.push_back(idx);
  if (layer.position_embedding_size() != _max_step * _hidden_size)
    return "wrong position_embedding_size !";
  for (float ele : layer.position_embedding()) value.push_back(ele);
  idx += _max_step * _hidden_size;

  offset.push_back(idx);
  if (layer.norm_scale_size() != _hidden_size) return "wrong norm_scale_size !";
  for (float ele : layer.norm_scale()) value.push_back(ele);
  idx += _hidden_size;

  offset.push_back(idx);
  if (layer.norm_bias_size() != _hidden_size) return "wrong norm_bias_size !";
  for (float ele : layer.norm_bias()) value.push_back(ele);
  idx += _hidden_size;

  std::vector<T> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_src_emb_wei = raw_value;
  for (int e : offset)
    for (int e : offset) _p_d_src_emb_wei.push_back(_d_src_emb_wei.data() + e);

  std::cout << "finish initializing emb_wei from host to device" << std::endl;
  return "";
}

/**
Load the weights of encoder into GPU memory.
*/
template <typename T>
std::string QuantBertWeight<T>::proto_parse_enc_wei(const QuantBert &bert) {
  std::vector<int> offset;
  std::vector<float> value;
  int idx = 0;

  int max_size =
      std::max(_hidden_size * _hidden_size * 3, _hidden_size * _inner_size);
  std::vector<float> temp_buffer(max_size);

  for (auto enc_layer : bert.encoder_stack()) {
    offset.push_back(idx);
    if (enc_layer.multihead_norm_scale_size() != _hidden_size)
      return "wrong multihead_norm_scale_size !";
    for (float ele : enc_layer.multihead_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_norm_bias_size() != _hidden_size)
      return "wrong multihead_norm_bias_size !";
    for (float ele : enc_layer.multihead_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_qkv().size() !=
        _hidden_size * _hidden_size * 3)
      return "wrong multihead_project_kernel_qkv_size !";
    for (unsigned char ele : enc_layer.multihead_project_kernel_qkv())
      value.push_back(
          dequantize(ele, _quant_range,
                     enc_layer.multihead_project_kernel_qkv_clip_max()));
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_qkv_size() != _hidden_size * 3)
      return "wrong multihead_project_bias_qkv_size !";
    for (float ele : enc_layer.multihead_project_bias_qkv())
      value.push_back(ele);
    idx += _hidden_size * 3;

    offset.push_back(idx);
    if (enc_layer.multihead_project_kernel_output().size() !=
        _hidden_size * _hidden_size)
      return "wrong multihead_project_kernel_output_size !";
    for (unsigned char ele : enc_layer.multihead_project_kernel_output())
      value.push_back(
          dequantize(ele, _quant_range,
                     enc_layer.multihead_project_kernel_output_clip_max()));
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    if (enc_layer.multihead_project_bias_output_size() != _hidden_size)
      return "wrong multihead_project_bias_output_size !";
    for (float ele : enc_layer.multihead_project_bias_output())
      value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_scale_size() != _hidden_size)
      return "wrong ffn_norm_scale_size !";
    for (float ele : enc_layer.ffn_norm_scale()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_norm_bias_size() != _hidden_size)
      return "wrong ffn_norm_bias_size !";
    for (float ele : enc_layer.ffn_norm_bias()) value.push_back(ele);
    idx += _hidden_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_kernel().size() != _hidden_size * _inner_size)
      return "wrong ffn_first_kernel_size !";
    for (unsigned char ele : enc_layer.ffn_first_kernel())
      value.push_back(
          dequantize(ele, _quant_range, enc_layer.ffn_first_kernel_clip_max()));
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_first_bias_size() != _inner_size)
      return "wrong ffn_first_bias_size !";
    for (float ele : enc_layer.ffn_first_bias()) value.push_back(ele);
    idx += _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_kernel().size() != _hidden_size * _inner_size)
      return "wrong ffn_second_kernel_size !";
    for (unsigned char ele : enc_layer.ffn_second_kernel())
      value.push_back(dequantize(ele, _quant_range,
                                 enc_layer.ffn_second_kernel_clip_max()));
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    if (enc_layer.ffn_second_bias_size() != _hidden_size)
      return "wrong ffn_second_bias_size !";
    for (float ele : enc_layer.ffn_second_bias()) value.push_back(ele);
    idx += _hidden_size;

    _enc_clip_max.push_back(enc_layer.multihead_project_kernel_qkv_clip_max());
    _enc_clip_max.push_back(
        enc_layer.multihead_project_kernel_output_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_first_kernel_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_second_kernel_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_ln_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_project_output_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_ln_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_first_act_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_qkv_dense_clip_max());
    _enc_clip_max.push_back(enc_layer.multihead_output_dense_clip_max());
    _enc_clip_max.push_back(enc_layer.ffn_first_output_clip_max());

  }  // for

  temp_buffer.clear();

  std::vector<T> raw_value;
  for (float e : value) raw_value.push_back(float2required(e));
  _d_enc_wei = raw_value;

  for (int e : offset) _p_d_enc_wei.push_back(_d_enc_wei.data() + e);
  std::cout << "finish initializing enc_wei from host to device" << std::endl;
  return "";
}

/**
Read model config stored in custom hdf5 file.
*/
template <typename T>
void QuantBertWeight<T>::hdf5_get_model_config(hid_t hdf5_file) {
  _hidden_size = get_hdf5_dataset_size(hdf5_file, "src_embedding/norm_scale");

  _inner_size =
      get_hdf5_dataset_size(hdf5_file, "encoder_stack/0/ffn_first_kernel") /
      _hidden_size;

  _max_step =
      get_hdf5_dataset_size(hdf5_file, "src_embedding/position_embedding") /
      _hidden_size;

  _src_vocab_size =
      get_hdf5_dataset_size(hdf5_file, "src_embedding/token_embedding") /
      _hidden_size;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/n_encoder_stack",
                           H5T_NATIVE_INT, &_n_enc_layer);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/head_num", H5T_NATIVE_INT,
                           &_head_num);

  _dim_per_head = _hidden_size / _head_num;
  _weight_per_enc_layer = 12;

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/src_padding_id",
                           H5T_NATIVE_INT, &_padding_id);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/is_post_ln", H5T_NATIVE_HBOOL,
                           &_is_post_ln);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/use_gelu", H5T_NATIVE_HBOOL,
                           &_use_gelu);

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/multilg_type",
                             H5T_NATIVE_INT, &_multilg_type);
  } catch (HDF5DatasetNotFoundError &e) {
    // default value
    _multilg_type = 0;
  }
}

/**
Load the weights of embedding layer into GPU memory.
*/
template <typename T>
void QuantBertWeight<T>::hdf5_parse_emb_wei(hid_t hdf5_file) {
  std::string dataset_prefix = "src_embedding";

  size_t value_size = _src_vocab_size * _hidden_size +
                      _max_step * _hidden_size + 2 * _hidden_size;

  std::vector<int> offset;
  std::vector<float> value(value_size);  // preallocate vector for performance
  std::vector<unsigned char> value_i8(value_size);
  std::cout << "loading " << value_size * sizeof(T) / (1024 * 1024)
            << " MB of embedding weight." << std::endl;
  int idx = 0;
  float clip_max;

  offset.push_back(idx);
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/token_embedding", H5T_NATIVE_UCHAR,
      value_i8.data() + idx,
      [=](int size) { return size != _src_vocab_size * _hidden_size; },
      "Wrong token_embedding_size !");
  read_hdf5_dataset_scalar(hdf5_file, dataset_prefix + "/emb_clip_max",
                           H5T_NATIVE_FLOAT, &clip_max);
  dequantize_array(value_i8, value, clip_max, _quant_range, idx,
                   _src_vocab_size * _hidden_size);
  _src_emb_clip_max = clip_max;
  idx += _src_vocab_size * _hidden_size;

  offset.push_back(idx);
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/position_embedding", H5T_NATIVE_FLOAT,
      value.data() + idx,
      [=](int size) { return size != _max_step * _hidden_size; },
      "Wrong position_embedding_size !");
  idx += _max_step * _hidden_size;

  offset.push_back(idx);
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/norm_scale", H5T_NATIVE_FLOAT,
      value.data() + idx, [=](int size) { return size != _hidden_size; },
      "Wrong norm_scale_size !");
  idx += _hidden_size;

  offset.push_back(idx);
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/norm_bias", H5T_NATIVE_FLOAT,
      value.data() + idx, [=](int size) { return size != _hidden_size; },
      "Wrong norm_bias_size !");
  idx += _hidden_size;

  std::vector<T> raw_value;
  raw_value.reserve(value.size());
  for (float e : value) raw_value.push_back(float2required(e));
  _d_src_emb_wei = raw_value;
  for (int e : offset) _p_d_src_emb_wei.push_back(_d_src_emb_wei.data() + e);

  std::cout << "Finish loading src_emb_wei from host to device" << std::endl;
}

/**
Load the weights of encoder into GPU memory.
*/
template <typename T>
void QuantBertWeight<T>::hdf5_parse_enc_wei(hid_t hdf5_file) {
  size_t value_size =
      (_hidden_size * 2 + _hidden_size * _hidden_size * 3 + _hidden_size * 3 +
       _hidden_size * _hidden_size + _hidden_size * 3 +
       _hidden_size * _inner_size + _inner_size + _hidden_size * _inner_size +
       _hidden_size) *
      _n_enc_layer;
  std::vector<int> offset;
  std::vector<float> value(value_size);
  std::vector<unsigned char> value_i8(value_size);
  std::cout << "loading " << value_size * sizeof(T) / (1024 * 1024)
            << " MB of encoder weight." << std::endl;

  int max_size =
      std::max(3 * _hidden_size * _hidden_size, _hidden_size * _inner_size);
  std::vector<float> temp_buffer(max_size);

  float clip_max;
  int idx = 0;
  for (int layer_id = 0; layer_id < _n_enc_layer; ++layer_id) {
    std::string dataset_prefix = "encoder_stack/" + std::to_string(layer_id);

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_norm_scale", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong multihead_norm_scale_size !");
    idx += _hidden_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_norm_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong multihead_norm_bias_size !");
    idx += _hidden_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_kernel_qkv",
        H5T_NATIVE_UCHAR, value_i8.data() + idx,
        [=](int size) { return size != _hidden_size * _hidden_size * 3; },
        "Wrong multihead_project_kernel_qkv_size !");
    read_hdf5_dataset_scalar(
        hdf5_file, dataset_prefix + "/multihead_project_kernel_qkv_clip_max",
        H5T_NATIVE_FLOAT, &clip_max);
    dequantize_array(value_i8, value, clip_max, _quant_range, idx,
                     _hidden_size * _hidden_size * 3);
    _enc_clip_max.push_back(clip_max);
    transform_param_shape(value.data() + idx, temp_buffer.data(), _hidden_size,
                          3 * _hidden_size);
    idx += _hidden_size * _hidden_size * 3;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_bias_qkv",
        H5T_NATIVE_FLOAT, value.data() + idx,
        [=](int size) { return size != _hidden_size * 3; },
        "Wrong multihead_project_bias_qkv_size !");
    idx += _hidden_size * 3;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_kernel_output",
        H5T_NATIVE_UCHAR, value_i8.data() + idx,
        [=](int size) { return size != _hidden_size * _hidden_size; },
        "Wrong multihead_project_kernel_output_size !");
    read_hdf5_dataset_scalar(
        hdf5_file, dataset_prefix + "/multihead_project_kernel_output_clip_max",
        H5T_NATIVE_FLOAT, &clip_max);
    dequantize_array(value_i8, value, clip_max, _quant_range, idx,
                     _hidden_size * _hidden_size);
    _enc_clip_max.push_back(clip_max);
    transform_param_shape(value.data() + idx, temp_buffer.data(), _hidden_size,
                          _hidden_size);
    idx += _hidden_size * _hidden_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/multihead_project_bias_output",
        H5T_NATIVE_FLOAT, value.data() + idx,
        [=](int size) { return size != _hidden_size; },
        "Wrong multihead_project_bias_output_size !");
    idx += _hidden_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_norm_scale", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong ffn_norm_scale_size !");
    idx += _hidden_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_norm_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong ffn_norm_bias_size !");
    idx += _hidden_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_first_kernel", H5T_NATIVE_UCHAR,
        value_i8.data() + idx,
        [=](int size) { return size != _hidden_size * _inner_size; },
        "Wrong ffn_first_kernel_size !");
    read_hdf5_dataset_scalar(hdf5_file,
                             dataset_prefix + "/ffn_first_kernel_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    dequantize_array(value_i8, value, clip_max, _quant_range, idx,
                     _hidden_size * _inner_size);
    _enc_clip_max.push_back(clip_max);
    transform_param_shape(value.data() + idx, temp_buffer.data(), _hidden_size,
                          _inner_size);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_first_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _inner_size; },
        "Wrong ffn_first_bias_size !");
    idx += _inner_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_second_kernel", H5T_NATIVE_UCHAR,
        value_i8.data() + idx,
        [=](int size) { return size != _hidden_size * _inner_size; },
        "Wrong ffn_second_kernel_size !");
    read_hdf5_dataset_scalar(hdf5_file,
                             dataset_prefix + "/ffn_second_kernel_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    dequantize_array(value_i8, value, clip_max, _quant_range, idx,
                     _hidden_size * _inner_size);
    _enc_clip_max.push_back(clip_max);
    transform_param_shape(value.data() + idx, temp_buffer.data(), _inner_size,
                          _hidden_size);
    idx += _hidden_size * _inner_size;

    offset.push_back(idx);
    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_second_bias", H5T_NATIVE_FLOAT,
        value.data() + idx, [=](int size) { return size != _hidden_size; },
        "Wrong ffn_second_bias_size !");
    idx += _hidden_size;

    read_hdf5_dataset_scalar(hdf5_file,
                             dataset_prefix + "/multihead_ln_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
    read_hdf5_dataset_scalar(
        hdf5_file, dataset_prefix + "/multihead_project_output_clip_max",
        H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
    read_hdf5_dataset_scalar(hdf5_file, dataset_prefix + "/ffn_ln_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
    read_hdf5_dataset_scalar(hdf5_file,
                             dataset_prefix + "/ffn_first_act_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
    read_hdf5_dataset_scalar(hdf5_file,
                             dataset_prefix + "/multihead_qkv_dense_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
    read_hdf5_dataset_scalar(
        hdf5_file, dataset_prefix + "/multihead_output_dense_clip_max",
        H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
    read_hdf5_dataset_scalar(hdf5_file,
                             dataset_prefix + "/ffn_first_output_clip_max",
                             H5T_NATIVE_FLOAT, &clip_max);
    _enc_clip_max.push_back(clip_max);
  }

  temp_buffer.clear();

  std::vector<T> raw_value;
  raw_value.reserve(value.size());
  for (float e : value) raw_value.push_back(float2required(e));
  _d_enc_wei = raw_value;

  for (int e : offset)
    for (int e : offset) _p_d_enc_wei.push_back(_d_enc_wei.data() + e);
  std::cout << "Finish loading enc_wei from host to device" << std::endl;
}

/**
Load the proto file into CPU memory and parse it.
*/
template <typename T>
std::string QuantBertWeight<T>::initializing(std::string weight_path) {
  if (endswith(weight_path, ".pb")) {
    std::cout << "Parsing protobuf: " << weight_path << std::endl;
    QuantBert bert;
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::fstream raw_input(weight_path, std::ios::in | std::ios::binary);
    if (!bert.ParseFromIstream(&raw_input)) {
      return "Parse weights from [" + weight_path + "] failed.";
    }

    proto_get_model_config(bert);
    if (_hidden_size % 4 != 0) {
      return "hidden_size should be a multiple of 4 to avoid misaligned "
             "address in CUDA";
    }

    std::string res = proto_parse_emb_wei(bert.src_embedding());
    if (!res.empty()) return res;

    res = proto_parse_enc_wei(bert);
    if (!res.empty()) return res;

    std::cout << "finish initializing all weight from host to device"
              << std::endl;
    // Optional:  Delete all global objects allocated by libprotobuf.
    // google::protobuf::ShutdownProtobufLibrary();
    return "";
  } else if (endswith(weight_path, ".hdf5")) {
    std::cout << "Parsing hdf5: " << weight_path << std::endl;

    hid_t hdf5_file = H5Fopen(weight_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (hdf5_file < 0) {
      return "Unable to read HDF5 file from " + weight_path;
    }
    hdf5_get_model_config(hdf5_file);
    if (_hidden_size % 4 != 0) {
      return "hidden_size should be a multiple of 4 to avoid misaligned "
             "address in CUDA";
    }
    // hdf5_parse_* would throw std::runtime_error on error
    hdf5_parse_emb_wei(hdf5_file);
    hdf5_parse_enc_wei(hdf5_file);
    H5Fclose(hdf5_file);

    std::cout << "Finish loading all weight from host to device" << std::endl;
    return "";
  } else {
    return "Unsupported weight extention for [" + weight_path +
           "]; Supported extensions: .pb, .hdf5\n";
  }
}

template class QuantBertWeight<__half>;
template class QuantBertWeight<float>;

}  // namespace lightseq
