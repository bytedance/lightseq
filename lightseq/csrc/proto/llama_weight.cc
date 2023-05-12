#include "llama_weight.h"

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
float LlamaWeight<float>::float2required(float value) {
  return value;
}

#ifdef LIGHTSEQ_cuda
/**
fp16 version, cast fp32 into fp16
*/
template <>
__half LlamaWeight<__half>::float2required(float value) {
  return __float2half_rn(value);
}
#endif

/**
Read model config stored in custom hdf5 file.
*/
template <typename T>
void LlamaWeight<T>::hdf5_get_model_config(hid_t hdf5_file) {
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/hidden_size", H5T_NATIVE_INT,
                           &_hidden_size);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/inner_size", H5T_NATIVE_INT,
                           &_inner_size);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/max_step", H5T_NATIVE_INT,
                           &_max_step);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/head_num", H5T_NATIVE_INT,
                           &_head_num);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/layer_num", H5T_NATIVE_INT,
                           &_layer_num);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/src_padding_id",
                           H5T_NATIVE_INT, &_padding_id);

  // special handling for string reading
  // string were converted to numpy array of np.int8 in python
  // hence needed to be read as an char array here
  char _generate_method_buf[128];  // get 128 character for sampling method
  int _generate_method_strlen = read_hdf5_dataset_data(
      hdf5_file, "model_conf/generate_method", H5T_NATIVE_CHAR,
      _generate_method_buf, [](int size) { return size > 128; },
      "Expect model_conf/generate_method to have less than 128 characters.");
  std::string _generate_method_read =
      std::string(_generate_method_buf, _generate_method_strlen);
  if (_generate_method_read != "") {
    _generate_method = _generate_method_read;
  }

  int _topk_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/topk", H5T_NATIVE_INT,
                           &_topk_read);
  if (_topk_read != 0) {
    _topk = _topk_read;
  }
  // _topk = 1;

  float _topp_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/topp", H5T_NATIVE_FLOAT,
                           &_topp_read);
  if (_topp_read != 0.0) {
    _topp = _topp_read;
  }

  int _eos_id_read;
  read_hdf5_dataset_scalar(hdf5_file, "model_conf/eos_id", H5T_NATIVE_INT,
                           &_eos_id_read);
  if (_eos_id_read != 0) {
    _eos_id = _eos_id_read;
  }

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/extra_decode_length",
                           H5T_NATIVE_INT, &_extra_decode_length);

  read_hdf5_dataset_scalar(hdf5_file, "model_conf/src_vocab_size",
                           H5T_NATIVE_INT, &_src_vocab_size);

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/beam_size", H5T_NATIVE_INT,
                             &_beam_size);
  } catch (HDF5DatasetNotFoundError& e) {
    _beam_size = 1;
  }

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/length_penalty",
                             H5T_NATIVE_FLOAT, &_length_penalty);
  } catch (HDF5DatasetNotFoundError& e) {
    _length_penalty = 1.0;
  }

  try {
    read_hdf5_dataset_scalar(hdf5_file, "model_conf/diverse_lambda",
                             H5T_NATIVE_FLOAT, &_diverse_lambda);
  } catch (HDF5DatasetNotFoundError& e) {
    _diverse_lambda = 0.;
  }

  _dim_per_head = _hidden_size / _head_num;
}

/**
Load the weights of embedding layer into GPU memory.
*/
template <typename T>
void LlamaWeight<T>::hdf5_parse_emb_wei(hid_t hdf5_file) {
  std::string dataset_prefix = "src_embedding";
  size_t value_size = _src_vocab_size * _hidden_size + _hidden_size;

  size_t max_value_size = _src_vocab_size * _hidden_size;

  std::vector<size_t> offset;
  std::vector<float> value(max_value_size);
  std::cout << "loading " << value_size / (1024 * 1024)
            << " M of decoder weight." << std::endl;

  const size_t max_buffer_size = max_value_size;
  float* source_buffer;
  T* target_buffer;
  cudaMalloc(&source_buffer, max_buffer_size * sizeof(float));
  cudaMalloc(&target_buffer, max_buffer_size * sizeof(T));
  T* addr = nullptr;

  size_t buffer_size;

  buffer_size = _src_vocab_size * _hidden_size;
  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/token_embedding", H5T_NATIVE_FLOAT,
      value.data(), [=](int size) { return size != buffer_size; },
      "Wrong token_embedding_size !");
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);

  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/post_norm_scale", H5T_NATIVE_FLOAT,
      value.data(), [=](int size) { return size != _hidden_size; },
      "Wrong norm_scale_size !");
  buffer_size = _hidden_size;
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);

  read_hdf5_dataset_data(
      hdf5_file, dataset_prefix + "/logits_linear_weight", H5T_NATIVE_FLOAT,
      value.data(),
      [=](int size) { return size != _src_vocab_size * _hidden_size; },
      "Wrong norm_scale_size !");
  buffer_size = _src_vocab_size * _hidden_size;
  addr = malloc_memory<T>(buffer_size);
  _p_d_src_emb_wei.push_back(addr);
  convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                          buffer_size, stream);

  std::cout << "finish initializing emb_wei from host to device" << std::endl;

  value.clear();
  value.shrink_to_fit();
  cudaFree(source_buffer);
  cudaFree(target_buffer);
}

/**
Load the weights of encoder into GPU memory.
*/
template <typename T>
void LlamaWeight<T>::hdf5_parse_enc_wei(hid_t hdf5_file) {
  size_t value_size =
      (_hidden_size + _hidden_size * _hidden_size * 3 +
       _hidden_size * _hidden_size + _hidden_size +
       _hidden_size * _inner_size * 2 + _hidden_size * _inner_size) *
      _layer_num;

  std::vector<size_t> value_size_vec = {_hidden_size,
                                        _hidden_size * _hidden_size * 3,
                                        _hidden_size * _hidden_size,
                                        _hidden_size,
                                        _hidden_size * _inner_size * 2,
                                        _hidden_size * _inner_size};
  size_t max_value_size =
      *max_element(value_size_vec.begin(), value_size_vec.end());

  std::vector<size_t> offset;
  std::vector<float> value(max_value_size);
  std::cout << "loading " << value_size / (1024 * 1024)
            << " M of decoder weight." << std::endl;

  const size_t max_buffer_size = max_value_size;
  float* source_buffer;
  T* target_buffer;
  cudaMalloc(&source_buffer, max_buffer_size * sizeof(float));
  cudaMalloc(&target_buffer, max_buffer_size * sizeof(T));

  T* addr = nullptr;
  size_t buffer_size;
  for (int layer_id = 0; layer_id < _layer_num; ++layer_id) {
    std::string dataset_prefix = "decoder_layers/" + std::to_string(layer_id);

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/attention_norm_scale", H5T_NATIVE_FLOAT,
        value.data(), [=](int size) { return size != _hidden_size; },
        "Wrong attention_norm_scale_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_enc_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/attention_project_qkv", H5T_NATIVE_FLOAT,
        value.data(),
        [=](int size) { return size != _hidden_size * _hidden_size * 3; },
        "Wrong attention_project_q_size !");
    buffer_size = _hidden_size * _hidden_size * 3;
    addr = malloc_memory<T>(buffer_size);
    _p_d_enc_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/attention_output", H5T_NATIVE_FLOAT,
        value.data(),
        [=](int size) { return size != _hidden_size * _hidden_size; },
        "Wrong attention_output_size !");
    buffer_size = _hidden_size * _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_enc_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/ffn_norm_scale", H5T_NATIVE_FLOAT,
        value.data(), [=](int size) { return size != _hidden_size; },
        "Wrong ffn_norm_scale_size !");
    buffer_size = _hidden_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_enc_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/gate_up_project_weight", H5T_NATIVE_FLOAT,
        value.data(),
        [=](int size) { return size != _hidden_size * _inner_size * 2; },
        "Wrong gate_up_project_weight_size !");
    buffer_size = _hidden_size * _inner_size * 2;
    addr = malloc_memory<T>(buffer_size);
    _p_d_enc_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);

    read_hdf5_dataset_data(
        hdf5_file, dataset_prefix + "/down_project_weight", H5T_NATIVE_FLOAT,
        value.data(),
        [=](int size) { return size != _hidden_size * _inner_size; },
        "Wrong down_project_weight_size !");
    buffer_size = _hidden_size * _inner_size;
    addr = malloc_memory<T>(buffer_size);
    _p_d_enc_wei.push_back(addr);
    convert_dtype_by_gpu<T>(value.data(), source_buffer, target_buffer, addr,
                            buffer_size, stream);
  }

  std::cout << "finish initializing dec_wei from host to device" << std::endl;

  value.clear();
  value.shrink_to_fit();
  cudaFree(source_buffer);
  cudaFree(target_buffer);
}

/**
Load the proto file into CPU memory and parse it.
*/
template <typename T>
std::string LlamaWeight<T>::initializing(std::string weight_path, GenerateConfig* gen_conf) {
  cudaStreamCreate(&stream);
  // If weight is of type pb, parse using proto parser.
  if (endswith(weight_path, ".hdf5")) {
    std::cout << "Parsing hdf5: " << weight_path << std::endl;

    hid_t hdf5_file = H5Fopen(weight_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (hdf5_file < 0) {
      return "Unable to read HDF5 file from " + weight_path;
    }
    hdf5_get_model_config(hdf5_file);

    // hdf5_parse_* would throw std::runtime_error on error
    hdf5_parse_emb_wei(hdf5_file);
    hdf5_parse_enc_wei(hdf5_file);
    H5Fclose(hdf5_file);

    cudaStreamSynchronize(stream);
    std::cout << "Finish loading all weight from host to device" << std::endl;
    return "";
  } else {
    return "Unsupported weight extention for [" + weight_path +
           "]; Supported extensions: .hdf5\n";
  }
  _gen_conf = gen_conf;
  *gen_conf = GenerateConfig(-1, _eos_id, _padding_id, 1., _generate_method != "beam_search", _topp, _topk, _extra_decode_length);
}
#ifdef LIGHTSEQ_cuda
template class LlamaWeight<__half>;
#endif
template class LlamaWeight<float>;

}  // namespace lightseq
