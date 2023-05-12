#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace lightseq {

enum DataType {
  kNotSupported = 0,
  kFloat32 = 1,
  kInt32 = 2,
  kInt64 = 3,
  kFloat16 = 4,
  kInt8 = 5,
  kInt16 = 6,
  kByte = 7,
  kUInt8 = 8,
  kUInt16 = 9,
  kUInt32 = 10,
  kUInt64 = 11,
  kFloat64 = 12
};

class GenerateConfig {
 public:
  bool _use_sampling;
  int _bos_id;
  int _eos_id;
  int _pad_id;
  int _top_k;
  int _max_new_tokens;
  float _temperature;
  float _top_p;
  // Note: After the model instance is initialized, the value of beam_size cannot be modified, 
  //       because it will participate in the model's GPU memory development calculation
  GenerateConfig() {}
  GenerateConfig(int bos_id, int eos_id, int pad_id, float temperature, bool use_sampling,
                 float top_p, int top_k, int max_new_tokens);
  void print_config();
};

#define SET_FLOAT_GEN_FIELD(field) \
void set_##field(float value) { _gen_conf->_##field = value; }

#define SET_INT_GEN_FIELD(field) \
void set_##field(int value) { _gen_conf->_##field = value; }

class LSModel {
 public:
  LSModel(std::vector<std::string> input_names,
          std::vector<std::string> output_names)
      : kInputNames(input_names), kOutputNames(output_names) {
    _gen_conf = new GenerateConfig();
    input_shapes_ = std::vector<std::vector<int>>(input_names.size());
    output_shapes_ = std::vector<std::vector<int>>(output_names.size());
  }

  virtual ~LSModel() {}
  virtual void Infer() = 0;

  // input getter and setter
  virtual void set_input_ptr(int index, void* input_ptr) = 0;
  void set_input_shape(int index, std::vector<int> shape) {
    input_shapes_.at(index) = std::move(shape);
  }
  virtual std::vector<int> get_input_max_shape(int index) = 0;
  std::string get_input_name(int index) { return kInputNames[index]; }
  int get_input_size() { return kInputNames.size(); }
  virtual DataType get_input_dtype(int index) = 0;

  // output getter and setter
  virtual void set_output_ptr(int index, void* output_ptr) = 0;
  virtual const void* get_output_ptr(int index) = 0;
  std::string get_output_name(int index) { return kOutputNames[index]; }
  int get_output_size() { return kOutputNames.size(); }
  std::vector<int> get_output_shape(int index) { return output_shapes_[index]; }
  virtual std::vector<int> get_output_max_shape(int index) = 0;
  virtual DataType get_output_dtype(int index) = 0;

  // void set_bos_id(int value) { _generate_config->bos_id = value; }
  SET_INT_GEN_FIELD(bos_id);
  SET_INT_GEN_FIELD(eos_id);
  SET_INT_GEN_FIELD(pad_id);
  SET_INT_GEN_FIELD(top_k);
  SET_INT_GEN_FIELD(max_new_tokens);
  SET_FLOAT_GEN_FIELD(temperature);
  SET_FLOAT_GEN_FIELD(top_p);

  virtual void benchmark_mode(bool is_benchmark) {}

 protected:
  void set_output_shape(int index, std::vector<int> shape) {
    output_shapes_.at(index) = std::move(shape);
  }
  const std::vector<std::string> kInputNames;
  const std::vector<std::string> kOutputNames;
  std::vector<std::vector<int>> input_shapes_;
  std::vector<std::vector<int>> output_shapes_;
  GenerateConfig* _gen_conf;
};

typedef LSModel* (*LSModelConstructor)(const std::string, const int);

class LSModelFactory {
 private:
  LSModelFactory() {}
  ~LSModelFactory() {}
  std::map<std::string, LSModelConstructor> object_map_;

 public:
  static LSModelFactory& GetInstance() {
    static LSModelFactory factory;
    return factory;
  }

  void ModelRegister(std::string class_name, LSModelConstructor obj);

  LSModel* CreateModel(std::string class_name, const std::string weight_path,
                       const int max_batch_size);
};

class Reflector {
 public:
  Reflector(std::string name, LSModelConstructor obj) {
    LSModelFactory::GetInstance().ModelRegister(name, obj);
  }
  virtual ~Reflector() {}
};

#define LSMODEL_REGISTER(className)                                 \
  LSModel* create_object_##className(const std::string weight_path, \
                                     const int max_batch_size) {    \
    return new className(weight_path, max_batch_size);              \
  }                                                                 \
  Reflector reflector_##className(#className, create_object_##className);

}  // namespace lightseq
#endif
