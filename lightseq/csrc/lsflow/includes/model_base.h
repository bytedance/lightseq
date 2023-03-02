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

// Bellow is an usage example for lightseq cpp API
//
// auto model = lightseq::LSModelFactory::GetInstance().CreateModel(
//     "Transformer", model_weights_path, max_batch_size);
// model->set_input_ptr(0, d_input);
// model->set_input_shape(0, {batch_size, batch_seq_len});

// for (int i = 0; i < model->get_output_size(); i++) {
//   void* d_output;
//   std::vector<int> shape = model->get_output_max_shape(i);
//   int total_size = 1;
//   for (int j = 0; j < shape.size(); j++) {
//     total_size *= shape[j];
//   }
//   lightseq::CHECK_GPU_ERROR(
//       cudaMalloc(&d_output, total_size * sizeof(int)));
//   model->set_output_ptr(i, d_output);
// }

// for (int i = 0; i < model->get_output_size(); i++) {
//   const float* d_output;
//   d_output = static_cast<const float*>(model->get_output_ptr(i));
//   std::vector<int> shape = model->get_output_shape(i);
//   std::cout << "output shape: ";
//   for (int j = 0; j < shape.size(); j++) {
//     std::cout << shape[j] << " ";
//   }
//   std::cout << std::endl;

//   lightseq::cuda::print_vec(d_output, "output", 5);
// }
class LSModel {
 public:
  LSModel(std::vector<std::string> input_names,
          std::vector<std::string> output_names)
      : kInputNames(input_names), kOutputNames(output_names) {
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

 protected:
  void set_output_shape(int index, std::vector<int> shape) {
    output_shapes_.at(index) = std::move(shape);
  }
  const std::vector<std::string> kInputNames;
  const std::vector<std::string> kOutputNames;
  std::vector<std::vector<int>> input_shapes_;
  std::vector<std::vector<int>> output_shapes_;
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

  void ModelRegister(std::string class_name, LSModelConstructor obj) {
    if (obj) {
      object_map_.insert(std::map<std::string, LSModelConstructor>::value_type(
          class_name, obj));
    }
  }

  LSModel* CreateModel(std::string class_name, const std::string weight_path,
                       const int max_batch_size) {
    std::map<std::string, LSModelConstructor>::const_iterator iter =
        object_map_.find(class_name);
    if (iter != object_map_.end()) {
      return iter->second(weight_path, max_batch_size);
    } else {
      throw std::runtime_error("Model not supported");
    }
  }
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
