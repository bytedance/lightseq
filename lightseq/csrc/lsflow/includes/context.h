/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/
#pragma once
#include "cstdio"
#include "queue"
#include "deque"
#include "stack"
#include "unordered_map"

#include "declaration.h"
#include "manager.h"
#include "layer.h"
#include "allocator.h"

namespace lightseq {

/*
  - Class:  Context
  - Description:
      Context is an object that manages model information, and each model
  instance corresponds to a context object. The context class mainly needs to
  play a role in the following scenarios:
        1. Record the hierarchical relationship between layer and node.
        2. Holds some global information about the model. eg. cudaStream,
  cublasHandle..
        3. Hold some global resources and retrieve them by unique key.
        4. *Obtain the life cycle and max shape information of tensors in the
  model, which will be used by MemoryManager for shared video memory resource
  planning.

      At the same time, a context object also corresponds to a MemoryManager
  object and an Allocator object, which manages the memory development and
  allocation of the entire model.
*/
class Context {
 private:
  // just for pybind interface.
  static std::unordered_map<std::string, std::shared_ptr<void>> pybind_layers;
  std::unordered_map<std::string, void*> _resources_pool;

  std::vector<Node*> _all_node_vec{};
  std::vector<Operator*> _model_ops{};
  std::vector<Layer*> _root_layers{};
  std::vector<Layer*> _all_layers{};
  std::deque<Layer*> _layer_context;
  StatusType _status_type;

  bool _built = false;
  bool _building = false;
  int _node_idx = 0;
  MemoryManagerPtr _mm_ptr;
  AllocatorPtr _allocator_ptr;

  int _device_id;

  static std::shared_ptr<Context> _global_context_ptr;

  bool check_validate();

  static std::unordered_map<int, std::shared_ptr<Context>> global_contexts_map;
  static int global_context_id;

  int _regress_begin_idx = -1;
  int _regress_end_idx = -1;
  bool _in_regress = false;

 public:
  Context(StatusType status_type = StatusType::Inference, int device_id = 0);
  virtual ~Context();

  AllocatorPtr allocator() { return _allocator_ptr; }

  void convert_into_train();
  void convert_into_eval();

  // Create a process-level global object
  static int create_global_context(
      StatusType status_type = StatusType::Inference, int device_id = -1);
  static std::shared_ptr<Context> global_instance();
  static void set_global_context(int context_id);
  static bool global_is_inference() {
    return Context::global_instance()->is_inference();
  }

  // Before the memory allocation, the tensor is not allocated the actual
  // effective address space, so it is necessary to give a temporary space for
  // some steps to test.
  size_t mx_tensor_size = 0;
  std::string mx_tensor_name = "";
  char* temporary_buffer_ = nullptr;

  std::map<std::string, int> layer_name_cnt;
  std::map<std::string, int> node_name_cnt;

  // property field
  bool is_training() { return _status_type == StatusType::Training; }
  bool is_inference() { return _status_type == StatusType::Inference; }
  const int& node_idx() const { return _node_idx; }
  void update_node_idx();
  const bool& is_built() const { return _built; }
  const bool& is_building() const { return _building; }
  MemoryManagerPtr memory_manager_ptr() { return _mm_ptr; }

  void add_op(Operator* op);
  void add_node(Node* node);

  void enter_layer(Layer* cur_layer, bool is_initial = true);

  // collaborate with enter_layer()
  void exit_layer() { _layer_context.pop_back(); }

  void build();

  void draw_all_context();

  Layer* last_layer();
  Node* last_node();

  // During the model network construction process, mark the start and end
  // positions of the autoregressive part.
  void regress_begin() { _in_regress = true; }
  void regress_end() { _in_regress = false; }

  // Get the start and end timestamps of the autoregressive part of the network
  // structure.
  int regress_begin_idx() { return _regress_begin_idx; }
  int regress_end_idx() { return _regress_end_idx; }

  void update_regr_begin(int node_idx);
  void update_regr_end(int node_idx);
  bool in_regress() { return _in_regress; }

  std::string status_type_str() { return StatusTypeString[_status_type]; }

  // Register model-level global resources in the context object, which is
  // stored in _resources_pool as untyped.
  void register_object(std::string object_name, void* object);

  // Obtain the untyped object registered globally by the model from
  // _resources_pool, and then the user converts it to the required type
  void* get_object(std::string object_name);

  // Synchronous processing of asynchronous operations, usually used for IO
  // processing or debug mode.
  void synchronize();

  static void regist_pybind_layer(std::string layer_name, int layer_id,
                                  std::shared_ptr<void> layer_ptr);
  static std::shared_ptr<void> get_pybind_layer(std::string layer_name,
                                                int layer_id);

#ifdef LIGHTSEQ_cuda
 private:
  cudaStream_t _stream;
  cublasHandle_t _cublasHandle;

 public:
  const cudaStream_t& get_stream() const { return _stream; }
  const cublasHandle_t& get_cublashandle() const { return _cublasHandle; }
  void set_stream(cudaStream_t stream);
#endif
};

}  // namespace lightseq
