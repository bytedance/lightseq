#pragma once
#include "cstdio"
#include "queue"
#include "deque"
#include "stack"

#include "declaration.h"
#include "manager.h"
#include "layer.h"
#include "node.h"
#include "cuda_util.h"
#include "unordered_map"

namespace lightseq {

enum StatusType { Training, Inference, Evaluation };
const std::string StatusTypeString[] = {"Training", "Inference", "Evaluation"};

class Context {  // model only
 private:
  static std::unordered_map<std::string, std::shared_ptr<void>> pybind_layers;

  std::vector<Node*> _all_node_vec{};
  std::vector<Operator*> _model_ops{};
  std::vector<Layer*> _root_layers{};
  std::vector<Layer*> _all_layers{};
  std::deque<Layer*> _layer_context;
  StatusType _st;

  bool _built = false;
  bool _building = false;
  int _node_idx = 0;
  MemoryManagerPtr _mm_ptr;

  int _device_id;

  cudaStream_t _stream;
  cublasHandle_t _cublasHandle;

  static std::shared_ptr<Context> _global_context_ptr;

  bool check_validate();

 public:
  Context(StatusType status_type = StatusType::Inference, int device_id = 0);
  virtual ~Context();

  cudaStream_t get_stream() { return _stream; }

  cublasHandle_t get_cublashandle() { return _cublasHandle; }

  void set_stream(cudaStream_t stream) {
    _stream = stream;
    CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
  }

  void convert_into_train();
  void convert_into_eval();

  static void create_global_context(
      StatusType status_type = StatusType::Inference, int device_id = 0);
  static void set_global_context(ContextPtr context_ptr);
  static std::shared_ptr<Context> global_instance() {
    return _global_context_ptr;
  }

  // for initial calculation
  size_t mx_tensor_size = 0;
  char* temporary_buffer_ = nullptr;

  std::map<std::string, int> layer_name_cnt;
  std::map<std::string, int> node_name_cnt;

  // property field
  bool is_training() { return _st == StatusType::Training; }
  int node_idx() { return _node_idx; }
  void update_node_idx() {
    if (_built) return;
    _node_idx++;
  }
  bool built() { return _built; }
  MemoryManagerPtr memory_manager_ptr() { return _mm_ptr; }

  void add_op(Operator* op);
  void add_node(Node* node);

  void enter_layer(Layer* cur_layer, bool is_initial = true);

  // collaborate with enter_layer()
  void exit_layer() { _layer_context.pop_back(); }

  void build();

  void draw_all_context();

  Layer* last_layer() {
    return _layer_context.size() ? _layer_context.back() : nullptr;
  }
  Node* last_node() {
    return _all_node_vec.size() ? _all_node_vec[_all_node_vec.size() - 1]
                                : nullptr;
  }

  static void regist_pybind_layer(std::string layer_name, int layer_id,
                                  std::shared_ptr<void> layer_ptr);
  static std::shared_ptr<void> get_pybind_layer(std::string layer_name,
                                                int layer_id);
};

}  // namespace lightseq
