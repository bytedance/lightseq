#include "context.h"

namespace lightseq {

Context::Context(StatusType status_type, int device_id)
    : _mm_ptr(new MemoryManager()),
      _device_id(device_id),
      _status_type(status_type) {
  printf("Initial Context, status_type: %s\n", status_type_str().c_str());
  if (device_id >= 0) CHECK_GPU_ERROR(cudaSetDevice(device_id));
  CHECK_GPU_ERROR(cudaStreamCreate(&_stream));
  CHECK_GPU_ERROR(cublasCreate(&_cublasHandle));
  CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
}

Context::~Context() {
  for (auto& iter : _all_node_vec) {
    delete iter;
  }
}

void Context::set_stream(cudaStream_t stream) {
  _stream = stream;
  CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
}

void Context::convert_into_train() { _status_type = StatusType::Training; }

void Context::convert_into_eval() {
  if (_status_type != StatusType::Inference)
    _status_type = StatusType::Evaluation;
}

int Context::create_global_context(StatusType status_type, int device_id) {
  global_context_id++;
  std::shared_ptr<Context> new_context =
      std::make_shared<Context>(status_type, device_id);
  _global_context_ptr = new_context;
  global_contexts_map.emplace(global_context_id, new_context);
  return global_context_id;
}

void Context::set_global_context(int context_id) {
  auto iter = global_contexts_map.find(context_id);
  if (iter == global_contexts_map.end()) {
    printf("Error occured! context_id %d does not exist!\n", context_id);
    exit(-1);
  }
  _global_context_ptr = iter->second;
}

std::shared_ptr<Context> Context::global_instance() {
  return _global_context_ptr;
}

void Context::update_node_idx() {
  if (_built) return;
  _node_idx++;
}

void Context::add_op(Operator* op) {
  if (is_built()) {
    printf("Context has constructed! should not add new operator!\n");
    exit(-1);
  }

  if (_layer_context.size()) {
    for (Layer* lyr : _layer_context) {
      lyr->_op_vec.push_back(op);
    }
    return;
  }
#if ONLY_OP == true
  _model_ops.push_back(op);
#else
  printf("ERROR! don't use operator directly!\n");
  printf("Node name: %s\n", op->name().c_str());
  exit(-1);
#endif
}
void Context::add_node(Node* node) { _all_node_vec.push_back(node); }

void Context::enter_layer(Layer* cur_layer, bool is_initial) {
  if (is_built()) {
    printf("Context has constructed! should not modify network\n");
    exit(-1);
  }

  if (_layer_context.size() == 0 && is_initial == false) {
    _root_layers.push_back(cur_layer);
  } else if (is_initial == true) {
    _all_layers.push_back(cur_layer);
  }
  _layer_context.push_back(cur_layer);
}

void Context::build() {
  if (_built || _building) {
    return;
  }
  _building = true;

#ifdef DEBUG_MODE
  printf(
      "========== start Context build, StatusType: %s, StatusType id: %d "
      "==========\n",
      status_type_str().c_str(), int(_status_type));
#endif

  if (!check_validate()) {
    printf("Check validate error!\n");
    exit(-1);
  }

  temporary_buffer_ = cuda_malloc<char>(mx_tensor_size);

#if ONLY_OP == true
  for (int idx = 0; idx < _model_ops.size(); idx++) {
    _model_ops[idx]->recursive_forward();
  }
  if (is_training()) {
    for (int idx = _model_ops.size() - 1; idx >= 0; idx--) {
      _model_ops[idx]->recursive_backward();
    }
  }
#endif

  for (Layer* rl : _root_layers) {
    // rl->gather_root_leaf_var();
#ifdef DEBUG_MODE
    printf("\n########## Context build layer %s forward ##########\n",
           rl->name().c_str());
#endif
    rl->forward();
  }

  if (is_training()) {
    printf("is training!\n");
    for (int idx = _root_layers.size() - 1; idx >= 0; idx--) {
      Layer* rl = _root_layers[idx];
#ifdef DEBUG_MODE
      printf("\n########## Context build layer %s backward ##########\n",
             rl->name().c_str());
#endif
      rl->backward();
    }
  }

  for (auto iter : _all_node_vec) {
    if (iter->node_type() == NodeType::Variable) {
      static_cast<Variable*>(iter)->update_regress_idx();
    }
  }

  cuda_free(temporary_buffer_);
  _mm_ptr->calculate_buffer_();
  _built = true;

  CHECK_GPU_ERROR(cudaStreamSynchronize(get_stream()));

#ifdef DEBUG_MODE
  draw_all_context();
  printf("===== finish Context build =====\n");
#endif
}

bool Context::check_validate() {
  bool check_flag = true;
  for (Layer* lyr : _all_layers) {
    if (lyr->name().size() == 0) {
      printf("error! some LAYERS didn't initialize!\n");
      check_flag = false;
    }
  }

  for (Operator* op : _model_ops) {
    if (op->name().size() == 0) {
      printf("error! some OPERATORS didn't initialize!\n");
      check_flag = false;
    }
  }

  return check_flag;
}

void Context::draw_all_context() {}

void Context::regist_pybind_layer(std::string layer_name, int layer_id,
                                  std::shared_ptr<void> layer_ptr) {
  std::string full_name = layer_name + std::to_string(layer_id);
  if (pybind_layers.find(full_name) != pybind_layers.end()) {
    printf(
        "The layer applied for registration has been occupied!\n"
        "Layer name is %s!\n",
        full_name.c_str());
    throw std::runtime_error(
        "The layer applied for registration has been occupied!\n");
  }
#ifdef DEBUG_MODE
  printf("regist_pybind_layer %s\n", full_name.c_str());
#endif
  pybind_layers.emplace(full_name, layer_ptr);
}

void Context::update_regr_begin(int node_idx) {
  if (node_idx < 0) {
    printf("Error! update_regr_begin with node_idx %d\n", node_idx);
    exit(-1);
  }
  _regress_begin_idx = (_regress_begin_idx == -1)
                           ? node_idx
                           : std::min(node_idx, _regress_begin_idx);
}

void Context::update_regr_end(int node_idx) {
  if (node_idx < 0) {
    printf("Error! update_regr_begin with node_idx %d\n", node_idx);
    exit(-1);
  }
  _regress_end_idx = (_regress_end_idx == -1)
                         ? node_idx
                         : std::max(node_idx, _regress_end_idx);
}

void Context::register_object(std::string object_name, void* object) {
  if (_resources_pool.find(object_name) != _resources_pool.end()) {
    printf("Error! register same name(%s) twice!\n", object_name.c_str());
    exit(-1);
  }
  _resources_pool.emplace(object_name, object);
}

void* Context::get_object(std::string object_name) {
  auto iter = _resources_pool.find(object_name);
  if (iter == _resources_pool.end()) {
    printf("Error! can't get %s\n", object_name.c_str());
    exit(-1);
  }
  return iter->second;
}

std::shared_ptr<void> Context::get_pybind_layer(std::string layer_name,
                                                int layer_id) {
  std::string full_name = layer_name + std::to_string(layer_id);
  auto iter = pybind_layers.find(full_name);
  if (iter == pybind_layers.end()) {
    printf(
        "The requested layer was not found!\n"
        "Layer name is %s!\n",
        full_name.c_str());
    throw std::runtime_error("The requested layer was not found!");
  }
  return iter->second;
}

std::shared_ptr<Context> Context::_global_context_ptr = nullptr;
std::unordered_map<std::string, std::shared_ptr<void>> Context::pybind_layers =
    {};
std::unordered_map<int, std::shared_ptr<Context>> Context::global_contexts_map =
    {};
int Context::global_context_id = 0;

}  // namespace lightseq
