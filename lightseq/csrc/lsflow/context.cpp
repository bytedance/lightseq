#include "context.h"

namespace lightseq {

Context::Context(StatusType status_type, int device_id)
    : _mm_ptr(new MemoryManager()),
      _device_id(device_id) {
  CHECK_GPU_ERROR(cudaSetDevice(device_id));
  CHECK_GPU_ERROR(cudaStreamCreate(&_stream));
  CHECK_GPU_ERROR(cublasCreate(&_cublasHandle));
  CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
}

Context::~Context() {
  for (auto& iter : _all_node_vec) {
    delete iter;
  }
}

void Context::convert_into_train() {
  _st = StatusType::Training;
}

void Context::convert_into_eval(){ 
  if(_st != StatusType::Inference) 
    _st = StatusType::Evaluation; 
}

void Context::create_global_context(StatusType status_type, int device_id) {
  _global_context_ptr.reset(new Context(status_type, device_id));
}

void Context::set_global_context(ContextPtr context_ptr) {
  _global_context_ptr = context_ptr;
}


void Context::add_op(Operator* op) {
  if (built()) {
    printf("Context has constructed! should not add new operator!\n");
    exit(-1);
  }

  if (_layer_context.size()) {
    _layer_context[0]->_op_vec.push_back(op);
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
  if (built()) {
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

  printf("===== start Context build =====\n");

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
    rl->gather_root_leaf_var();
    rl->forward();
  }

  if (is_training()) {
    for (int idx = _root_layers.size() - 1; idx >= 0; idx--) {
      Layer* rl = _root_layers[idx];
      rl->backward();
    }
  }

  cuda_free(temporary_buffer_);
  _mm_ptr->calculate_buffer_();
  _built = true;

#ifdef DEBUG_TYPE
  draw_all_context();
#endif

  printf("===== finish Context build =====\n");
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

std::shared_ptr<Context> Context::_global_context_ptr = nullptr;

}  // namespace lightseq
