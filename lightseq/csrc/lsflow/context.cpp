#include "context.h"

namespace lightseq {

Context::Context(bool training, int device_id)
    : _mm_ptr(new MemoryManager()),
      _is_training(training),
      _device_id(device_id) {
  CHECK_GPU_ERROR(cudaSetDevice(device_id));
  CHECK_GPU_ERROR(cudaStreamCreate(&_stream));
  CHECK_GPU_ERROR(cublasCreate(&_cublasHandle));
  CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
}

Context::~Context() {
  _root_layers.clear();
  _layer_context.clear();
  for (auto& iter : _all_node_vec) {
    delete iter;
  }
  _all_node_vec.clear();
}

void Context::new_thread_context(bool training) {
  thread_context_ptr.reset(new Context(training));
}

void Context::set_thread_context(ContextPtr context_ptr) {
  thread_context_ptr = context_ptr;
}

void Context::remove_thread_context() { thread_context_ptr.reset(); }

void Context::add_op(Operator* op) {
  if (_layer_context.size()) _layer_context[0]->_op_vec.push_back(op);
  _all_op_vec.push_back(op);
}
void Context::add_node(Node* node) { _all_node_vec.push_back(node); }

void Context::enter_layer(Layer* cur_layer) {
  if (_layer_context.size() == 0) {
    _root_layers.push_back(cur_layer);
  }
  _layer_context.push_back(cur_layer);
}

void Context::build() {
  if (_built || _building) {
    return;
  }
  _building = true;

  printf("Running DEBUG.1! %zu\n", mx_tensor_size);
  temporary_buffer_ = cuda_malloc<char>(mx_tensor_size);

  printf("before fake_forward!\n");
  for (Layer* rl : _root_layers) {
    rl->gather_root_leaf_var();
    rl->forward();
  }
  printf("after fake_forward!\n");

  if (_is_training) {
    for (Layer* rl : _root_layers) {
      rl->backward();
    }
  }

#ifdef ONLY_OP
  for (int idx = 0; idx < _all_op_vec.size(); idx++) {
    _all_op_vec[idx]->forward();
  }
  if (is_training()) {
    for (int idx = _all_op_vec.size() - 1; idx >= 0; idx--) {
      _all_op_vec[idx]->backward();
    }
  }
#endif

  cuda_free(temporary_buffer_);
  _mm_ptr->calculate_buffer_();
  _built = true;

#ifndef ONLY_OP
  thread_context_ptr.reset();
#endif

#ifdef DEBUG
  draw_all_context();
#endif
}

thread_local ContextPtr thread_context_ptr = nullptr;

void Context::draw_all_context() {}

// thread_local ContextPtr thread_context_ptr = nullptr;

}  // namespace lightseq
