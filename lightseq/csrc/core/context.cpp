#include "context.h"

namespace lightseq {

Context::~Context() {
  // printf("~Context()\n");
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

void Context::add_op(Operator* op) { _layer_context[0]->_op_vec.push_back(op); }
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

  temporary_buffer_ = (char*)malloc(mx_tensor_size);

  for (Layer* rl : _root_layers) {
    rl->gather_root_leaf_var();
    rl->forward();
  }

  if (_is_training) {
    for (Layer* rl : _root_layers) {
      rl->backward();
    }
  }

  free(temporary_buffer_);
  memory_manager_->calculate_buffer_();
  _built = true;

  // thread_context_ptr.reset(nullptr);

#ifdef DEBUG
  draw_all_context();
#endif
}

void Context::draw_all_context() {}

}  // namespace lightseq
