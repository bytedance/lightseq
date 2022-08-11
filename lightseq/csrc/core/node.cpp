#include "node.h"

namespace lightseq {

Node::Node(std::string name) : _context_ptr(thread_context_ptr.get()) {
  int idx = _context_ptr->node_name_cnt[name];
  _context_ptr->node_name_cnt[name] += 1;
  _name = name + "_" + std::to_string(idx);
  _bw_first_flag = true;
  thread_context_ptr->add_node(this);
}

Node::~Node() {
  // printf("~Node() %s\n", _name.c_str());
  _parents.clear();
  _children.clear();
}

void Node::set_parents(std::vector<Node*> parents) {
  for (Node* iter : parents) {
    _parents.push_back(iter);
    iter->add_child(this);
  }
}

void Node::recursive_forward() {
  if (_fw_flag) return;
  for (Node* iter : _parents) {
    iter->recursive_forward();
  }

  _fw_flag = true;
  // auto real_context_ptr = _context_ptr.lock();
  _context_ptr->update_node_idx();
  forward();
}

void Node::recursive_backward() {
  if (_bw_flag) return;
  for (Node* iter : _children) {
    iter->recursive_backward();
  }

  _bw_flag = true;
  // auto real_context_ptr = _context_ptr.lock();
  _context_ptr->update_node_idx();
  backward();
}

bool Node::is_cover() {  // true means assign, false means accumulate
  if (this->_bw_first_flag) {
    this->_bw_first_flag = false;
    return true;
  }
  if (this->_children.size() > 1) {
    return true;
  }
  return false;
}

Variable::Variable(std::string name, size_t mx_size, size_t sizeof_value,
                   size_t sizeof_grad)
    : Node(name), _mx_size(mx_size) {
  _value.reset(
      new Tensor(this->_name + "_value", mx_size * sizeof_value, true));
  // auto real_context_ptr = _context_ptr.lock();
  if (_context_ptr->is_training())
    _grad.reset(new Tensor(this->_name + "_grad", mx_size * sizeof_grad, true));
}

template <class T1, class T2>
Variable::Variable(std::string name, size_t mx_size, const T1* para_ptr,
                   T2* grad_ptr)  // for parameter
    : Node(name), _mx_size(mx_size) {
  // auto real_context_ptr = _context_ptr.lock();
  _value.reset(new Tensor(this->_name + "_value", mx_size * sizeof(T1), false));
  if (_context_ptr->is_training())
    _grad.reset(new Tensor(this->_name + "_grad", mx_size * sizeof(T2), false));
  if (para_ptr) {
    _value->set_tensor(para_ptr);
  }
  if (grad_ptr && _context_ptr->is_training()) {
    _grad->set_tensor(grad_ptr);
  }
}

template Variable::Variable<int, int>(std::string name, size_t mx_size,
                                      const int* para_ptr, int* grad_ptr);

void Variable::fixed_memory() {  // Convert VariableNode to IONode
  if (this->_value->memory_type() != FixedMemory) {
    if (parents().size() > 0 && children().size() > 0) {
      printf("ERROR! this node is not a IONode!\n");
      exit(-1);
    }
    this->_value->reset_fixed();
  }
  // auto real_context_ptr = _context_ptr.lock();
  if (_context_ptr->is_training() &&
      this->_grad->memory_type() != FixedMemory) {
    this->_grad->reset_fixed();
  }
  return;
}

template <class T>
void Variable::set_value(T* value_ptr) {
  fixed_memory();
  _value->set_tensor<T>(value_ptr);
}

template void Variable::set_value<int>(int* value_ptr);
template void Variable::set_value<char>(char* value_ptr);
template void Variable::set_value<float>(float* value_ptr);

template <class T>
void Variable::set_value(const T* value_ptr) {
  fixed_memory();
  _value->set_tensor<T>(value_ptr);
}

template void Variable::set_value<int>(const int* value_ptr);
template void Variable::set_value<char>(const char* value_ptr);
template void Variable::set_value<float>(const float* value_ptr);

template <class T>
void Variable::set_grad(T* grad_ptr) {
  fixed_memory();
  _grad->set_tensor<T>(grad_ptr);
}

template void Variable::set_grad<int>(int* grad_ptr);
template void Variable::set_grad<char>(char* grad_ptr);
template void Variable::set_grad<float>(float* grad_ptr);

char* Variable::value() { return _value->tensor(); }

char* Variable::grad() { return _grad->tensor(); }

bool Variable::enable_override_grad() {
  if (this->_children.size() == 1) {
    return true;
  } else {
    return false;
  }
}

Operator::Operator(std::string name) : Node(name) {
  // auto real_context_ptr = _context_ptr.lock();
  _context_ptr->add_op(this);
}

void Operator::check_override_grad() {
  for (Node* p : this->_parents) {
    Variable* rp = static_cast<Variable*>(p);
    if (!rp->enable_override_grad()) {
      printf("can not override");
      exit(-1);
    }
  }
  return;
}

void Operator::set_children(std::vector<Node*> children) {
  if (!this->_children.empty()) {
    printf("children not empty!");
    exit(-1);
  }
  for (Node* iter : children) {
    iter->set_parents({this});
  }
}

}  // namespace lightseq
