#include "node.h"

namespace lightseq {

Node::Node(std::string name) : _context_ptr(thread_context_ptr.get()) {

  //printf("Running Step.2.6\n");
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

Variable::Variable(std::string name, size_t value_byte_size,
                   size_t grad_byte_size)
    : Node(name),
      _value_byte_size(value_byte_size),
      _grad_byte_size(grad_byte_size) {
  _value.reset(new Tensor(this->_name + "/value", _value_byte_size));
  if (_context_ptr->is_training())
    _grad.reset(new Tensor(this->_name + "/grad", _grad_byte_size));
}

Variable::Variable(std::string name, const char* para_ptr, char* grad_ptr)
    : Variable(name, (size_t)0, (size_t)0) {
  _value->set_tensor(para_ptr);
  if (_grad) {
    _grad->set_tensor(grad_ptr);
  }
}

void Variable::fixed_memory() {
  if (parents().size() > 0 && children().size() > 0) {
    printf("ERROR! this node is not a IONode!\n");
    exit(-1);
  }
  _value->reset_fixed();
  if (_grad) {
    _grad->reset_fixed();
  }
  return;
}

void Variable::set_value(char* value_ptr) {
  _value->reset_fixed();
  _value->set_tensor(value_ptr);
}

void Variable::set_value(const char* value_ptr) {
  _value->reset_fixed();
  _value->set_tensor(value_ptr);
}

void Variable::set_grad(char* grad_ptr) {
  _grad->reset_fixed();
  _grad->set_tensor(grad_ptr);
}

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
  //printf("Running Step.2.5\n");
  std::cout << "_context_ptr " << _context_ptr << std::endl;
  // auto real_context_ptr = _context_ptr.lock();
  _context_ptr->add_op(this);
  //printf("Running Step.2.7\n");
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
