#include "node.h"

#ifdef FP16_MODE
typedef __half TENSOR_TYPE;
#else
typedef float TENSOR_TYPE;
#endif

namespace lightseq {

Node::Node(std::string name, NodeType nt_)
    : _context_ptr(Context::global_instance().get()),
      _bw_first_flag(true),
      _node_type(nt_) {
  std::string prefix_name = _context_ptr->last_layer()
                                ? (_context_ptr->last_layer()->name() + ":")
                                : "";
  std::string real_name = prefix_name + name;
  int idx = _context_ptr->node_name_cnt[real_name];
  _context_ptr->node_name_cnt[real_name] += 1;
  _name = real_name + "_" + std::to_string(idx);
  _context_ptr->add_node(this);
}

Node::~Node() {
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

  _context_ptr->update_node_idx();

#ifdef DEBUG_MODE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  forward();

#ifdef DEBUG_MODE
  if (node_type() != NodeType::Operator || !_context_ptr->is_built()) {
    return;
  }
  CHECK_GPU_ERROR(cudaStreamSynchronize(_context_ptr->get_stream()));
  printf("##### %s forward #####\n", name().c_str());
  print_time_duration(start, "time cost", 0);
  Operator* this_op = static_cast<Operator*>(this);
  for (int idx = 0; idx < _parents.size(); idx++) {
    if (_parents[idx] != nullptr && this_op->parent(idx)->value() != nullptr)
      print_vec((TENSOR_TYPE*)this_op->parent(idx)->value(),
                this_op->parent(idx)->name() + ":value", 10);
  }
  for (int idx = 0; idx < _children.size(); idx++) {
    if (_children[idx] != nullptr && this_op->child(idx)->value() != nullptr)
      print_vec((TENSOR_TYPE*)this_op->child(idx)->value(),
                this_op->child(idx)->name() + ":value", 10);
  }
  printf("\n");
#endif
}

void Node::recursive_backward() {
  if (_bw_flag) return;
  for (Node* iter : _children) {
    iter->recursive_backward();
  }

  _bw_flag = true;
  _context_ptr->update_node_idx();

#ifdef DEBUG_MODE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  backward();

#ifdef DEBUG_MODE
  if (node_type() != NodeType::Operator || !_context_ptr->is_built()) {
    return;
  }
  CHECK_GPU_ERROR(cudaStreamSynchronize(_context_ptr->get_stream()));
  printf("##### %s backward #####\n", name().c_str());
  print_time_duration(start, "time cost", 0);
  Operator* this_op = static_cast<Operator*>(this);
  for (int idx = 0; idx < _parents.size(); idx++) {
    if (_parents[idx] != nullptr && this_op->parent(idx)->grad() != nullptr)
      print_vec((TENSOR_TYPE*)this_op->parent(idx)->grad(),
                this_op->parent(idx)->name() + ":grad", 10);
  }
  for (int idx = 0; idx < _children.size(); idx++) {
    if (_children[idx] != nullptr && this_op->child(idx)->grad() != nullptr)
      print_vec((TENSOR_TYPE*)this_op->child(idx)->grad(),
                this_op->child(idx)->name() + ":grad", 10);
  }
  printf("\n");
#endif
}

bool Node::is_cover() {  // true means assign, false means accumulate
  if (this->_bw_first_flag) {
    this->_bw_first_flag = false;
    return true;
  }
  return false;
}

Variable::Variable(std::string name)
    : Node(name, NodeType::FixedVariable),
      _value_byte_size(0),
      _grad_byte_size(0) {
  _value.reset(new Tensor("value", 0));
  if (_context_ptr->is_training()) _grad.reset(new Tensor("grad", 0));
}

Variable::Variable(std::string name, size_t value_byte_size,
                   size_t grad_byte_size)
    : Node(name, NodeType::SharedVariable),
      _value_byte_size(value_byte_size),
      _grad_byte_size(grad_byte_size) {
  _value.reset(new Tensor("value", _value_byte_size));
  if (_context_ptr->is_training())
    _grad.reset(new Tensor("grad", _grad_byte_size));
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
  if (_context_ptr->is_training()) {
    _grad->reset_fixed();
    _grad->set_tensor(grad_ptr);
  }
}

char* Variable::value(bool is_open_interval) {
  return _value->tensor(is_open_interval);
}

char* Variable::grad() { return _grad->tensor(); }

bool Variable::enable_override_grad() {
  if (this->_children.size() == 1) {
    return true;
  } else {
    return false;
  }
}

Operator::Operator(std::string name) : Node(name, NodeType::Operator) {
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
