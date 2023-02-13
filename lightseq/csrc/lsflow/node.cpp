#include "node.h"

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
  if (_context_ptr->in_regress()) {
    _in_regress_scope = true;
  }

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
  if (node_type() == NodeType::Operator) {
    if (_context_ptr->in_regress()) {
      _in_regress_scope = true;
    }
  }
}

void Node::recursive_forward() {
  if (_fw_flag) return;
  for (Node* iter : _parents) {
    iter->recursive_forward();
  }

  if (node_type() == NodeType::Variable) {
    Variable* this_var = static_cast<Variable*>(this);
    if (this_var->variable_type() == VariableType::OffsetVariable) {
      this_var->_parent_variable->recursive_forward();
    }
  }

  _fw_flag = true;
  if (node_type() == NodeType::Operator) {
    _context_ptr->update_node_idx();
  }
  if (!_context_ptr->is_built()) {
    _fw_node_idx = _context_ptr->node_idx();
    if (_in_regress_scope) {
      _context_ptr->update_regr_begin(_fw_node_idx);
      _context_ptr->update_regr_end(_fw_node_idx);
    }
  }

#ifdef DEBUG_MODE
  if (node_type() == NodeType::Operator) {
    printf("##### %s forward ##### fw node idx: %d\n", name().c_str(),
           _fw_node_idx);
    Operator* this_op = static_cast<Operator*>(this);
    printf("_parents.size(): %zu\n", _parents.size());
    for (int idx = 0; idx < _parents.size(); idx++) {
      if (_parents[idx] == nullptr) {
        printf("nullptr!\n");
      } else {
        this_op->parent(idx)->print_var(true);
      }
    }
  }
  _context_ptr->synchronize();
  auto start = std::chrono::high_resolution_clock::now();
#endif

  forward();

#ifdef DEBUG_MODE
  if (node_type() != NodeType::Operator) {
    return;
  }
  _context_ptr->synchronize();
  print_time_duration(start, "time cost");
  Operator* this_op = static_cast<Operator*>(this);
  printf("_children.size(): %zu\n", _children.size());
  for (int idx = 0; idx < _children.size(); idx++) {
    if (_children[idx] == nullptr)
      printf("nullptr\n");
    else
      this_op->child(idx)->print_var(true);
  }
  printf("\n");
#endif
}

void Node::recursive_backward() {
  if (_bw_flag) return;
  for (Node* iter : _children) {
    iter->recursive_backward();
  }

  if (node_type() == NodeType::Variable) {
    Variable* this_var = static_cast<Variable*>(this);
    for (Variable* iter : this_var->descendants()) {
      iter->recursive_backward();
    }
  }

  _bw_flag = true;
  if (node_type() == NodeType::Operator) {
    _context_ptr->update_node_idx();
  }
  if (!_context_ptr->is_built()) {
    _bw_node_idx = _context_ptr->node_idx();
    if (_in_regress_scope) {
      _context_ptr->update_regr_begin(_bw_node_idx);
      _context_ptr->update_regr_end(_bw_node_idx);
    }
  }

#ifdef DEBUG_MODE
  if (node_type() == NodeType::Operator) {
    printf("##### %s backward ##### bw node idx: %d\n", name().c_str(),
           _bw_node_idx);
    Operator* this_op = static_cast<Operator*>(this);
    printf("_children.size(): %zu\n", _children.size());
    for (int idx = 0; idx < _children.size(); idx++) {
      if (_children[idx] == nullptr)
        printf("nullptr\n");
      else
        this_op->child(idx)->print_var(false);
    }
  }
  _context_ptr->synchronize();
  auto start = std::chrono::high_resolution_clock::now();
#endif

  backward();

#ifdef DEBUG_MODE
  if (node_type() != NodeType::Operator) {
    return;
  }
  _context_ptr->synchronize();
  print_time_duration(start, "time cost");
  Operator* this_op = static_cast<Operator*>(this);
  printf("_parents.size(): %zu\n", _parents.size());
  for (int idx = 0; idx < _parents.size(); idx++) {
    if (_parents[idx] == nullptr)
      printf("nullptr\n");
    else
      this_op->parent(idx)->print_var(false);
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

}  // namespace lightseq
