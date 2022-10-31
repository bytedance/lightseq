#pragma once
#include "declaration.h"
#include "tensor.h"
#include "context.h"
#include "manager.h"
#include "memory"
#include <cstdlib>
#include <iostream>

namespace lightseq {

class Node {
 protected:
  Context* _context_ptr;
  std::string _name;
  NodeType _node_type;

  bool _fw_flag;
  bool _bw_flag;
  bool _bw_first_flag;

  std::vector<Node*> _parents{};
  std::vector<Node*> _children{};
  int _fw_node_idx, _bw_node_idx;

 public:
  Node(std::string name, NodeType nt_);
  std::string name() { return _name; }
  virtual ~Node();
  NodeType node_type() { return _node_type; }
  void set_parents(std::vector<Node*> parents);

  void add_child(Node* child) { _children.push_back(child); }

  virtual void forward() {}   // need to implement
  virtual void backward() {}  // need to implement

  const std::vector<Node*>& parents() { return _parents; }
  const std::vector<Node*>& children() { return _children; }

  void recursive_forward();

  void recursive_backward();

  void clear_fw_flag() { _fw_flag = false; }
  void tag_fw_flag() { _fw_flag = true; }
  void clear_bw_flag() { _bw_flag = false, _bw_first_flag = true; }
  void tag_bw_flag() { _bw_flag = true; }

  bool is_cover();  // true means assign, false means accumulate
};

// std::map<std::string, int> Node::_name_cnt = {};

class Variable : public Node {
 private:
  size_t _value_byte_size;
  size_t _grad_byte_size;
  TensorPtr _value = nullptr;
  TensorPtr _grad = nullptr;
  bool _is_descendants = false;
  size_t _offset_value;
  size_t _offset_grad;
  Variable* _parent_variable;
  std::unordered_set<Variable*> _children_variable;

 public:
  Variable(std::string name);  // for Fixed memory
  Variable(std::string name, size_t value_byte_size,
           size_t grad_byte_size = 0);  // for Shared memory
  Variable(std::string name, const char* para_ptr,
           char* grad_ptr = nullptr);  // for Fixed memory
  Variable(std::string name, Variable* parent_variable, size_t offset_value = 0,
           size_t offset_grad = 0);  // for inherit Variable

  virtual ~Variable() {}

  void fixed_memory();  // Convert VariableNode to IONode

  void set_value(char* value_ptr);

  void set_value(const char* value_ptr);

  void set_grad(char* grad_ptr);

  /*
    value() / grad() means get the value or grad of this node, when
    is_open_interval is true, it doesn't update the lifecycle of the tensor.
  */
  char* value(bool is_open_interval = false);
  char* grad(bool is_open_interval = false);

  bool enable_override_grad();
  bool is_descendants() { return _is_descendants; }
  bool is_ancestor() { return _children_variable.size(); }
  Variable* ancestor() { return _parent_variable; }
  std::unordered_set<Variable*>& descendants() { return _children_variable; }
  void set_ancestor(Variable* parent_variable, size_t offset_value = 0, size_t offset_grad = 0);
  void set_offset(size_t offset_value, size_t offset_grad);
  void remove_ancestor();
  void add_descendants(Variable* var);
  void remove_descendants(Variable* var);

  friend class Node;

#ifdef DEBUG_MODE
  void debug_var();
#endif
};

class Operator : public Node {
 protected:
 public:
  Operator(std::string name);
  virtual ~Operator() {}
  void check_override_grad();

  void set_children(std::vector<Node*> children);

  Variable* child(int index) {
    return static_cast<Variable*>(_children[index]);
  }

  Variable* parent(int index) {
    return static_cast<Variable*>(_parents[index]);
  }
};
}  // namespace lightseq
