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

  bool _fw_flag;
  bool _bw_flag;
  bool _bw_first_flag;

  std::vector<Node*> _parents{};
  std::vector<Node*> _children{};

 public:
  Node(std::string name);
  std::string name() { return _name; }
  virtual ~Node();

  void set_parents(std::vector<Node*> parents);

  void add_child(Node* child) { _children.push_back(child); }

  virtual void forward() {}   // need to implement
  virtual void backward() {}  // need to implement
  const std::vector<Node*>& parents() { return _parents; }
  const std::vector<Node*>& children() { return _children; }

  void recursive_forward();

  void recursive_backward();

  void clear_fw_flag() { _fw_flag = false; }
  void clear_bw_flag() { _bw_flag = false, _bw_first_flag = true; }

  bool is_cover();  // true means assign, false means accumulate
};

// std::map<std::string, int> Node::_name_cnt = {};

class Variable : public Node {
 private:
  size_t _value_byte_size;
  size_t _grad_byte_size;
  TensorPtr _value = nullptr;
  TensorPtr _grad = nullptr;

 public:
  Variable(std::string name, size_t value_byte_size, size_t grad_byte_size = 0);
  Variable(std::string name, const char* para_ptr, char* grad_ptr = nullptr);
  virtual ~Variable() {}

  void fixed_memory();  // Convert VariableNode to IONode

  void set_value(char* value_ptr);

  void set_value(const char* value_ptr);

  void set_grad(char* grad_ptr);

  char* value();

  char* grad();

  bool enable_override_grad();
};

class Operator : public Node {
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
