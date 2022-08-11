#pragma once
#include "iostream"
#include "map"

#include "declaration.h"
#include "context.h"
#include "node.h"

namespace lightseq {

class Layer {
 protected:
  ContextPtr _context_ptr;
  std::string _name;
  // static std::map<std::string, int> _name_cnt;
  std::vector<Variable*> _root_var_vec = {};
  std::vector<Variable*> _leaf_var_vec = {};

 public:
  Layer(std::string name);
  virtual ~Layer();

  virtual void forward();

  virtual void backward();

  void clear_fw_flag();

  void clear_bw_flag();

  void gather_root_leaf_var();

  std::vector<Operator*> _op_vec;
};

// std::map<std::string, int> Layer::_name_cnt = {};

}