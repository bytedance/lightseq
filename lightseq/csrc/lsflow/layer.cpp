#include "layer.h"

namespace lightseq {

Layer::Layer(std::string name) : _op_vec({}) {
  _context_ptr = Context::global_instance();
  std::string real_name =
      _context_ptr->last_layer()
          ? (_context_ptr->last_layer()->name() + "/" + name)
          : name;
  int idx = _context_ptr->layer_name_cnt[real_name];
  _context_ptr->layer_name_cnt[real_name] += 1;
  _name = real_name + "_" + std::to_string(idx);
  _context_ptr->enter_layer(this, true);
}

Layer::~Layer() {}

void Layer::forward() {
  _context_ptr->build();
  clear_fw_flag();
  for (Variable* var : _out_var_vec) {
    var->recursive_forward();
  }
}

void Layer::backward() {
  _context_ptr->build();
  clear_bw_flag();
  for (Variable* var : _inp_var_vec) {
    var->recursive_backward();
  }
}

Variable* Layer::input(int idx) {
  if (idx >= _inp_var_vec.size()) {
    printf("ERROR OCCURRED!\n");
    printf("layer %s input idx is out of range!\n", name().c_str());
    exit(0);
  }
  return _inp_var_vec[idx];
}

Variable* Layer::output(int idx) {
  if (idx >= _out_var_vec.size()) {
    printf("ERROR OCCURRED!\n");
    printf("layer %s output idx is out of range!\n", name().c_str());
    exit(0);
  }
  return _out_var_vec[idx];
}

void Layer::clear_fw_flag() {
  for (Operator* op : _op_vec) {
    op->clear_fw_flag();
    for (Node* var : op->children()) {
      var->clear_fw_flag();
    }
  }
}

void Layer::tag_fw_flag() {
  for (Operator* op : _op_vec) {
    op->tag_fw_flag();
    for (Node* var : op->children()) {
      var->tag_fw_flag();
    }
  }
}

void Layer::clear_bw_flag() {
  for (Operator* op : _op_vec) {
    op->clear_bw_flag();
    for (Node* var : op->parents()) {
      var->clear_bw_flag();
    }
  }
}

void Layer::tag_bw_flag() {
  for (Operator* op : _op_vec) {
    op->tag_bw_flag();
    for (Node* var : op->parents()) {
      var->tag_bw_flag();
    }
  }
}

void Layer::gather_root_leaf_var() {
  _leaf_var_vec.clear();
  _root_var_vec.clear();
  for (Operator* op : _op_vec) {
    // gather leaf var
    for (Node* var : op->children()) {
      Variable* vvar = static_cast<Variable*>(var);
      if (var->children().size() == 0) {
        _leaf_var_vec.push_back(vvar);
        vvar->fixed_memory();
      }
    }
    // gather root var
    for (Node* var : op->parents()) {
      Variable* vvar = static_cast<Variable*>(var);
      if (var->parents().size() == 0) {
        _root_var_vec.push_back(vvar);
        vvar->fixed_memory();
      }
    }
  }  // each op
}

}  // namespace lightseq
