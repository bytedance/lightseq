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
  _context_ptr->update_node_idx();

  forward_process();
  for (Variable* var : _out_var_vec) {
    var->recursive_forward();
  }
}

void Layer::backward() {
  _context_ptr->build();
  clear_bw_flag();
  _context_ptr->update_node_idx();

  backward_process();
  for (Variable* var : _inp_var_vec) {
    var->recursive_backward();
  }
}

void Layer::set_inputs(std::vector<Variable*> inps) {
  _inp_var_vec = inps;
  _context_ptr->enter_layer(this, false);
  macro_inputs_check = true;
}

void Layer::set_outputs(std::vector<Variable*> outs) {
  _out_var_vec = outs;
  _context_ptr->exit_layer();
  macro_outputs_check = true;
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
      Variable* this_var = static_cast<Variable*>(var);
      for (Variable* iter : this_var->descendants()) {
        iter->clear_fw_flag();
      }
      var->clear_fw_flag();
    }
  }
}

void Layer::tag_fw_flag() {
  for (Operator* op : _op_vec) {
    op->tag_fw_flag();
    for (Node* var : op->children()) {
      Variable* this_var = static_cast<Variable*>(var);
      for (Variable* iter : this_var->descendants()) {
        iter->tag_fw_flag();
      }
      var->tag_fw_flag();
    }
  }
}

void Layer::clear_bw_flag() {
  for (Operator* op : _op_vec) {
    op->clear_bw_flag();
    for (Node* var : op->parents()) {
      Variable* this_var = static_cast<Variable*>(var);
      if (this_var->variable_type() == VariableType::OffsetVariable)
        this_var->ancestor()->clear_bw_flag();
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

}  // namespace lightseq
