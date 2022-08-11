#pragma once
#include "cstdio"
#include "queue"
#include "deque"
#include "stack"

#include "declaration.h"
#include "manager.h"
#include "layer.h"
#include "node.h"

namespace lightseq {

class Context {  // model only
 private:
  std::vector<Node*> _all_node_vec{};
  std::vector<Layer*> _root_layers{};
  std::deque<Layer*> _layer_context;
  bool _is_training = false;

  bool _built = false;
  bool _building = false;
  int _node_idx = 0;
  MemoryManagerPtr memory_manager_;

 public:
  Context(bool training = false)
      : memory_manager_(new MemoryManager()), _is_training(training) {}
  virtual ~Context();

  static void new_thread_context(bool training = false);

  static void set_thread_context(ContextPtr context_ptr);

  // for initial calculation
  size_t mx_tensor_size = 0;
  char* temporary_buffer_ = nullptr;

  std::map<std::string, int> layer_name_cnt;
  std::map<std::string, int> node_name_cnt;

  // property field
  bool is_training() { return _is_training; }
  int node_idx() { return _node_idx; }
  void update_node_idx() {
    if (_built) return;
    _node_idx++;
  }
  bool built() { return _built; }
  MemoryManagerPtr memory_manager_ptr() { return memory_manager_; }

  void add_op(Operator* op);
  void add_node(Node* node);

  void enter_layer(Layer* cur_layer);

  // collaborate with enter_layer()
  void exit_layer() { _layer_context.pop_back(); }

  void build();

  void draw_all_context();
};

}  // namespace lightseq
