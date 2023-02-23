#include "manager.h"

namespace lightseq {
void MemoryManager::update_tensor_life_idx(int unique_id, int node_idx,
                                           size_t size, std::string name) {
  if (size == 0) {
    return;
  }
  std::map<int, TensorUsage>::iterator iter = tensor_usages_.find(unique_id);
  if (iter == tensor_usages_.end()) {
    tensor_usages_.emplace(
        unique_id, TensorUsage(unique_id, node_idx, node_idx, size, name));
    return;
  }

  iter->second.first_idx = std::min(iter->second.first_idx, node_idx);

  iter->second.last_idx = std::max(iter->second.last_idx, node_idx);

  return;
}

void MemoryManager::remove_life_cycle(int unique_id) {
  if (tensor_usages_.find(unique_id) != tensor_usages_.end()) {
    tensor_usages_.erase(unique_id);
  }
}

void MemoryManager::calculate_buffer_() {
  printf("========== Execute MemoryManager calculate_buffer_ ==========\n\n");

  tensor_ptr.clear();
  std::vector<std::pair<TensorUsage, size_t>> tensor_usages_vec{};
  for (auto iter : tensor_usages_) {
    tensor_usages_vec.push_back(std::make_pair(iter.second, 0));
  }
  std::sort(tensor_usages_vec.begin(), tensor_usages_vec.end(),
            [](const std::pair<TensorUsage, size_t> &x,
               const std::pair<TensorUsage, size_t> &y) -> bool {
              return x.first.size > y.first.size;
            });

  // Algorithm.3: Greedy by Size for Offset Calculation
  // arxiv url: https://arxiv.org/abs/2001.03288
  // tensor_usages_vec means: <TensorUsage, offset>
  size_t total_consumption = 0;
  std::vector<std::pair<TensorUsage, size_t>> ordered_tensor_usages{};

  for (int idx = 0; idx < tensor_usages_vec.size(); idx++) {
    size_t prev_offset = 0;
    size_t best_offset = 0;
    bool best_offset_flag = false;
    size_t smallest_gap = SIZE_MAX;
    TensorUsage cal_tensor_usage = tensor_usages_vec[idx].first;
    for (auto allocated_tensor : ordered_tensor_usages) {
      TensorUsage allocated_tensor_usage = allocated_tensor.first;
      size_t max_first_op = std::max(cal_tensor_usage.first_idx,
                                     allocated_tensor_usage.first_idx);
      size_t min_last_op =
          std::min(cal_tensor_usage.last_idx, allocated_tensor_usage.last_idx);
      size_t allocated_offset = allocated_tensor.second;
      if (max_first_op <= min_last_op) {
        size_t gap = allocated_offset - prev_offset;
        if (allocated_offset > prev_offset && gap >= cal_tensor_usage.size &&
            gap < smallest_gap) {  // Note the subtraction handling for unsigned
                                   // types
          smallest_gap = gap;
          best_offset = prev_offset;
          best_offset_flag = true;
        }
        prev_offset = std::max(prev_offset,
                               allocated_offset + allocated_tensor_usage.size);
      }
    }
    if (!best_offset_flag) {
      best_offset = prev_offset;
    }
    tensor_usages_vec[idx].second = best_offset;
    ordered_tensor_usages.push_back(tensor_usages_vec[idx]);

    std::sort(ordered_tensor_usages.begin(), ordered_tensor_usages.end(),
              [](const std::pair<TensorUsage, size_t> &x,
                 const std::pair<TensorUsage, size_t> &y) -> bool {
                return x.second < y.second;
              });
    total_consumption =
        std::max(total_consumption, best_offset + cal_tensor_usage.size);
  }
  _total_buffer_size = total_consumption;

  printf("******** shared buffer memory size: %zu MB ********\n",
         total_consumption / MB_SIZE);

  for (auto iter : buffer_vec_) {
    _allocator_ptr->free_mem(iter);
  }
  buffer_vec_.clear();

  size_t max_last_addr = 0;
  size_t record_last_addr = 0;
  std::vector<std::pair<TensorUsage, size_t>> temp_usages_vec{};

  int buffer_idx = 0;
  for (int i = 0; i < ordered_tensor_usages.size(); i++) {
    max_last_addr =
        std::max(max_last_addr, (size_t)(ordered_tensor_usages[i].first.size +
                                         ordered_tensor_usages[i].second));
    temp_usages_vec.push_back(ordered_tensor_usages[i]);
    if ((i + 1 == ordered_tensor_usages.size()) ||
        (max_last_addr == ordered_tensor_usages[i + 1].second)) {
      printf("****** Buffer Idx: %d, buffer memory: %.2f MB, ", buffer_idx,
             float(max_last_addr - record_last_addr) / MB_SIZE);

      char *current_buffer = nullptr;
      try {
        current_buffer =
            _allocator_ptr->malloc_mem(max_last_addr - record_last_addr);
      } catch (...) {
        std::string error_message =
            ("allocate shared buffer " + std::to_string(buffer_vec_.size()) +
             " failed!\n"
             "buffer size is: " +
             std::to_string((max_last_addr - record_last_addr) / MB_SIZE) +
             " MB\n");
        throw std::runtime_error(error_message);
      }

      printf("allocate success! ******\n");

      buffer_vec_.push_back(current_buffer);
      buffer_size_vec_.push_back(max_last_addr - record_last_addr);

      buffer_idx++;
      for (auto iter : temp_usages_vec) {
        int unique_id = iter.first.unique_id;
        tensor_ptr.emplace(unique_id,
                           current_buffer + iter.second - record_last_addr);
      }
      temp_usages_vec.clear();
      record_last_addr = max_last_addr;
    }
  }

  // Add algorithm check module
  // return true means check success,
  auto judge_func = [](const std::pair<TensorUsage, size_t> &x,
                       const std::pair<TensorUsage, size_t> &y) {
    auto max_time_l = std::max(x.first.first_idx, y.first.first_idx);
    auto min_time_r = std::min(x.first.last_idx, y.first.last_idx);
    if (min_time_r < max_time_l) {
      return true;
    }
    auto max_space_l = std::max(x.second, y.second);
    auto min_space_r =
        std::min(x.first.size + x.second, y.first.size + y.second);
    if (min_space_r <= max_space_l) {
      return true;
    }
    return false;
  };
  temp_usages_vec.clear();
  // print order
  std::sort(tensor_usages_vec.begin(), tensor_usages_vec.end(),
            [](const std::pair<TensorUsage, size_t> &x,
               const std::pair<TensorUsage, size_t> &y) -> bool {
              // return x.first.first_idx < y.first.first_idx;
              if (x.second != y.second) return x.second < y.second;
              if (x.second + x.first.size != y.second + y.first.size)
                return x.second + x.first.size > y.second + y.first.size;
              return x.first.first_idx < y.first.first_idx;
            });
  for (auto iter : tensor_usages_vec) {
    int unique_id = iter.first.unique_id;
    size_t size = iter.first.size;
    char *addr = tensor_ptr.find(unique_id)->second;
#ifdef MEM_DEBUG
    printf(
        "idx: %d, life cycle : [%d, %d], name: \"%s\", memory size: %.2f MB, "
        "end "
        "memory: %.2f MB\n"
        "offset: %zu, size: %zu, end_offset: %zu, address: %p, end_addr: "
        "%p\n\n",
        unique_id, iter.first.first_idx, iter.first.last_idx,
        iter.first._name.c_str(), float(size) / MB_SIZE,
        float(iter.second + size) / MB_SIZE, iter.second, size,
        iter.second + size, addr, addr + size);
#endif
  }

  for (auto iter : tensor_usages_vec) {
    for (auto check_iter : temp_usages_vec) {
      if (judge_func(check_iter, iter)) {
        continue;
      }
      int unique_id = iter.first.unique_id;
      size_t size = iter.first.size;

      // Logically, this part of the processing will never be executed. If it is
      // executed, it means that there is a bug in the shared memory scheduling
      // algorithm.

      printf("================================\n");
      printf("ERROR occurred!\n");
      printf(
          "idx: %d, life cycle : [%d, %d], name: \"%s\", size: %zu, offset: "
          "%zu\n",
          unique_id, iter.first.first_idx, iter.first.last_idx,
          iter.first._name.c_str(), size, iter.second);

      int check_unique_id = check_iter.first.unique_id;
      size_t check_size = check_iter.first.size;
      printf(
          "idx: %d, life cycle : [%d, %d], name: \"%s\", size: %zu, offset: "
          "%zu\n",
          check_unique_id, check_iter.first.first_idx,
          check_iter.first.last_idx, check_iter.first._name.c_str(), check_size,
          check_iter.second);

      printf("================================\n");
      exit(-1);
    }
    temp_usages_vec.push_back(iter);
  }

  printf("\n========== Finish MemoryManager calculate_buffer_ ==========\n\n");
}

}  // namespace lightseq
