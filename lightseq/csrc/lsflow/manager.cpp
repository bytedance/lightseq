#include "manager.h"

namespace lightseq {
void MemoryManager::update_tensor_life_idx(int unique_id, int node_idx,
                                           size_t size, std::string name) {
  std::map<int, TensorUsage>::iterator iter = tensor_usages_.find(unique_id);
  if (iter == tensor_usages_.end()) {
    tensor_usages_.emplace(
        unique_id, TensorUsage(unique_id, node_idx, node_idx, size, name));
    return;
  }
  if (iter->second.first_idx > node_idx) {
    iter->second.first_idx = node_idx;
  }
  if (iter->second.last_idx < node_idx) {
    iter->second.last_idx = node_idx;
  }
  return;
}

void MemoryManager::remove_life_cycle(int unique_id) {
  if (tensor_usages_.find(unique_id) != tensor_usages_.end()) {
    tensor_usages_.erase(unique_id);
  }
}

void MemoryManager::calculate_buffer_() {
  if (buffer_ != nullptr) {
    cudaFree(buffer_);
  }
  tensor_ptr.clear();
  std::vector<std::pair<TensorUsage, size_t>> tensor_usages_vec{};
  size_t tmp_buffer_size_ = 0;
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
            gap < smallest_gap) {  // 注意对于无符号类型的减法处理
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

  buffer_ = cuda_malloc<char>(total_consumption);
  buffer_size_ = total_consumption;

  for (auto iter : tensor_usages_vec) {
    int unique_id = iter.first.unique_id;
    tensor_ptr.emplace(unique_id, buffer_ + iter.second);
  }

  printf("total_consumption: %d\n", total_consumption);

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
  std::vector<std::pair<TensorUsage, size_t>> temp_check_vec{};
  for (auto iter : tensor_usages_vec) {
    int unique_id = iter.first.unique_id;
    size_t size = iter.first.size;
#if DEBUG == true
    printf("idx: %d, life cycle : [%d, %d], name: %s, size: %zu, offset: %zu\n",
           unique_id, iter.first.first_idx, iter.first.last_idx,
           iter.first._name.c_str(), size, iter.second);
#endif

    for (auto check_iter : temp_check_vec) {
      if (judge_func(check_iter, iter)) {
        continue;
      }
      printf("================================\n");
      printf("ERROR occurred!\n");
      printf(
          "idx: %d, life cycle : [%d, %d], name: %s, size: %zu, offset: %zu\n",
          unique_id, iter.first.first_idx, iter.first.last_idx,
          iter.first._name.c_str(), size, iter.second);

      int check_unique_id = check_iter.first.unique_id;
      size_t check_size = check_iter.first.size;
      printf(
          "idx: %d, life cycle : [%d, %d], name: %s, size: %zu, offset: %zu\n",
          check_unique_id, check_iter.first.first_idx,
          check_iter.first.last_idx, check_iter.first._name.c_str(), check_size,
          check_iter.second);

      printf("================================\n");
      exit(-1);
    }
    temp_check_vec.push_back(iter);
  }
}

}  // namespace lightseq
