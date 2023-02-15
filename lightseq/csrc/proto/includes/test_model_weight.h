#pragma once
#include "bert.pb.h"
#include "proto_headers.h"
#include "proto_util.h"

namespace lightseq {
template <typename T>
class TestModelWeight {
 private:
  const T* _p_d_weight_emb;
  std::vector<T> _d_weight_emb;

 public:
  TestModelWeight(int weight_size) {
    _d_weight_emb.clear();
    for (int i = 0; i < weight_size; i++) {
      _d_weight_emb.push_back(rand() % 100);
    }
  }
  const T*& weight_emb() const { return _p_d_weight_emb; }
};
}  // namespace lightseq
