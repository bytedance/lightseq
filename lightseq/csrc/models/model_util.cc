#include "model_util.h"

namespace lightseq {

GenerateMethod get_generate_method(std::string method_) {
  if (method_ == "topk") return GenerateMethod::Topk;
  if (method_ == "topp") return GenerateMethod::Topp;
  if (method_ == "beam_search") return GenerateMethod::BeamSearch;

  printf("Error!\n");
  return GenerateMethod::UnDefined;
}

}  // namespace lightseq
