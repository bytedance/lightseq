#include "test_model_weight.h"
#include "linear_layer.h"

namespace lightseq {

void test_func() {
  Context::create_global_context(StatusType::Inference);
  std::shared_ptr<Context> _context_ptr = Context::global_instance();

  const int batch_size = 1;
  const int seq_len = 2;
  const int batch_tokens = batch_size * seq_len;
  const int input_size = 2;
  const int output_size = 2;

  // create weight matrix
  std::vector<float> wei_emb;
  int weight_size = input_size * output_size;
  for (int i = 0; i < weight_size; i++) {
    wei_emb.push_back(rand() % 100);
  }
  std::vector<const float*> _p_wei_emb;
  _p_wei_emb.push_back(wei_emb.data());

  // create linear layer & load params
  LinearLayerPtr<float, float> linear_layer(
      new LinearLayer<float, float>(batch_tokens, input_size, output_size));
  linear_layer->load_params(_p_wei_emb, 0);

  // construct network
  Variable* inp(new Variable("input"));
  Variable* out = (*linear_layer)(inp);

  // set input_ptr & output_ptr
  float* input_ptr = (float*)malloc(batch_tokens * input_size * sizeof(float));
  for (int i = 0; i < batch_tokens * input_size; i++) {
    *(input_ptr + i) = rand() % 100;
  }
  inp->set_value((char*)input_ptr);
  float* output_ptr =
      (float*)malloc(batch_tokens * output_size * sizeof(float));
  out->set_value((char*)output_ptr);

  // calculate
  linear_layer->before_forward(batch_size, seq_len);
  linear_layer->forward();

  //
  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < input_size; j++) {
      printf("%f, ", input_ptr[i * input_size + j]);
    }
    printf("\n");
  }

  printf("==========\n");

  for (int i = 0; i < input_size; i++) {
    for (int j = 0; j < batch_tokens; j++) {
      printf("%f, ", wei_emb[i * batch_tokens + j]);
    }
    printf("\n");
  }

  printf("==========\n");

  for (int i = 0; i < output_size; i++) {
    for (int j = 0; j < batch_tokens; j++) {
      printf("%f, ", output_ptr[i * batch_tokens + j]);
    }
    printf("\n");
  }

  printf("==========\n");
}
}  // namespace lightseq

int main() { lightseq::test_func(); }
