#include "llama.h"

namespace lightseq {
namespace cuda {
Llama::Llama(const std::string weight_path, const int max_batch_size)
    : LSModel({"token_ids"}, {"llama_out"}),
      _max_batch_size(max_batch_size) {
  /* --- step.1 initial context --- */
  Context::create_global_context(StatusType::Inference);
  _context_ptr = Context::global_instance();

  /* --- step.2 load model weights into GPU memory --- */
  // saved in custom proto file
  std::string model_weights_path = weight_path;
  std::string res = tw_.initializing(model_weights_path);
  if (!res.empty()) {
    throw std::runtime_error(res);
  }
  printf("*** model max_batch_size: %d ***\n", max_batch_size);
  _generate_method = get_generate_method(tw_._generate_method);
  if (_generate_method != GenerateMethod::BeamSearch) {
    tw_._beam_size = 1;
  }
  tw_.print_model_config();

  /* --- step.3 initial input Variable node --- */
  _inp_tokens = new Variable("inp_tokens", g_dtype<int>());

  /* --- step.4 inital operator & layer --- */
  int max_batch_tokens = tw_._max_step * _max_batch_size;

  _context_ptr->build();
  printf("Finish construct network!\n");
}

Llama::~Llama() {}

void Llama::before_forward(int batch_size, int prompt_len, int steps) {
//   if (steps == 0) {
//     _launch_gpt_emb_layer->before_forward(batch_size, prompt_len, 0);
//     for (auto iter : _gpt_layers_vec) {
//       iter->before_forward(batch_size * tw_._beam_size, prompt_len, 0);
//     }
//     _lyr_norm_layer->before_forward(batch_size * tw_._beam_size, 1);
//     _linear_layer->before_forward(batch_size * tw_._beam_size, 1);
//     _generator_layer->before_forward(batch_size, prompt_len, 0);
//   } else {
//     _launch_gpt_emb_layer->before_forward(batch_size, 1,
//                                           prompt_len + steps - 1);
//     for (auto iter : _gpt_layers_vec) {
//       iter->before_forward(batch_size * tw_._beam_size, 1,
//                            prompt_len + steps - 1);
//     }
//     _lyr_norm_layer->before_forward(batch_size * tw_._beam_size, 1);
//     _linear_layer->before_forward(batch_size * tw_._beam_size, 1);
//     _generator_layer->before_forward(batch_size, prompt_len, steps);
//   }
}

void Llama::Infer() {
  int batch_size = input_shapes_[0][0], prompt_len = input_shapes_[0][1];

  /* --- notice that the order of forward should be the same with network --- */

}

void Llama::set_input_ptr(int index, void *input_ptr) {
  switch (index) {
    case 0:
      _input_ptr = (int *)input_ptr;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

void Llama::set_output_ptr(int index, void *output_ptr) {
  switch (index) {
    case 0:
      _llama_out_ptr = (int *)output_ptr;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

const void *Llama::get_output_ptr(int index) {
  switch (index) {
    case 0:
      return static_cast<void *>(_llama_out_ptr);

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

std::vector<int> Llama::get_input_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}
std::vector<int> Llama::get_output_max_shape(int index) {
  switch (index) {
    case 0:
      return {_max_batch_size, tw_._beam_size, tw_._max_step};

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}

DataType Llama::get_input_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid input index");
      break;
  }
}

DataType Llama::get_output_dtype(int index) {
  switch (index) {
    case 0:
      return DataType::kInt32;
      break;

    default:
      throw std::runtime_error("invalid output index");
      break;
  }
}
}  // namespace cuda
}  // namespace lightseq
