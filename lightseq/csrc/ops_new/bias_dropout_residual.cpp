#include "bias_dropout_residual.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasDropoutResOp<T1, T2>::operator()(Variable* inp, Variable* bias,
                                               Variable* residual) {
  Variable* result =
      new Variable(this->_name + "/out", _max_ele_num * sizeof(T1),
                   _max_ele_num * sizeof(T2));
  this->set_parents({inp, bias, residual});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void BiasDropoutResOp<T1, T2>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();

  // printf("Running! BiasDropoutResOp name: %s\n", this->name().c_str());

  T1* input = (T1*)parent(0)->value();
  T1* bias = (T1*)parent(1)->value();
  T1* residual = (T1*)parent(2)->value();
  T1* output = (T1*)child(0)->value();
  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  launch_ls_dropout_res_bias<T1>(output, input, mask_ptr, bias, residual,
                                 _rows * _cols, _cols, RATIO(), stream);

#ifdef DEBUG
  if (_context_ptr->built()) {
    cudaStreamSynchronize(_context_ptr->get_stream());
    printf("%s forward\n", name().c_str());
    print_vec(residual, this->name() + " residual", 10);
    print_vec(bias, this->name() + " bias", 10);
    print_vec(output, this->name() + " ans", 10);
    printf("\n");
  }
#endif
}

template <typename T1, typename T2>
void BiasDropoutResOp<T1, T2>::backward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T2* input_grad = (T2*)parent(0)->grad();
  T2* bias_grad = (T2*)parent(1)->grad();
  T2* residual_grad = (T2*)parent(2)->grad();

  T2* output_grad = (T2*)child(0)->grad();

  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  bool is_res_cover = parent(2)->is_cover();

  launch_ls_dropout_bias_bwd<T2>(input_grad, bias_grad, output_grad, mask_ptr,
                                 _rows, _cols, RATIO(), stream);

  if (is_res_cover) {  // cover
    CHECK_GPU_ERROR(cudaMemcpyAsync((void*)residual_grad, (void*)output_grad,
               _cols * _rows * sizeof(T2), cudaMemcpyDefault, stream));
  } else {  // accumulate
            // launch_fused_add2 ...
    launch_fused_add2(residual_grad, output_grad, residual_grad, _rows, 1,
                      _cols, stream);
  }


#ifdef DEBUG
  if (_context_ptr->built()) {
    cudaStreamSynchronize(stream);
    printf("%s backward is_res_cover: %d\n", name().c_str(), is_res_cover);
    print_vec(input_grad, this->name() + " input_grad", 10);
    print_vec(output_grad, this->name() + " output_grad", 10);
    print_vec(residual_grad, this->name() + " residual_grad", 10);
    printf("\n");
  }
#endif
}

template class BiasDropoutResOp<float, float>;
template class BiasDropoutResOp<__half, __half>;

}  // namespace lightseq
