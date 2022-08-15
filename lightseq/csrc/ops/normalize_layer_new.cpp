#include "normalize_layer_new.h"

namespace lightseq {
  
template <typename T1, typename T2>
NormalizeLayerOp<T1, T2>::NormalizeLayerOp(uint32_t max_batch_tokens, uint32_t hidden_dim, bool use_mean):
    _max_batch_tokens(max_batch_tokens),
    _hidden_dim(hidden_dim),
    Operator("NormalizeLayerOp") {
#ifdef ONLY_OP

    //printf("Running Step.2.1\n");
    static_vars_ = cuda_malloc<T1>(max_batch_tokens);
    if (use_mean) {
        static_means_ = cuda_malloc<T1>(max_batch_tokens);
    }
    //printf("Running Step.2.2\n");
#else
    //printf("Running Step.2.3\n");
    vars_.reset(new Tensor(_name + "/vars", max_batch_tokens * sizeof(T1)));
    if (use_mean) 
        means_.reset(new Tensor(_name + "/means", max_batch_tokens * sizeof(T1)));
    //printf("Running Step.2.4\n");
#endif

}


template <typename T1, typename T2>
NormalizeLayerOp<T1, T2>::~NormalizeLayerOp() {
#ifdef ONLY_OP
    cuda_free(static_means_);
    cuda_free(static_vars_);
#endif
}

template <typename T1, typename T2>
Variable* NormalizeLayerOp<T1, T2>::operator()(Variable* inp, Variable* gamma, Variable* betta) {
    size_t max_size = _max_batch_tokens * _hidden_dim;
    Variable* result = new Variable(
        this->_name + "-out", max_size * sizeof(T1), max_size * sizeof(T2));
    this->set_parents({inp, gamma, betta});
    this->set_children({result});
    return result;
}

template <typename T1, typename T2>
void NormalizeLayerOp<T1, T2>::before_forward(size_t batch_tokens) { 
    _batch_tokens = batch_tokens; 
    _max_batch_dim = _batch_tokens * _hidden_dim;
}

template <typename T1, typename T2>
void NormalizeLayerOp<T1, T2>::forward() {
    T1* ln_res_val = (T1*)child(0)->value();
    T1* inp_val = (T1*)parent(0)->value();
    T1* gamma_val = (T1*)parent(1)->value();
    T1* betta_val = (T1*)parent(2)->value();

    cudaStream_t stream = _context_ptr->get_stream();

#ifdef ONLY_OP
    T1* vars_val = static_vars_;
    T1* means_val = static_means_;
#else 
    T1* vars_val = vars_->tensor();
    T1* means_val = means_->tensor();
#endif
    //printf("Running Step.3.3\n");
    std::cout << ln_res_val << std::endl;
    std::cout << vars_val << std::endl;
    std::cout << means_val << std::endl;
    std::cout << inp_val << std::endl;
    std::cout << gamma_val << std::endl;
    std::cout << betta_val << std::endl;

    launch_layer_norm(ln_res_val, vars_val, means_val, inp_val, gamma_val, betta_val, _batch_tokens,
                    _hidden_dim, stream);
}

template <typename T1, typename T2>
void NormalizeLayerOp<T1, T2>::before_backward(size_t batch_tokens) { 
    _batch_tokens = batch_tokens; 
    _max_batch_dim = _batch_tokens * _hidden_dim;
}

template <typename T1, typename T2>
void NormalizeLayerOp<T1, T2>::backward() {

    T2* gamma_grad = (T2*)parent(1)->grad();
    T2* betta_grad = (T2*)parent(2)->grad();
    T2* inp_grad = (T2*)parent(0)->grad();
    T2* out_grad = (T2*)child(0)->grad();
    T2* residual_grad = nullptr;

    T1* out_val = (T1*)child(0)->value();
    T1* gamma_val = (T1*)parent(1)->value();
    T1* betta_val = (T1*)parent(2)->value();

    cudaStream_t streams[2] = {_context_ptr->get_stream(), _context_ptr->get_stream()};

#ifdef ONLY_OP
    T1* vars_val = static_vars_;
    T1* means_val = static_means_;
#else 
    T1* vars_val = (T1*)vars_->value();
    T1* means_val = (T1*)means_->value();
#endif 

    launch_ln_bw(gamma_grad, betta_grad, inp_grad, out_grad, residual_grad,
            out_val, gamma_val, betta_val, vars_val, means_val, _batch_tokens,
            _hidden_dim, streams);
}


template class NormalizeLayerOp<__half, __half>;
template class NormalizeLayerOp<float, float>;

} // namespace lightseq