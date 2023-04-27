#pragma once
#include "declaration.h"
#include "node.h"
#include "cmath"

namespace lightseq {

template<typename T1, typename T2>
class RataryPositionQK: public Operator {
private:
    // std::vector<float> sin_pos;
    // std::vector<float> cos_pos;
    float* _sin_ptr;
    float* _cos_ptr;
    size_t _max_step;
    size_t _max_batch_size;
    size_t _batch_size;
    size_t _head_num;
    size_t _head_dim;
    size_t _offset_seq_len;
    size_t _query_len;

    float* _device_sin_ptr;
    float* _device_cos_ptr;

public:
    RataryPositionQK(int max_step, int head_num, int head_dim): Operator("RataryPositionQK"), _max_step(max_step), _head_num(head_num), _head_dim(head_dim) {
        if (head_dim & 1) {
            printf("Error! head dim should be even number while using RataryPositionQK Operator.\n");
            exit(0);
        }
        
        int total_size = max_step * head_dim / 2;
        _sin_ptr = (float*)malloc(total_size * sizeof(float));
        _cos_ptr = (float*)malloc(total_size * sizeof(float));
        
        for(int i = 0; i < head_dim / 2; i ++) {
            float theta = std::pow(10000, -2. * i / head_dim);
            for(int j = 0; j < max_step; j ++) {
                *(_sin_ptr + j * head_dim / 2 + i) = sin(j * theta); // shape: [max_step, head_dim / 2]
                *(_cos_ptr + j * head_dim / 2 + i) = cos(j * theta); // shape: [max_step, head_dim / 2]
            }
        }

#ifdef LIGHTSEQ_cuda
        _device_sin_ptr = _context_ptr->allocator_ptr->malloc_mem(total_size * sizeof(float));
        _device_cos_ptr = _context_ptr->allocator_ptr->malloc_mem(total_size * sizeof(float));
        CHECK_GPU_ERROR(cudaMemcpy(_device_sin_ptr, _sin_ptr, total_size * sizeof(float)));
        CHECK_GPU_ERROR(cudaMemcpy(_device_cos_ptr, _cos_ptr, total_size * sizeof(float)));
#else 

#endif 
    }

    void before_forward(int batch_size, int offset_seq_len, int query_len) {
        _batch_size = batch_size;
        _offset_seq_len = offset_seq_len;
        _query_len = query_len;
    }

    Variable* operator() (Variable* inp_tensor);

    T1* forward(T1* input_ptr);
};

}