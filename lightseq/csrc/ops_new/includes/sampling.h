#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template<typename T>
class SamplingOp: public Operator {
private:
    GenerateMethod _generate_method;
    int _max_batch_size;
    int _max_step;
    int _max_thread_per_block;
    int _trg_vocab_size;
    int _topk;
    float _topp;
    int _eos_id;
    bool _has_logits_bias;
    int* _p_d_unfinished;

    int _batch_size;
    int _seq_len;
    int _logits_seq_len;

#ifdef LIGHTSEQ_cuda
    curandState *_p_d_curandstate;  //[batch_size]
#endif

    Variable* _out_token_ids;

public:
    SamplingOp(GenerateMethod gm, int max_batch_size, int max_step, int max_thread_per_block, 
    int trg_vocab_size, int topk, float topp, int eos_id);

    // output: new_token_ids
    Variable* operator() (Variable* logits, Variable* logit_bias, Variable* token_ids);

    void before_forward(int batch_size, int seq_len, int logits_seq_len) {
        _batch_size = batch_size;
        _seq_len = seq_len;
        _logits_seq_len = logits_seq_len;
    }

    void forward() override;

    void backward() override {}
};

} // namespace lightseq 