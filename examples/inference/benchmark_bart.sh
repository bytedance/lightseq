#!/bin/bash

SCRIPT=$(realpath "$0")
CUR_DIR=$(dirname "$SCRIPT")

model_full_name=facebook/bart-base
model_name=$(echo $model_full_name | cut -d "/" -f 2)
all_log=$CUR_DIR/${model_name}_bench.log
res_log=$CUR_DIR/${model_name}_bench.txt
if [ -f $res_log ]; then
    rm $res_log
fi
if [ -f $all_log ]; then
    rm $all_log
fi
echo "batch_size input_seq_len output_seq_len beam_size latency" >>$res_log

for batch_size in 1 8 32; do
    for beam_size in 1 4 32; do
        for input_seq_len in 8 16 32 64; do
            output_seq_len=$input_seq_len
            cd $CUR_DIR/python

            python3 generate_model.py --model_name $model_full_name --sampling_method beam_search \
                --beam_size $beam_size --input_seq_len $input_seq_len --output_seq_len=$output_seq_len
            model_path=$(realpath lightseq_${model_name}_bench.hdf5)

            cd $CUR_DIR/../../build
            ./examples/inference/cpp/transformer_example \
                $model_path $batch_size $input_seq_len |& tee temp.log

            cat temp.log >>$all_log
            latency=$(tail -n 5 temp.log | head -n 1 | awk '{print $4}')
            echo "$batch_size $input_seq_len $output_seq_len $beam_size $latency" >>$res_log
            rm temp.log
        done
    done
done
pip3 install tabulate
tabulate --header $res_log
