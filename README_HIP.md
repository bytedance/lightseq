## Lightseq HIP Quick Start
LightSeq supports CUDA backend and HIP backend. Now LightSeq HIP only support training.
### Build

1. ENV Setting
```shell
export C_INCLUDE_PATH=${ROCM_PATH}/rocblas/include:${ROCM_PATH}/rocrand/include/:${ROCM_PATH}/hiprand/include:${ROCM_PATH}/hip/include/hip:$ROCM_PATH/hip/include/hip/hsa_detail:$ROCM_PATH/hipcub/include/hipcub:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export LD_LIBRART_PATH=${ROCM_PATH}/rocblas/lib/:${ROCM_PATH}/rocrand/lib:$LD_LIBRART_PATH
export LIBRARY_PATH=${ROCM_PATH}/rocblas/lib/:$LIBRARY_PATH
```
2. Compile
```
verbose=1 ENABLE_FP32=1 ENABLE_DEBUG=1 CXX=hipcc CC=hipcc python3 setup.py install bdist_wheel
```
After compiling, whl file is generated under dist, which can be conveniently migrated to other platforms.

### Fast training From Fairseq with LightSeq HIP

You can experience lightning fast training by running following commands,
Firstly install these requirements.

```shell
pip install fairseq sacremoses
```
Users need no modification with python training. It works exactly the same as CUDA environment under the HIP backend. See [README.md](README.md) for more examples.

You can set HIP_VISIBLE_DEVICES to specify which gpu card for training. Example of training transformer with LightSeq HIP is as follows:

```
export HIP_VISIBLE_DEVICES=0
lightseq-train /public/DL_DATA/wmt14_en_de_joined_dict  \
    --task translation \
    --arch ls_transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16
```
