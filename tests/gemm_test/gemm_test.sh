python3 gemm_test.py -hd 1024 -id 4096 -minb 1 -maxb 10000 -d configs
python3 gemm_test.py -hd 512 -id 2048 -minb 1 -maxb 10000 -d configs
python3 gemm_test.py -hd 768 -id 3072 -minb 1 -maxb 10000 -d configs

if [ ! -d "$HOME/.lightseq/igemm_configs" ];then
  mkdir $HOME/.lightseq/igemm_configs
fi

cp configs/* $HOME/.lightseq/igemm_configs
