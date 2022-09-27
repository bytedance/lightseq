python3 gemm_test.py -hd 1024 -id 4096 -minb 1 -maxb 10000 -d configs
python3 gemm_test.py -hd 512 -id 2048 -minb 1 -maxb 10000 -d configs
python3 gemm_test.py -hd 768 -id 3072 -minb 1 -maxb 10000 -d configs
cp configs/* /tmp/igemm_configs