nvcc -c gemm.cu -o gemm.cuda.o
nvcc gemm.cuda.o test.cpp -o test -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lcublasLt
