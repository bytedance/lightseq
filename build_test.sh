nvcc -o gemm gemm.cpp -lcublasLt -lcublas
./gemm 512 1024 1024
