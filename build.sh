if [ ! -d 'build' ]; then
    mkdir build
fi
# DEVICE_ARCH could be cuda/x86/arm
cd build && cmake -DUSE_NEW_ARCH=ON -DDEVICE_ARCH=cuda -DUSE_TRITONBACKEND=OFF -DDEBUG_MODE=ON -DFP16_MODE=ON -DMEM_DEBUG=OFF .. && make -j${nproc}
# you can use comand like below to compile lightseq with pybind interface:
# sudo PATH=$PATH:/usr/local/hdf5 CUDACXX=/usr/local/cuda/bin/nvcc  DEVICE_ARCH=cuda ENABLE_FP32=0 ENABLE_DEBUG=0 ENABLE_NEW_ARCH=1 python3 setup.py install
