if [ ! -d 'build' ]; then
    mkdir build
fi
# DEVICE_ARCH could be cuda/x86/arm
cd build && cmake -DUSE_NEW_ARCH=ON -DDEVICE_ARCH=x86 -DUSE_TRITONBACKEND=OFF -DDEBUG_MODE=OFF -DFP16_MODE=OFF -DMEM_DEBUG=OFF .. && make -j${nproc}
# you can use comand like below to compile lightseq with pybind interface:
# sudo PATH=$PATH:/usr/local/hdf5 CUDACXX=/usr/local/cuda/bin/nvcc  DEVICE_ARCH=x86 ENABLE_FP32=1 ENABLE_DEBUG=0 ENABLE_NEW_ARCH=1 python3 setup.py install
