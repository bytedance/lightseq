if [ ! -d 'build' ]; then
    mkdir build
fi
# DEVICE_ARCH could be cuda/x86/arm
cd build && cmake -DUSE_NEW_ARCH=ON -DDEVICE_ARCH=arm -DUSE_TRITONBACKEND=ON -DDEBUG_MODE=OFF -DFP16_MODE=OFF -DMEM_DEBUG=OFF .. && make -j${nproc}
