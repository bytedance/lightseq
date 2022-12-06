if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DUSE_NEW_ARCH=OFF -DUSE_TRITONBACKEND=ON -DDEBUG_MODE=OFF -DFP16_MODE=ON -DMEM_DEBUG=OFF .. && make -j${nproc}
