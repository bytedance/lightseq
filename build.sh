if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DUSE_NEW_ARCH=ON -DUSE_TRITONBACKEND=OFF -DDEBUG_MODE=ON .. && make -j${nproc}
