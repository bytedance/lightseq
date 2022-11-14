if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DUSE_NEW_ARCH=ON -DUSE_TRITONBACKEND=ON -DDEBUG_MODE=OFF .. && make -j${nproc}
