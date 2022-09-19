if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DUSE_NEW_ARCH=ON -DUSE_TRITONBACKEND=OFF .. && make -j${nproc}
