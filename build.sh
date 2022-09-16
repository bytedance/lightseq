if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DNEW_ARCH=ON .. && make -j${nproc}
