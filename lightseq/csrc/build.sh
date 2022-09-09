if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DDEBUG_TYPE=FP16 .. && make -j${nproc}
