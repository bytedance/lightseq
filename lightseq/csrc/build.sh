if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DDEBUG_MODE=OFF .. && make -j${nproc}
