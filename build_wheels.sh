#!bash
set -e -u -x

function repair_wheel() {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat manylinux2010_x86_64 -w $(dirname $(readlink -e $wheel))
    fi
}

PROJECT_DIR=$(dirname $(readlink -e $0))
cd $PROJECT_DIR

echo $LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH}

# Compile wheels
for PYBIN in /opt/python/*/bin/python; do
    "${PYBIN}" -m pip install -U build
    ENABLE_FP32=0 ENABLE_DEBUG=0 "${PYBIN}" -m build
done

# Bundle external shared libraries into the wheels
# mkdir -p $PROJECT_DIR/dist/manylinux
for whl in $PROJECT_DIR/dist/*.whl; do
    repair_wheel "$whl"
done
