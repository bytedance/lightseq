#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" wheel /workspace/ --no-deps -w /workspace/output/wheels
done

# rm -rf /io/wheelhouse

# Bundle external shared libraries into the wheels
# for whl in wheelhouse/*.whl; do
#     repair_wheel "$whl" 
# done
