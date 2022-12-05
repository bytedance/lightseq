import os
import re
import sys
import platform
import subprocess
import multiprocessing
import glob
import logging
from setuptools import setup, Extension
import setuptools
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

from lightseq import __version__
import torch
import pathlib

logging.basicConfig()
logger = logging.getLogger(__file__)

ENABLE_FP32 = int(os.environ.get("ENABLE_FP32", 0))
ENABLE_DEBUG = int(os.environ.get("ENABLE_DEBUG", 0))


from lightseq.training.ops.pytorch.builder import ALL_OPS, OpBuilder

torch_available = True
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
except ImportError:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )

is_rocm_pytorch = OpBuilder.is_rocm_pytorch()
print("is_rocm_pytorch: ", is_rocm_pytorch)

if torch.utils.cpp_extension.CUDA_HOME is None and (not is_rocm_pytorch):
    raise RuntimeError(
        "--cuda_ext was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc."
    )

cmdclass = {}
ext_modules = []
define_macros = []
install_requires = []
compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", *args, **kwargs):
        Extension.__init__(self, name, sources=[], *args, **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            print("ext: ", ext)
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            # fixed for lightseq.inference
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.join(extdir, "lightseq"),
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            cmake_args += ["-DFP16_MODE=OFF"] if ENABLE_FP32 else ["-DFP16_MODE=ON"]
            cmake_args += ["-DDEBUG_MODE=ON"] if ENABLE_DEBUG else ["-DDEBUG_MODE=OFF"]
            cmake_args += ["-DDYNAMIC_API=OFF"]
            build_args += ["--target", "lightseq"]
            build_args += ["--", "-j{}".format(multiprocessing.cpu_count())]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


for op_name, builder in ALL_OPS.items():
    print(f"Install Ops={op_name}")
    op_compatible = builder.is_compatible()
    print(f"op_name: {op_name}")

    if op_compatible:
        reqs = builder.python_requirements()
        install_requires += builder.python_requirements()

        assert (
            torch_available
        ), f"Unable to pre-compile {op_name}, please first install torch"
        if is_rocm_pytorch:
            define_macros += [("WITH_HIP", None)]
            ext_modules.append(builder.builder())
            cmd_class = {"build_ext": BuildExtension.with_options(use_ninja=False)}
        else:
            cmd_class = {build_ext: CMakeBuild}
            ext_modules = [CMakeExtension("inference")]

with open("README.md", "r") as fh:
    long_description = fh.read()


setup_kwargs = dict(
    name="lightseq",
    version=__version__,
    author="Xiaohui Wang, Ying Xiong, Xian Qian, Yang Wei",
    author_email=(
        "wangxiaohui.neo@bytedance.com, xiongying.taka@bytedance.com"
        ", qian.xian@bytedance.com, weiyang.god@bytedance.com"
    ),
    description=(
        "LightSeq is a high performance library for sequence processing and generation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bytedance/lightseq",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=install_requires,
    python_requires=">=3.6",
    cmdclass=cmd_class,
    zip_safe=False,
    packages=setuptools.find_packages(exclude=["docs", "tests"]) + ["."],
    entry_points={
        "console_scripts": [
            "lightseq-train = lightseq.training.cli."
            "lightseq_fairseq_train_cli:ls_cli_main",
            "lightseq-generate = lightseq.training.cli."
            "lightseq_fairseq_generate_cli:ls_cli_main",
            "lightseq-validate = lightseq.training.cli."
            "lightseq_fairseq_validate_cli:ls_cli_main",
            "lightseq-deepspeed = lightseq.training.cli."
            "lightseq_deepspeed_cli:ls_cli_main",
        ],
    },
)

try:
    setup(ext_modules=ext_modules, **setup_kwargs)
except Exception as e:
    logger.warning(e)
    logger.warning("The extension could not be compiled")

    # If this new 'setup' call don't fail, the module
    # will be successfully installed, without the C extension :
    setup(**setup_kwargs)
    logger.info("lightseq extension installation succeeded.")
