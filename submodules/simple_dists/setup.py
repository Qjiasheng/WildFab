from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="simple_dists",  # name of the package
    packages=['simple_dists'],
    ext_modules=[ #  extension modules to be compiled from C/C++ or CUDA source files.
        CUDAExtension(  # means compiling a CUDA extension module
            name="simple_dists._C", # with different kernel radii
            sources=[
                "dists.cu",
                "ext.cpp", ],
            extra_compile_args={"nvcc": ["-O3", "-use_fast_math", "-Xcompiler", "-fno-gnu-unique", "--expt-relaxed-constexpr",
             "-DTILE_CLOUD=128"]})
        ],
    cmdclass={
        'build_ext': BuildExtension # to build this
    }
)