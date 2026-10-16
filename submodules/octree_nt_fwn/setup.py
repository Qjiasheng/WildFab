from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(ROOT_DIR, "includes")]

setup(
    name="octree_nt_clouds_wn",
    packages=['octree_nt_clouds_wn'],
    ext_modules=[ #  extension modules to be compiled from C/C++ or CUDA source files.
        CUDAExtension(  # means compiling a CUDA extension module
            name="octree_nt_clouds_wn._C", # with different kernel radii
            sources=[
                "octree.cu",
                "fast_wn.cu",
                "ext.cpp", ],
            include_dirs=include_dirs,
            extra_compile_args={"nvcc": ["-O3", "-use_fast_math", "-Xcompiler", "-fno-gnu-unique"]})
        ],
    cmdclass={
        'build_ext': BuildExtension # to build this
    }
)