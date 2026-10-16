from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='contour_connect', packages=['contour_connect'], ext_modules=[CUDAExtension(name='contour_connect._C', sources=['ext.cpp', 'connect.cu', 'graph.cpp'], extra_compile_args={'nvcc': ['-O3']})], cmdclass={'build_ext': BuildExtension})
