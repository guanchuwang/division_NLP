from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='actnn',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'minimax',
              ['minimax.cc', 'minimax_cuda_kernel.cu'],
          extra_compile_args = {'nvcc': ['--expt-extended-lambda']}
),
          cpp_extension.CUDAExtension(
              'quantization',
              ['quantization.cc', 'quantization_cuda_kernel.cu'],
              extra_compile_args={'nvcc': ['--expt-extended-lambda']}
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
