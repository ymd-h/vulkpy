import os
import platform
from setuptools import setup, find_packages, Extension

import pybind11


pkg = "vulkpy"

if platform.system() != "Windows":
    extra_args = {
        "extra_compile_args": ["-std=c++20", "-O3", "-march=native", "-Wall"],
        "extra_link_args": ["-std=c++20"],
    }
else:
    extra_args = {
        "extra_compile_args": ["/std:c++20", "/O2", "/Wall"],
        "extra_link_args": None,
    }

ext = [Extension(f"{pkg}._vkarray",
                 [os.path.join(f"{pkg}", "_vkarray.cc")],
                 include_dirs=[pybind11.get_include()],
                 libraries=["vulkan"],
                 **extra_args)]

setup(name="vulkpy",
      version="0.0.0",
      packages=find_packages(),
      ext_modules=ext,
      package_data={f"{pkg}.shader": [os.path.join(f"{pkg}", "shader", "*.spv")]},
      install_requires=["numpy"])
