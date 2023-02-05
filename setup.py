import os
import platform
from setuptools import setup, find_packages, Extension
import subprocess

import pybind11

pkg = "vulkpy"

# Compile Compute Shader
for shader in ["add", "sub", "mul", "div",
               "iadd", "isub", "imul", "idiv",
               "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
               "iadd_scalar", "isub_scalar", "imul_scalar", "idiv_scalar",
               "rsub_scalar", "rdiv_scalar"]:
    s = os.path.join(pkg, "shader", shader)
    spv = s+".spv"
    comp = s+".comp"

    if ((not os.path.exists(spv)) or
        (os.path.exists(comp) and (os.stat(comp).st_mtime > os.stat(spv).st_mtime))):
        cmd = subprocess.run(["glslc", "-o", spv, comp],
                             capture_output=True, text=True)
        if cmd.stdout:
            print(cmd.stdout)
        if cmd.stderr:
            print(cmd.stderr)
        cmd.check_returncode()


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
      author="H. Yamada",
      description="GPGPU array on Vulkan",
      packages=find_packages(),
      ext_modules=ext,
      package_data={f"{pkg}.shader": [os.path.join(f"{pkg}", "shader", "*.spv")]},
      install_requires=["numpy"])
