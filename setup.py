import os
import platform
from setuptools import setup, find_packages, Extension
import subprocess

import pybind11

pkg = "vulkpy"

# Compile Compute Shader
for shader in [
        "add", "sub", "mul", "div",
        "iadd", "isub", "imul", "idiv",
        "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
        "iadd_scalar", "isub_scalar", "imul_scalar", "idiv_scalar",
        "rsub_scalar", "rdiv_scalar",
        "add_broadcast", "sub_broadcast", "mul_broadcast", "div_broadcast",
        "iadd_broadcast", "isub_broadcast", "imul_broadcast", "idiv_broadcast",
        "matmul",
        "max", "min", "imax", "imin",
        "max_scalar", "min_scalar", "imax_scalar", "imin_scalar",
        "max_broadcast", "min_broadcast", "imax_broadcast", "imin_broadcast",
        "abs", "sign", "iabs", "isign",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "isin", "icos", "itan", "iasin", "iacos", "iatan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "isinh", "icosh", "itanh", "iasinh", "iacosh", "iatanh",
        "exp", "log", "exp2", "log2",
        "iexp", "ilog", "iexp2", "ilog2",
        "sqrt", "invsqrt", "isqrt", "iinvsqrt",
        "pow", "ipow", "pow_scalar", "ipow_scalar", "rpow_scalar",
        "pow_broadcast", "ipow_broadcast",
        "clamp", "clamp_sv", "clamp_vs", "clamp_ss",
        "iclamp", "iclamp_sv", "iclamp_vs", "iclamp_ss",
        "prng_xoshiro128pp_uint32", "prng_xoshiro128pp_float",
        "prng_box_muller", "prng_ibox_muller",
        "prng_randrange",
        "sum", ("sum_v1.3", "--target-env=vulkan1.1"), "sum_axis",
        "prod", ("prod_v1.3", "--target-env=vulkan1.1"), "prod_axis",
        "sum_axis_rebroadcast", "prod_axis_rebroadcast",
        "maximum", ("maximum_v1.3", "--target-env=vulkan1.1"), "maximum_axis",
        "minimum", ("minimum_v1.3", "--target-env=vulkan1.1"), "minimum_axis",
        "maximum_axis_rebroadcast", "minimum_axis_rebroadcast",
        "broadcast",
        "batch_affine",
        "gather", "gather_axis",
        "nn_cross_entropy", "nn_cross_entropy_backward",
]:
    if isinstance(shader, tuple):
        shader, flag = shader
        flag = (flag,)
    else:
        shader = shader
        flag = tuple()
    s = os.path.join(pkg, "shader", shader)
    spv = s+".spv"
    comp = s+".comp"

    if ((not os.path.exists(spv)) or
        (os.path.exists(comp) and (os.stat(comp).st_mtime > os.stat(spv).st_mtime))):
        cmd = subprocess.run(["glslc", *flag, "-o", spv, comp],
                             capture_output=True, text=True)
        if cmd.stdout:
            print(cmd.stdout)
        if cmd.stderr:
            print(cmd.stderr)
        cmd.check_returncode()


if platform.system() != "Windows":
    extra_args = {
        "extra_compile_args": ["-std=c++2a", "-O3", "-march=native", "-Wall"],
        "extra_link_args": ["-std=c++2a"],
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

desc = {}
README = "README.md"
if os.path.exists(README):
    with open(README) as f:
        desc["long_description"] = f.read()
        desc["long_description_content_type"] = "text/markdown"

setup(name="vulkpy",
      version="0.0.3",
      author="H. Yamada",
      description="GPGPU array on Vulkan",
      **desc,
      url="https://github.com/ymd-h/vulkpy",
      packages=find_packages(),
      ext_modules=ext,
      include_package_data=True,
      install_requires=["numpy", "well-behaved-logging"],
      extras_require={
          "test": ["coverage", "unittest-xml-reporting"],
          "doc": ["sphinx", "sphinx-rtd-theme", "myst-parser"],
      },
      classifiers=[
          "Development Status :: 4 - Beta",
          "Environment :: GPU",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3 :: Only",
          "Programming Language :: Python :: Implementation :: CPython",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ])
