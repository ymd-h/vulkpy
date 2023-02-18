# vulkpy: GPGPU array on Vulkan

## Requirements

* C++20 compatible compiler
* `libvulkan`
* Vulkan SDK
  * Headers (`vulkan/vulkan.hpp` and so on)
  * Shaderc (`glslc`)


On Ubuntu 22.0,
```shell
wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
apt update
apt install -y libvulkan1 libvulkan-dev vulkan-headers shaderc vulkan-validationlayers
```

> **Note**  
> `vulkan-sdk` cannot be installed because it requires obsolete package `qt5-default`.


## Example

```python
import vulkpy as vk

gpu = vk.GPU()

a = vk.Array(gpu, data=[10, 10, 10])
b = vk.Array(gpu, data=[5, 5, 5])

c = a + b
c.wait()

print(c)
# [15, 15, 15]
```

## Features

* Element-wise Arithmetic Operators between 2 `Array`s.
  * [x] `+`, `-`, `*`, `/`, `**`, `+=`, `-=`, `*=`, `/=`, `**=`
* Arithmetic Operators between `Array` and `float`.
  * [x] `+`, `-`, `*`, `/`, `**`, `+=`, `-=`, `*=`, `/=`, `**=`
* Arithmetic Operators between `float` and `Array`.
  * [x] `+`, `-`, `*`, `/`, `**`
* Matrix Multiplication Operator between 1d/2d `Array`s.
  * [x] `@`
* Element-wise math functions as `Array`'s member function
  * [x] `max(other, inplace=False)`, `min(other, inplace=False)`
  * [x] `abs(inplace=False)`, `sign(inplace=False)`
  * [x] `sin(inplace=False)`, `cos(inplace=False)`, `tan(inplace=False)`
  * [x] `asin(inplace=False)`, `acos(inplace=False)`, `atan(inplace=False)`
  * [x] `sinh(inplace=False)`, `cosh(inplace=False)`, `tanh(inplace=False)`
  * [x] `asinh(inplace=False)`, `acosh(inplace=False)`, `atanh(inplace=False)`
  * [x] `exp(inplace=False)`, `log(inplace=False)`
  * [x] `exp2(inplace=False)`, `log2(inplace=False)`
  * [x] `sqrt(inplace=False)`, `invsqrt(inplace=False)`
  * [x] `clamp(min, max, inplace=False)`
* Reduction as `Array`'s member function
  * [x] `sum(axis=None)`, `prod(axis=None)`
  * [x] `maximum(axis=None)`, `minimum(axis=None)`
  * [x] `mean(axis=None)`
  * [ ] ...
* General Tensor Product
  * [ ] tensordot
* Bload Cast
  * [ ] ???
* Pseudo Random Number Generator (PRNG)
  * [x] xoshiro128++ (`vulkpy.random.Xoshiro128pp(gpu, *, size=None, data=None)`)
    * `[0, 1)` uniform (`.random(shape=None, buffer=None)`)
    * Gaussian with Box-Muller (`.normal(shape=None, buffer=None, mean=0.0, stddev=1.0)`)
  * [ ] pcg32
* Neural Network
  * [ ] dense, conv, ...
  * [ ] sgd, adam, ...
  * [ ] ONNX support
