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

* Element-wise Arithmetic Operations between 2 `Array`s.
  * `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`
* Arithmetic Operations between `Array` and `float`.
  * `+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`
* Arithmetic Operations between `float` and `Array`.
  * `+`, `-`, `*`, `/`
* Matrix Multiplication between 1d/2d `Array`s.
  * `@`
