# Broadcasting

vulkpy obeys [NumPy broadcasting rule](https://numpy.org/doc/stable/user/basics.broadcasting.html).


We implement 3 patterns of broadcasting implementations.


## 1. Simple Copy
The simplest broadcasting is to create new broadcasted array.
Usually broadcasting is executed just before other operations,
so that this implementation might allocate unnecessary temporary array.
This is memory- and computationally-inefficient,
but it works fine in most cases.
Acutually, we still use this in `clamp()` method.

Users can execute this broadcasting by `broadcast_to(shape)` method.

```python
import vulkpy as vk

gpu = vk.GPU()
a = vk.Array(gpu, data=[1, 2])

b = a.broadcast_to((2, 2))
# => [[1, 2], [1, 2]]
```

````{note}
In Vulkan compute shader, we can use only 3 global indices at most.
Thay are not sufficient to point elements of
`N`-dimensional array directly.
Instead, we utilize linearly flattened index
and calculate the position from it on GPU.
We assume this index calculation is computetionally-inefficient.

The following is a partial code of `broadcast.comp`.

```glsl
void main(){
  uint i = gl_GlobalInvocationID.x;
  if(i >= params.size[1]){ return; }

  uint i_tmp = i;
  uint j = 0;
  uint sizeA = params.size[0];
  uint sizeB = params.size[1];
  for(uint dim = 0; dim < params.ndim; dim++){
    sizeA = sizeA / a_shape[dim];
    sizeB = sizeB / b_shape[dim];

    uint d = min(i_tmp / sizeB, a_shape[dim]-1);
    j += d * sizeA;

    i_tmp = i_tmp % sizeB;
  }

  b[i] = a[j];
}
```
````

## 2. Special Implementation
We also provide special implementations for some operations.
For example, a compute shader `add_broadcast.comp` implements
a fused operation of broadcasting and addition.
Although we still need index calculation, we can omit temporary memory allocation.

For these special implementations, users don't need to call explicitly,
if operations are supported, such special implementations are used automatically.

```python
import vulkpy as vk

gpu = vk.GPU()
a = vk.Array(gpu, data=[1, 2])
b = vk.Array(gpu, data=[[1, 2], [3, 4]])

c = a + b
# => [[2, 4], [4, 6]]
```

```{note}
For inplace operations, only ~other~ (non-inplaced) array
can be broadcasted because we cannot grow already allocated memory.

Since we can skip index computation for inplaced array,
inplace broadcasting is more efficient in terms of not only memory
but also computation.
```


## 3. Re-broadcasting of Reduction

For specific usecase like softmax, broadcasting is executed just after reduction.
We define such usecase as re-broadcasting.

In re-broadcasting, inefficient index calculation is not necessary, so
that it is more efficient in terms of computation.

Users can pass `rebroadcast=True` to reduction methods;

```python
import vulkpy as vk

gpu = vk.GPU()
a = vk.Array(gpu, data=[[1, 2, 3], [4, 5, 6]])

b = a.mean(axis=0, rebroadcast=True)
# => [[2.5, 3.5, 4.5], [2.5, 3.5, 4.5]]
```
