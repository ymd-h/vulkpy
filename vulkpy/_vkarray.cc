#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_vkutil.hh"

std::uint32_t findMemIndex(const vk::PhysicalDeviceMemoryProperties& ps,
                           vk::MemoryRequirements req,
                           vk::MemoryPropertyFlags flags){
  for(std::uint32_t i = 0, n = ps.memoryTypeCount; i < n; i++){
    auto pf = ps.memoryTypes[i].propertyFlags;
    auto match = ((pf & flags) == flags);
    if(match && (req.memoryTypeBits & (1 << i))){ return i; }
  }
  throw std::runtime_error("Fail to find MemoryIndex");
}


class GPU;

template<typename T> class Buffer {
private:
  std::shared_ptr<GPU> gpu;
  std::size_t nSize;
  std::uint64_t mSize;
  vk::UniqueBuffer b;
  vk::UniqueDeviceMemory m;
  T* ptr;

public:
  Buffer(std::shared_ptr<GPU> gpu, vk::UniqueDevice& device,
         const vk::PhysicalDeviceMemoryProperties& ps, std::size_t n)
    : gpu(gpu), nSize(n), mSize(sizeof(T) * n)
  {
    auto bInfo = vk::BufferCreateInfo{
      .size=this->mSize,
      .usage=vk::BufferUsageFlagBits::eStorageBuffer
    };
    this->b = device->createBufferUnique(bInfo);

    auto req = device->getBufferMemoryRequirements(this->b.get());

    auto midx = findMemIndex(ps, req,
                             vk::MemoryPropertyFlagBits::eHostVisible |
                             vk::MemoryPropertyFlagBits::eHostCoherent);

    auto alloc = vk::MemoryAllocateInfo{
      .allocationSize=req.size,
      .memoryTypeIndex=midx
    };
    this->m = device->allocateMemoryUnique(alloc);

    device->bindBufferMemory(this->b.get(), this->m.get(), 0);
    this->ptr = static_cast<T*>(device->mapMemory(this->m.get(), 0, this->mSize));
  }

  Buffer(std::shared_ptr<GPU> gpu, vk::UniqueDevice& device,
         const vk::PhysicalDeviceMemoryProperties& ps, const std::vector<T>& data)
    : Buffer<T>(gpu, device, ps, data.size())
  {
    memcpy((void*)this->ptr, (void*)data.data(), this->mSize);
  }

  T get(std::size_t i) const {
    return this->ptr[i];
  }

  T* data() const {
    return this->ptr;
  }

  std::size_t size() const {
    return this->nSize;
  }

  vk::DescriptorBufferInfo info() const {
    return vk::DescriptorBufferInfo{
      .buffer=this->b.get(),
      .offset=0,
      .range=this->mSize
    };
  }

  vk::MappedMemoryRange range() const {
    return vk::MappedMemoryRange{
      .memory=this->m.get(),
      .offset=0,
      .size=this->mSize
    };
  }
};

namespace OpParams {
  struct Empty{};

  struct Vector{
    std::uint32_t size;
  };

  template<typename T>
  struct VectorScalar{
    std::uint32_t size;
    T scalar;
  };

  template<typename T, std::size_t N>
  struct VectorMultiScalar{
    std::uint32_t size;
    T scalar[N];
  };

  template<typename T>
  struct MatMul{
    std::uint32_t rowA;
    std::uint32_t contractSize;
    std::uint32_t columnB;
  };
}

struct DataShape {
  std::uint32_t x, y, z;
};

template <std::uint32_t N, typename Parameters = OpParams::Empty>
class Op {
private:
  std::uint32_t x, y, z;
  std::shared_ptr<GPU> gpu;
  vk::UniqueDescriptorPool pool;
  vk::UniqueShaderModule shader;
  vk::UniqueDescriptorSetLayout dlayout;
  vk::UniquePipelineLayout playout;
  vk::UniquePipeline pipe;
  vk::UniqueDescriptorSet desc;
public:
  Op(std::shared_ptr<GPU> gpu, vk::UniqueDevice& device,
     std::string_view spv, std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1)
    : x(x), y(y), z(z), gpu(gpu)
  {
    auto psize = vk::DescriptorPoolSize{
      .type=vk::DescriptorType::eStorageBuffer,
      .descriptorCount=N
    };
    auto pool = vk::DescriptorPoolCreateInfo{
      .flags=vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets=1,
      .poolSizeCount=1,
      .pPoolSizes=&psize,
    };
    this->pool = device->createDescriptorPoolUnique(pool);

    auto code = util::readCode(spv);

    auto shader = vk::ShaderModuleCreateInfo{
      .codeSize=code.size(),
      .pCode=(const std::uint32_t*)code.data()
    };
    this->shader = device->createShaderModuleUnique(shader);

    auto dbind = util::generate_from_range([](auto i){
      return vk::DescriptorSetLayoutBinding{
        .binding=i,
        .descriptorType=vk::DescriptorType::eStorageBuffer,
        .descriptorCount=1,
        .stageFlags=vk::ShaderStageFlagBits::eCompute
      };
    }, N);

    auto dlayout = vk::DescriptorSetLayoutCreateInfo{
      .bindingCount=N,
      .pBindings=dbind.data()
    };
    this->dlayout = device->createDescriptorSetLayoutUnique(dlayout);

    auto params = vk::PushConstantRange{
      .stageFlags=vk::ShaderStageFlagBits::eCompute,
      .offset=0,
      .size=sizeof(Parameters),
    };

    auto playout = vk::PipelineLayoutCreateInfo{
      .setLayoutCount=1,
      .pSetLayouts=&this->dlayout.get(),
      .pushConstantRangeCount=1,
      .pPushConstantRanges=&params
    };
    this->playout = device->createPipelineLayoutUnique(playout);

    auto ssinfo = vk::PipelineShaderStageCreateInfo{
      .stage=vk::ShaderStageFlagBits::eCompute,
      .module=this->shader.get(),
      .pName="main"
    };
    auto pipe = vk::ComputePipelineCreateInfo{
      .stage=ssinfo,
      .layout=this->playout.get()
    };
    auto p = device->createComputePipelineUnique({}, pipe);

    switch(p.result){
    case vk::Result::eSuccess:
      this->pipe = std::move(p.value);
      break;
    default:
      throw std::runtime_error("Fail: createComputePipeline");
    }

    auto desc = vk::DescriptorSetAllocateInfo{
      .descriptorPool=this->pool.get(),
      .descriptorSetCount=1,
      .pSetLayouts=&this->dlayout.get()
    };
    this->desc = std::move(device->allocateDescriptorSetsUnique(desc)[0]);
  }

  void writeDescriptorSet(vk::UniqueDevice& device,
                          const vk::DescriptorBufferInfo (&info)[N]) const {
    auto w = util::generate_from_range([this, &info](auto i){
      return vk::WriteDescriptorSet{
        .dstSet=this->desc.get(),
        .dstBinding=i,
        .descriptorCount=1,
        .descriptorType=vk::DescriptorType::eStorageBuffer,
        .pBufferInfo=info+i
      };
    }, N);
    device->updateDescriptorSets(w, nullptr);
  }

  std::uint32_t groupCount(std::uint32_t size, std::uint32_t local_size) const {
    return size / local_size + ((size % local_size) != 0);
  }

  vk::SubmitInfo getSubmitInfo(vk::UniqueCommandBuffer& buffer,
                               const DataShape& shape, const Parameters& params) const {
    buffer->begin(vk::CommandBufferBeginInfo{});
    buffer->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipe.get());
    buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                               this->playout.get(),
                               0,
                               this->desc.get(),
                               nullptr);
    buffer->pushConstants(this->playout.get(),
                          vk::ShaderStageFlagBits::eCompute,
                          0,
                          sizeof(Parameters),
                          &params);
    buffer->dispatch(this->groupCount(shape.x, this->x),
                     this->groupCount(shape.y, this->y),
                     this->groupCount(shape.z, this->z));
    buffer->end();

    return vk::SubmitInfo{
      .commandBufferCount=1,
      .pCommandBuffers=&buffer.get()
    };
  }
};

class Job {
private:
  std::shared_ptr<GPU> gpu;
  vk::UniqueCommandPool pool;
  vk::UniqueCommandBuffer buffer;
  vk::UniqueFence fence;
  vk::UniqueSemaphore semaphore;
  std::function<vk::Result(std::uint64_t)> w;
public:
  template<std::uint32_t N, typename Parameters>
  Job(std::shared_ptr<GPU> gpu,
      vk::UniqueDevice& device,
      vk::CommandPoolCreateInfo info,
      vk::Queue& queue,
      const Op<N, Parameters>& op,
      const DataShape& shape,
      const Parameters& params,
      const std::vector<vk::Semaphore>& wait) : gpu(gpu)
  {
    this->pool = device->createCommandPoolUnique(info);

    auto alloc = vk::CommandBufferAllocateInfo{
      .commandPool = this->pool.get(),
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1
    };
    this->buffer = std::move(device->allocateCommandBuffersUnique(alloc)[0]);

    this->fence = device->createFenceUnique(vk::FenceCreateInfo{});
    this->semaphore = device->createSemaphoreUnique(vk::SemaphoreCreateInfo{});

    this->w = [this, &device](std::uint64_t timeout_ns){
      return device->waitForFences({ this->fence.get() }, VK_TRUE, timeout_ns);
    };

    auto submit = op.getSubmitInfo(this->buffer, shape, params);
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &this->semaphore.get();
    if(!wait.empty()){
      submit.waitSemaphoreCount = wait.size();
      submit.pWaitSemaphores = wait.data();
    }
    queue.submit(submit, this->fence.get());
  }

  Job(Job&& other) = default;

  ~Job(){
    this->wait();
  }

  void wait(std::uint64_t timeout_ns = std::numeric_limits<std::uint64_t>::max()){
    switch(this->w(timeout_ns)){
    case vk::Result::eSuccess:
      break;
    case vk::Result::eTimeout:
      throw std::runtime_error("Timeout at Command Wait");
      break;
    default:
      throw std::runtime_error("Error at Command Wait");
    }
  }

  vk::Semaphore getSemaphore(){
    return this->semaphore.get();
  }
};


class GPU : public std::enable_shared_from_this<GPU> {
private:
  float priority;
  std::uint32_t queueFamilyIndex;
  vk::UniqueInstance instance;
  vk::PhysicalDevice physical;
  vk::DeviceQueueCreateInfo queueInfo;
  vk::DeviceCreateInfo deviceInfo;
  vk::UniqueDevice device;
  vk::Queue queue;
public:
  GPU(std::size_t id, float priority = 0.0f): priority(priority) {
    this->instance = vk::createInstanceUnique(vk::InstanceCreateInfo{});
    this->physical = this->instance->enumeratePhysicalDevices()[id];

    this->queueFamilyIndex = this->findQueueFamilyIndex(vk::QueueFlagBits::eCompute);

    this->queueInfo = vk::DeviceQueueCreateInfo{
      .queueFamilyIndex=this->queueFamilyIndex,
      .queueCount=1,
      .pQueuePriorities=&this->priority
    };
    this->deviceInfo = vk::DeviceCreateInfo{
      .queueCreateInfoCount=1,
      .pQueueCreateInfos=&this->queueInfo
    };

    this->device = this->physical.createDeviceUnique(this->deviceInfo);
    this->queue = this->device->getQueue(this->queueFamilyIndex, 0);
  }

  std::uint32_t findQueueFamilyIndex(vk::QueueFlagBits flag){
    auto ps = this->physical.getQueueFamilyProperties();
    for(std::uint32_t i = 0, n = ps.size(); i < n; i++){
      if(ps[i].queueFlags & flag){ return i; }
    }
    throw std::runtime_error("Failed to find Queue");
  }

  template<typename T> Buffer<T> toBuffer(const std::vector<T>& data){
    return Buffer<T>(this->shared_from_this(),
                     this->device, this->physical.getMemoryProperties(), data);
  }

  template<typename T> Buffer<T> createBuffer(std::size_t n){
    return Buffer<T>(this->shared_from_this(),
                     this->device, this->physical.getMemoryProperties(), n);
  }

  template<std::uint32_t N, typename Parameters>
  Op<N, Parameters> createOp(std::string_view spv,
                             std::uint32_t x, std::uint32_t y, std::uint32_t z){
    return Op<N, Parameters>{this->shared_from_this(), this->device, spv, x, y, z};
  }

  template<std::uint32_t N, typename Parameters>
  std::shared_ptr<Job> submit(const Op<N, Parameters>& op,
                              const vk::DescriptorBufferInfo (&info)[N],
                              const DataShape& shape, const Parameters& params = {},
                              const std::vector<vk::Semaphore>& wait = {}){
    op.writeDescriptorSet(this->device, info);

    return std::shared_ptr<Job>(new Job{
        this->shared_from_this(),
        this->device,
        vk::CommandPoolCreateInfo{
          .flags=vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          .queueFamilyIndex=this->queueFamilyIndex
        },
        this->queue, op, shape, params, wait
      });
  }

  template<typename Ranges>
  void flush(Ranges&& ranges){
    this->device->flushMappedMemoryRanges(std::forward<Ranges>(ranges));
  }

  template<typename T>
  void flush(std::initializer_list<T> ranges){
    this->device->flushMappedMemoryRanges(ranges);
  }

  void wait(){
    this->queue.waitIdle();
  }
};


// Helper Functions
template<typename Parameters, pybind11::size_t ...I>
pybind11::object createOp(GPU& m, int n, const Parameters&,
                          std::string_view spv,
                          std::uint32_t x, std::uint32_t y, std::uint32_t z){
  using pybind11::cast;
  using Tuple = std::tuple<pybind11::size_t, std::function<pybind11::object()>>;
  auto ops = {
    Tuple(I, [&](){ return cast(m.createOp<I, Parameters>(spv, x, y, z)); })...
  };

  for(auto op : ops){
    auto [i, f] = op;
    if(i == n){ return f(); }
  }

  throw std::runtime_error("Unknown Operation");
}

template<std::size_t N, typename Parameters>
auto submit(GPU& m,
            const Op<N, Parameters>& op,
            const pybind11::list& py_info,
            const DataShape& shape,
            const Parameters& params,
            const std::vector<vk::Semaphore>& wait){
  // Automatic conversion cannot work for `const T(&)[N]`,
  // so that we manually convert from Python's `list`.

  using B = vk::DescriptorBufferInfo;
  return util::pylist2array<B>([&](const B (&info)[N]){
    return m.submit(op, info, shape, params, wait);
  }, py_info, std::make_index_sequence<N>());
}


PYBIND11_MODULE(_vkarray, m){
  m.doc() = "_vkarray internal module";

  m.def("createGPU",
        [](std::size_t n, float priority){
          return std::make_shared<GPU>(n, priority);
        },
        "Create GPU");

  pybind11::class_<GPU, std::shared_ptr<GPU>>(m, "GPU")
    .def("toBuffer", &GPU::toBuffer<float>, "Copy to GPU Buffer")
    .def("createBuffer", &GPU::createBuffer<float>, "Create GPU Buffer")
    .def("createOp", &createOp<OpParams::Vector, 1, 2, 3, 4>,
         "Create Vector Operation")
    .def("createOp", &createOp<OpParams::VectorScalar<float>, 1, 2, 3>,
         "Create Vector-Scalar Operation")
    .def("createOp", &createOp<OpParams::VectorMultiScalar<float, 2>, 1, 2>,
         "Create Vector-Scalar[2] Operation")
    .def("createOp", &createOp<OpParams::MatMul<float>, 3>,
         "Create Matrix Multiplication Operation")
    .def("submit", &submit<1, OpParams::Vector>, "Submit Vector Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<2, OpParams::Vector>, "Submit Vector Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<3, OpParams::Vector>, "Submit Vector Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<4, OpParams::Vector>, "Submit Vector Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<1, OpParams::VectorScalar<float>>,
         "Submit Vector-Scalar Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<2, OpParams::VectorScalar<float>>,
         "Submit Vector-Scalar Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<3, OpParams::VectorScalar<float>>,
         "Submit Vector-Scalar Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<1, OpParams::VectorMultiScalar<float, 2>>,
         "Submit Vector-Scalar[2] Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<2, OpParams::VectorMultiScalar<float, 2>>,
         "Submit Vector-Scalar[2] Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<3, OpParams::MatMul<float>>,
         "Submit Matrix Multiplication Operation",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("wait", &GPU::wait, "Wait all submission")
    .def("flush",
         [](GPU& m, const std::vector<vk::MappedMemoryRange>& range){ m.flush(range); },
         "Flush Memories to GPU");

  pybind11::class_<Buffer<float>>(m, "Buffer", pybind11::buffer_protocol())
    .def("info", &Buffer<float>::info, "Get Buffer Info")
    .def("range", &Buffer<float>::range, "Get Buffer Range")
    .def("size", &Buffer<float>::size, "Get Buffer Size")
    .def_buffer([](Buffer<float>& m) {
      return pybind11::buffer_info {
        .ptr=m.data(),
        .itemsize=sizeof(float),
        .format=pybind11::format_descriptor<float>::format(),
        .ndim=1,
        .shape={ m.size() },
        .strides={ sizeof(float) }
      };
    });

  pybind11::class_<OpParams::Vector>(m, "VectorParams")
    .def(pybind11::init<std::uint32_t>());

  pybind11::class_<OpParams::VectorScalar<float>>(m, "VectorScalarParams")
    .def(pybind11::init<std::uint32_t, float>());

  pybind11::class_<OpParams::VectorMultiScalar<float, 2>>(m, "VectorScalar2Params")
    .def(pybind11::init<std::uint32_t, float, float>());

  pybind11::class_<OpParams::MatMul<float>>(m, "MatMulParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<DataShape>(m, "DataShape")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<Op<1, OpParams::Vector>>(m, "OpVec1");
  pybind11::class_<Op<2, OpParams::Vector>>(m, "OpVec2");
  pybind11::class_<Op<3, OpParams::Vector>>(m, "OpVec3");
  pybind11::class_<Op<4, OpParams::Vector>>(m, "OpVec4");
  pybind11::class_<Op<1, OpParams::VectorScalar<float>>>(m, "OpVecScalar1");
  pybind11::class_<Op<2, OpParams::VectorScalar<float>>>(m, "OpVecScalar2");
  pybind11::class_<Op<3, OpParams::VectorScalar<float>>>(m, "OpVecScalar3");
  pybind11::class_<Op<1, OpParams::VectorMultiScalar<float, 2>>>(m, "OpVec2Scalar1");
  pybind11::class_<Op<2, OpParams::VectorMultiScalar<float, 2>>>(m, "OpVec2Scalar2");
  pybind11::class_<Op<3, OpParams::MatMul<float>>>(m, "OpMatMul");

  pybind11::class_<Job, std::shared_ptr<Job>>(m, "Job")
    .def("wait", &Job::wait, "Wait for this Job",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("wait", [](Job& m){ m.wait(); }, "Wait for this Job")
    .def("getSemaphore", &Job::getSemaphore, "Get Semaphore");

  pybind11::class_<vk::DescriptorBufferInfo>(m, "BufferInfo");
  pybind11::class_<vk::MappedMemoryRange>(m, "MemoryRange");
  pybind11::class_<vk::Semaphore>(m, "Semaphore");
}
