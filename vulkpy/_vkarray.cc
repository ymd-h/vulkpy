#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>
#include <random>
#include <tuple>

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
  std::function<void(void)> d;

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

    this->d = [&device, this](){
      if(this->m.get()){
        device->unmapMemory(this->m.get());
      }
    };
  }

  Buffer(std::shared_ptr<GPU> gpu, vk::UniqueDevice& device,
         const vk::PhysicalDeviceMemoryProperties& ps, const std::vector<T>& data)
    : Buffer<T>(gpu, device, ps, data.size())
  {
    memcpy((void*)this->ptr, (void*)data.data(), this->mSize);
  }

  ~Buffer(){
    if(this->d){
      this->d();
    }
  }

  T get(std::size_t i) const {
    return this->ptr[i];
  }

  void set(std::size_t i, T v){
    this->ptr[i] = v;
  }

  void set(std::size_t i, const std::vector<T>& data){
    auto m = std::min(this->nSize, data.size()-i) * sizeof(T);
    memcpy((void*)(this->ptr+i), (void*)data.data(), m);
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

  template<std::size_t N>
  struct MultiVector{
    std::uint32_t size[N];
  };

  struct ShiftVector{
    std::uint32_t shift;
    std::uint32_t size;
  };

  struct VectorRange{
    std::uint32_t size;
    std::uint32_t low;
    std::uint32_t high;
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

  struct BatchAffine{
    std::uint32_t batch_size;
    std::uint32_t input_size;
    std::uint32_t output_size;
  };

  struct AxisReduction{
    std::uint32_t prev_prod;
    std::uint32_t axis_size;
    std::uint32_t post_prod;
  };

  struct Broadcast {
    std::uint32_t size[2];
    std::uint32_t ndim;
  };

  template<std::size_t N>
  struct MultiBroadcast {
    std::uint32_t size[N];
    std::uint32_t ndim;
  };

  struct AxisGather{
    std::uint32_t prev_prod;
    std::uint32_t post_prod;
    std::uint32_t axis_size;
    std::uint32_t index_size;
  };
}

struct DataShape {
  std::uint32_t x, y, z;
};

using DescriptorSet = std::tuple<vk::UniqueDescriptorPool, vk::UniqueDescriptorSet>;

template <std::uint32_t N, typename Parameters = OpParams::Empty>
class Op : std::enable_shared_from_this<Op<N, Parameters>> {
private:
  std::uint32_t x, y, z;
  std::shared_ptr<GPU> gpu;
  vk::PipelineCache cache;
  vk::UniqueShaderModule shader;
  vk::UniqueDescriptorSetLayout dlayout;
  vk::UniquePipelineLayout playout;
  std::function<vk::UniquePipeline()> pipe;
  std::function<DescriptorSet()> desc;
public:
  Op(std::shared_ptr<GPU> gpu, vk::UniqueDevice& device,
     std::string_view spv, std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1)
    : x(x), y(y), z(z), gpu(gpu), cache(device->createPipelineCache({}))
  {
    auto psize = vk::DescriptorPoolSize{
      .type=vk::DescriptorType::eStorageBuffer,
      .descriptorCount=N
    };

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

    auto dlinfo = vk::DescriptorSetLayoutCreateInfo{
      .bindingCount=N,
      .pBindings=dbind.data()
    };
    this->dlayout = device->createDescriptorSetLayoutUnique(dlinfo);

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
    auto pinfo = vk::ComputePipelineCreateInfo{
      .stage=ssinfo,
      .layout=this->playout.get()
    };

    this->pipe = [this, ssinfo, pinfo, &device](){
      auto p = device->createComputePipelineUnique(this->cache, pinfo);

      switch(p.result){
      case vk::Result::eSuccess:
        break;
      default:
        throw std::runtime_error("Fail: createComputePipeline");
      }

      return std::move(p.value);
    };

    this->desc = [this, &device, psize](){
      auto pool = device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{
          .flags=vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
          .maxSets=1,
          .poolSizeCount=1,
          .pPoolSizes=&psize,
        });

      auto d = vk::DescriptorSetAllocateInfo{
        .descriptorPool=pool.get(),
        .descriptorSetCount=1,
        .pSetLayouts=&this->dlayout.get()
      };
      return std::make_tuple(std::move(pool),
                             std::move(device->allocateDescriptorSetsUnique(d)[0]));
    };
  }

  DescriptorSet createDescriptorSet() const {
    return this->desc();
  }

  vk::UniquePipeline createPipeline(){
    return this->pipe();
  }

  void writeDescriptorSet(vk::UniqueDevice& device,
                          vk::UniqueDescriptorSet& desc,
                          const vk::DescriptorBufferInfo (&info)[N]) const {
    auto w = util::generate_from_range([this, &info, &desc](auto i){
      return vk::WriteDescriptorSet{
        .dstSet=desc.get(),
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
                               const vk::UniquePipeline& pipe,
                               const vk::UniqueDescriptorSet& desc,
                               const DataShape& shape,
                               const Parameters& params) const {
    buffer->begin(vk::CommandBufferBeginInfo{});
    buffer->bindPipeline(vk::PipelineBindPoint::eCompute, pipe.get());
    buffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                               this->playout.get(),
                               0,
                               desc.get(),
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

template<std::uint32_t N, typename Parameter>
using Op_t = std::shared_ptr<Op<N, Parameter>>;

using OpVariant_t = std::variant<
  Op_t<1, OpParams::Vector>,
  Op_t<2, OpParams::Vector>,
  Op_t<3, OpParams::Vector>,
  Op_t<4, OpParams::Vector>,
  Op_t<2, OpParams::MultiVector<2>>,
  Op_t<1, OpParams::VectorScalar<float>>,
  Op_t<2, OpParams::VectorScalar<float>>,
  Op_t<3, OpParams::VectorScalar<float>>,
  Op_t<1, OpParams::VectorMultiScalar<float, 2>>,
  Op_t<2, OpParams::VectorMultiScalar<float, 2>>,
  Op_t<3, OpParams::MatMul<float>>,
  Op_t<2, OpParams::AxisReduction>,
  Op_t<2, OpParams::ShiftVector>,
  Op_t<3, OpParams::Broadcast>,
  Op_t<4, OpParams::Broadcast>,
  Op_t<4, OpParams::MultiBroadcast<3>>,
  Op_t<4, OpParams::BatchAffine>,
  Op_t<2, OpParams::VectorRange>,
  Op_t<3, OpParams::AxisGather>
  >;


class Job {
private:
  OpVariant_t opV;
  vk::UniquePipeline pipe;
  vk::UniqueDescriptorPool dpool;
  vk::UniqueDescriptorSet desc;
  vk::UniqueCommandPool pool;
  vk::UniqueCommandBuffer buffer;
  vk::UniqueFence fence;
  std::function<vk::Result(std::uint64_t)> w;
public:
  template<std::uint32_t N, typename Parameter>
  Job(vk::UniqueDevice& device,
      vk::CommandPoolCreateInfo info,
      vk::Queue& queue,
      Op_t<N, Parameter> op,
      const vk::DescriptorBufferInfo (&infos)[N],
      const DataShape& shape,
      const Parameter& params,
      const std::vector<std::shared_ptr<Job>>& wait)
    : opV(op), pipe(op->createPipeline())
  {
    std::tie(this->dpool, this->desc) = op->createDescriptorSet();
    this->pool = device->createCommandPoolUnique(info);

    auto alloc = vk::CommandBufferAllocateInfo{
      .commandPool = this->pool.get(),
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1
    };
    this->buffer = std::move(device->allocateCommandBuffersUnique(alloc)[0]);

    this->fence = device->createFenceUnique(vk::FenceCreateInfo{});

    this->w = [this, &device](std::uint64_t timeout_ns){
      return device->waitForFences({ this->fence.get() }, VK_TRUE, timeout_ns);
    };

    // ToDo: Since semaphore is problematic,
    //       we temporary wait depending job with fence.
    for(auto& ws: wait){ ws->wait(); }

    op->writeDescriptorSet(device, this->desc, infos);
    auto submit = op->getSubmitInfo(this->buffer, this->pipe, this->desc,
                                    shape, params);
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
  std::unordered_map<std::string_view, OpVariant_t> opMap;
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

  template<typename T>
  std::shared_ptr<Buffer<T>> toBuffer(const std::vector<T>& data){
    return std::make_shared<Buffer<T>>(this->shared_from_this(),
                                       this->device,
                                       this->physical.getMemoryProperties(),
                                       data);
  }

  template<typename T>
  std::shared_ptr<Buffer<T>> createBuffer(std::size_t n){
    return std::make_shared<Buffer<T>>(this->shared_from_this(),
                                       this->device,
                                       this->physical.getMemoryProperties(),
                                       n);
  }

  template<std::uint32_t N, typename Parameter>
  std::shared_ptr<Job> submit(std::string_view spv,
                              std::uint32_t x, std::uint32_t y, std::uint32_t z,
                              const vk::DescriptorBufferInfo (&info)[N],
                              const DataShape& shape, const Parameter& params = {},
                              const std::vector<std::shared_ptr<Job>>& wait = {}){
    if(!this->opMap.contains(spv)){
      this->opMap.emplace(spv, Op_t<N, Parameter>(new Op<N, Parameter>{
            this->shared_from_this(),
            this->device, spv, x, y, z
          }));
    }
    auto op = std::get<Op_t<N, Parameter>>(this->opMap[spv]);
    return std::shared_ptr<Job>(new Job{
        this->device,
        vk::CommandPoolCreateInfo{
          .flags=vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          .queueFamilyIndex=this->queueFamilyIndex
        },
        this->queue, op, info, shape, params, wait
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

  bool canSubgroupArithmetic() const {
    if(this->physical.getProperties().apiVersion < util::VK_API_VERSION(1, 1, 0)){
      return false;
    }

    using sub_t = vk::PhysicalDeviceSubgroupProperties;
    constexpr const auto eArithmetic = vk::SubgroupFeatureFlagBits::eArithmetic;
    auto p = this->physical.getProperties2<vk::PhysicalDeviceProperties2, sub_t>();
    return bool(p.get<sub_t>().supportedOperations & eArithmetic);
  }
};


namespace PRNG {
  // xoshiro128++
  // https://prng.di.unimi.it/xoshiro128plusplus.c
  class Xoshiro128pp {
  private:
    const std::uint32_t size;
    std::shared_ptr<GPU> gpu;
    std::shared_ptr<Job> job;
    std::shared_ptr<Buffer<std::uint32_t>> state;
    std::string_view spv_uint32;
    std::string_view spv_float;

    std::uint64_t splitmix64(std::uint64_t x) const noexcept {
      // https://prng.di.unimi.it/splitmix64.c
      std::uint64_t z = (x += 0x9e3779b97f4a7c15);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
      z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
      return z ^ (z >> 31);
    }

    void jump(std::uint32_t (&s)[4]) const noexcept {
      // Equivalent to 2^64 calls of next()
      constexpr const std::uint32_t JUMP[] = {
        0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b
      };
      constexpr const std::uint32_t one = 1;

      std::uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for(auto j : JUMP){
        for(auto b = 0; b < 32; b++){
          if(j & one << b){
            s0 ^= s[0];
            s1 ^= s[1];
            s2 ^= s[2];
            s3 ^= s[3];
          }
          this->next_on_cpu(s);
        }

        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
        s[3] = s3;
      }
    }

    std::uint32_t rtol(const std::uint32_t x, int k) const noexcept {
      return (x << k) | (x >> (32 - k));
    }

    std::uint32_t next_on_cpu(std::uint32_t (&s)[4]) const noexcept {
      // Only for Initialization
      const std::uint32_t result = rtol(s[0] + s[3], 7) + s[0];
      const std::uint32_t t = s[1] << 9;

      s[2] ^= s[0];
      s[3] ^= s[1];
      s[1] ^= s[2];
      s[0] ^= s[3];

      s[2] ^= t;
      s[3] = this->rtol(s[3], 11);

      return result;
    }
  public:
    Xoshiro128pp(std::shared_ptr<GPU> gpu,
                 std::string_view spv_uint32, std::string_view spv_float,
                 std::uint32_t size, std::uint64_t seed)
      : size(size),
        gpu(gpu),
        job(),
        state(gpu->createBuffer<std::uint32_t>(4 * size)),
        spv_uint32(spv_uint32),
        spv_float(spv_float)
    {
      std::vector<std::uint32_t> state_vec{};
      state_vec.reserve(4 * this->size);

      std::uint32_t s[4]{};

      // Compute Initial State with SplitMix64
      for(auto i = 0; i < 4; i++){
        seed = this->splitmix64(seed);
        s[i] = (std::uint32_t)seed;
        state_vec.push_back((std::uint32_t)seed);
      }

      // Create enough far Starting Points.
      for(std::uint32_t i = 1, N = this->size; i < N; i++){
        this->jump(s);
        state_vec.push_back(s[0]);
        state_vec.push_back(s[1]);
        state_vec.push_back(s[2]);
        state_vec.push_back(s[3]);
      }

      this->state->set(0, state_vec);
    }
    Xoshiro128pp(std::shared_ptr<GPU> gpu,
                 std::string_view spv_uint32, std::string_view spv_float,
                 std::uint32_t size)
      : Xoshiro128pp(gpu, spv_uint32, spv_float, size, std::random_device{}()) {}

    std::shared_ptr<Job> random_uint32(std::uint32_t n,
                                       const vk::DescriptorBufferInfo& info){
      return this->random(n, this->spv_uint32, info);
    }

    ~Xoshiro128pp(){
      if(this->job){
        this->job->wait();
      }
    }

    std::shared_ptr<Job> random_float(std::uint32_t n,
                                      const vk::DescriptorBufferInfo& info){
      return this->random(n, this->spv_float, info);
    }

    std::shared_ptr<Job> random(std::uint32_t n, std::string_view spv,
                                const vk::DescriptorBufferInfo& info){
      vk::DescriptorBufferInfo b[]{ this->state->info(), info };
      auto f = [this, &b, spv](std::uint32_t i, std::uint32_t n){
        return this->gpu->submit<2>(spv, 64, 1, 1, b, {n, 1, 1},
                                    OpParams::ShiftVector{i, n}, {});
      };

      if(n <= this->size){
        if(this->job){ this->job->wait(); }
        this->job = f(0, n);
        return this->job;
      }

      for(std::uint32_t i = 0; i < n; i += this->size){
        if(this->job){ this->job->wait(); }
        auto local_size = std::min(this->size, (n-i));
        this->job = f(i, local_size);
      }
      return this->job;
    }
  };
} // namespace PRNG



// Helper Functions
template<typename Parameter, pybind11::size_t ...I>
auto submit(GPU& m,
            std::string_view spv,
            std::uint32_t x, std::uint32_t y, std::uint32_t z,
            const pybind11::list& py_info,
            const DataShape& shape,
            const Parameter& params,
            const std::vector<std::shared_ptr<Job>>& wait){
  using F = std::function<std::shared_ptr<Job>()>;
  using Tuple = std::tuple<std::uint32_t, F>;
  using B = vk::DescriptorBufferInfo;

  auto ops = {
    // Automatic conversion cannot work for `const T(&)[N]`,
    // so that we manually convert from Python's `list`.
    Tuple(std::uint32_t(I), [&](){
      return util::pylist2array<B>([&](const B (&info)[I]){
        return m.submit<I, Parameter>(spv, x, y, z, info, shape, params, wait);
      }, py_info, std::make_index_sequence<I>());
    })...
  };

  std::uint32_t n = py_info.size();
  for(auto op : ops){
    auto [i, f] = op;
    if(i == n){ return f(); }
  }

  throw std::runtime_error("Unknown Operation");
}


PYBIND11_MODULE(_vkarray, m){
  m.doc() = "_vkarray internal module";

  m.def("createGPU",
        [](std::size_t n, float priority){
          return std::make_shared<GPU>(n, priority);
        },
        "Create GPU");

  pybind11::class_<GPU, std::shared_ptr<GPU>>(m, "GPU")
    .def("toBuffer", &GPU::toBuffer<float>)
    .def("createBuffer", &GPU::createBuffer<float>)
    .def("toU32Buffer", &GPU::toBuffer<std::uint32_t>)
    .def("createU32Buffer", &GPU::createBuffer<std::uint32_t>)
    .def("submit", &submit<OpParams::Vector, 1, 2, 3, 4>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::MultiVector<2>, 2>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::VectorScalar<float>, 1, 2, 3>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::VectorMultiScalar<float, 2>, 1, 2>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::MatMul<float>, 3>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::AxisReduction, 2>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::Broadcast, 3, 4>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::MultiBroadcast<3>, 4>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::BatchAffine, 4>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::VectorRange, 2>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("submit", &submit<OpParams::AxisGather, 3>,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("wait", &GPU::wait)
    .def("flush",
         [](GPU& m, const std::vector<vk::MappedMemoryRange>& r){ m.flush(r); })
    .def("canSubgroupArithmetic", &GPU::canSubgroupArithmetic);

  using FBuffer_t = Buffer<float>;
  pybind11::class_<FBuffer_t,
                   std::shared_ptr<FBuffer_t>>(m,
                                               "Buffer",
                                               pybind11::buffer_protocol())
    .def("info", &Buffer<float>::info)
    .def("range", &Buffer<float>::range)
    .def("size", &Buffer<float>::size)
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

  using U32Buffer_t = Buffer<std::uint32_t>;
  pybind11::class_<U32Buffer_t,
                   std::shared_ptr<U32Buffer_t>>(m,
                                                 "Shape",
                                                 pybind11::buffer_protocol())
    .def("info", &Buffer<std::uint32_t>::info)
    .def("range", &Buffer<std::uint32_t>::range)
    .def("size", &Buffer<std::uint32_t>::size)
    .def_buffer([](Buffer<std::uint32_t>& m) {
      return pybind11::buffer_info {
        .ptr=m.data(),
        .itemsize=sizeof(std::uint32_t),
        .format=pybind11::format_descriptor<std::uint32_t>::format(),
        .ndim=1,
        .shape={ m.size() },
        .strides={ sizeof(std::uint32_t) }
      };
    });

  pybind11::class_<OpParams::Vector>(m, "VectorParams")
    .def(pybind11::init<std::uint32_t>());

  pybind11::class_<OpParams::MultiVector<2>>(m, "MultiVector2Params")
    .def(pybind11::init<std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::VectorScalar<float>>(m, "VectorScalarParams")
    .def(pybind11::init<std::uint32_t, float>());

  pybind11::class_<OpParams::VectorMultiScalar<float, 2>>(m, "VectorScalar2Params")
    .def(pybind11::init<std::uint32_t, float, float>());

  pybind11::class_<OpParams::MatMul<float>>(m, "MatMulParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::AxisReduction>(m, "AxisReductionParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::Broadcast>(m, "BroadcastParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::MultiBroadcast<3>>(m, "Multi3BroadcastParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::BatchAffine>(m, "BatchAffineParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::VectorRange>(m, "VectorRangeParams")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<OpParams::AxisGather>(m, "AxisGatherParams")
    .def(pybind11::init<
         std::uint32_t,
         std::uint32_t,
         std::uint32_t,
         std::uint32_t
         >());

  pybind11::class_<DataShape>(m, "DataShape")
    .def(pybind11::init<std::uint32_t, std::uint32_t, std::uint32_t>());

  pybind11::class_<Job, std::shared_ptr<Job>>(m, "Job")
    .def("wait", &Job::wait, "Wait for this Job",
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("wait", [](Job& m){ m.wait(); }, "Wait for this Job");

  pybind11::class_<vk::DescriptorBufferInfo>(m, "BufferInfo");
  pybind11::class_<vk::MappedMemoryRange>(m, "MemoryRange");

  pybind11::class_<PRNG::Xoshiro128pp>(m, "Xoshiro128pp")
    .def(pybind11::init<
         std::shared_ptr<GPU>,
         std::string_view, std::string_view,
         std::uint32_t, std::uint64_t
         >())
    .def(pybind11::init<
         std::shared_ptr<GPU>,
         std::string_view, std::string_view,
         std::uint32_t>())
    .def("random_uint32", &PRNG::Xoshiro128pp::random_uint32,
         pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("random_float", &PRNG::Xoshiro128pp::random_float,
         pybind11::call_guard<pybind11::gil_scoped_release>());
}
