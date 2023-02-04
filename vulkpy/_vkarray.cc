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


#include "_vkutil.hh"

constexpr const auto _1sec = std::uint64_t(1e+9);

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


std::vector<char> readCode(std::string_view name){
  auto f = std::ifstream(name.data(), std::ios::ate | std::ios::binary);
  if(!f.is_open()){
    throw std::runtime_error("failed to open file");
  }
  auto size = f.tellg();
  f.seekg(0);

  auto v = std::vector<char>(size);
  f.read(v.data(), size);

  f.close();
  return v;
}


template<typename T> class Buffer {
private:
  std::uint64_t mSize;
  vk::UniqueBuffer b;
  vk::UniqueDeviceMemory m;
  T* ptr;

public:
  Buffer(vk::UniqueDevice& device,
         const vk::PhysicalDeviceMemoryProperties& ps, std::size_t n)
    : mSize(sizeof(T) * n)
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

  Buffer(vk::UniqueDevice& device,
         const vk::PhysicalDeviceMemoryProperties& ps, const std::vector<T>& data)
    : Buffer<T>(device, ps, data.size())
  {
    memcpy((void*)this->ptr, (void*)data.data(), this->mSize);
  }

  T get(std::size_t i) const {
    return this->ptr[i];
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
}

struct DataShape {
  std::uint32_t x, y, z;
};

template <std::uint32_t N, typename Parameters = OpParams::Empty> class Op {
private:
  std::uint32_t x, y, z;
  vk::UniqueDescriptorPool pool;
  vk::UniqueShaderModule shader;
  vk::UniqueDescriptorSetLayout dlayout;
  vk::UniquePipelineLayout playout;
  vk::UniquePipeline pipe;
  vk::UniqueDescriptorSet desc;
public:
  Op(vk::UniqueDevice& device, std::string_view spv,
     std::uint32_t x, std::uint32_t y = 1, std::uint32_t z = 1){
    this->x = x;
    this->y = y;
    this->z = z;

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

    auto code = readCode(spv);

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

class GPU;
class Job {
private:
  vk::UniqueCommandPool pool;
  vk::UniqueCommandBuffer buffer;
  vk::UniqueFence fence;
  std::function<vk::Result(std::uint64_t)> w;
public:
  template<std::uint32_t N, typename Parameters>
  Job(vk::UniqueDevice& device,
      vk::CommandPoolCreateInfo info,
      vk::Queue& queue,
      const Op<N, Parameters>& op,
      const DataShape& shape,
      const Parameters& params)
  {
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

    queue.submit(op.getSubmitInfo(this->buffer, shape, params), this->fence.get());
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


class GPU {
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
    return Buffer<T>(this->device, this->physical.getMemoryProperties(), data);
  }

  template<std::uint32_t N, typename Parameters>
  Op<N, Parameters> createOp(std::string_view spv,
                             std::uint32_t x, std::uint32_t y, std::uint32_t z){
    return Op<N, Parameters>{this->device, spv, x, y, z};
  }

  template<std::uint32_t N, typename Parameters>
  Job submit(const Op<N, Parameters>& op,
             const vk::DescriptorBufferInfo (&info)[N],
             const DataShape& shape, const Parameters& params = {}){
    op.writeDescriptorSet(this->device, info);

    return Job{
      this->device,
      vk::CommandPoolCreateInfo{
        .flags=vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex=this->queueFamilyIndex
      },
      this->queue, op, shape, params
    };
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


int main(int argc, char** argv){
  const auto nhead = 3;

  auto gpu = GPU(0, 0.0f);

  std::uint32_t size = 100;
  auto A = std::vector<float>(size, 3);
  auto B = std::vector<float>(size, 5);
  auto C = std::vector<float>(size, 0);

  std::cout << "vector\nA B C" << std::endl;
  for(auto i=0; i < nhead; i++){
    std::cout << A[i] << " " << B[i] << " " << C[i] << "\n";
  }
  std::cout << std::endl;

  auto bA = gpu.toBuffer(A);
  auto bB = gpu.toBuffer(B);
  auto bC = gpu.toBuffer(C);

  gpu.flush({bA.range(), bB.range(), bC.range()});

  std::cout << "GPU\nA B C" << std::endl;
  for(auto i=0; i < nhead; i++){
    std::cout << bA.get(i) << " " << bB.get(i) << " " << bC.get(i) << "\n";
  }
  std::cout << std::endl;

  auto add = gpu.createOp<3, OpParams::Vector>("./shader/add.spv", 64, 1, 1);

  auto job = gpu.submit(add, {bA.info(), bB.info(), bC.info()}, {size, 1, 1}, { size });

  job.wait();
  gpu.flush({bA.range(), bB.range(), bC.range()});

  std::cout << "Result\nA B C" << std::endl;
  for(auto i=0; i < nhead; i++){
    std::cout << bA.get(i) << " " << bB.get(i) << " " << bC.get(i) << "\n";
  }
  std::cout << std::endl;
}
