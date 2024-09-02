#include <cuda_fp16.h>
#include <memory.h>
#include <utils.h>

#include <cstring>

#include "iostream"

namespace trt
{
Memory::Memory(size_t num_of_byte, bool alloc)
{
    cpu_size_ = num_of_byte;
    gpu_size_ = num_of_byte;
    if (alloc)
        malloc_host_and_device_memory();
}

Memory::Memory(int id, size_t num_of_byte, bool alloc)
{
    id_       = id;
    cpu_size_ = num_of_byte;
    gpu_size_ = num_of_byte;
    if (alloc)
        malloc_host_and_device_memory();
}

Memory::Memory(int id, size_t num_of_byte, bool alloc, cudaStream_t stream)
{
    id_       = id;
    cpu_size_ = num_of_byte;
    gpu_size_ = num_of_byte;
    stream_   = stream;
    if (alloc)
        malloc_host_and_device_memory();
}

void Memory::malloc_host_memory()
{
    if (cpu_size_ == 0)
    {
        INFO("Malloc num_of_byte == 0.");
        return;
    }
    if (host_ptr_ != nullptr)
    {
        free_host_memory();
    }

    //    cpu_size_= align_size(cpu_size_,ALIGN_SIZE);
    checkRuntime(cudaMallocHost(&host_ptr_, cpu_size_));
    Assert(host_ptr_);
    INFO("Malloc Host Mem: %d Byte.", cpu_size_);
}

void Memory::malloc_host_memory(size_t num_of_byte)
{
    if (num_of_byte > cpu_size_)
    {
        cpu_size_ = num_of_byte;
        malloc_host_memory();
    }
}

void Memory::malloc_device_memory()
{
    if (gpu_size_ == 0)
    {
        INFO("Malloc num_of_byte == 0.");
    }
    if (device_ptr_ != nullptr)
    {
        free_device_memory();
    }

    //    gpu_size_ = align_size(gpu_size_, ALIGN_SIZE);
    checkRuntime(cudaMalloc(&device_ptr_, gpu_size_));
    INFO("Malloc Device Mem: %d Byte.", gpu_size_);
}

void Memory::malloc_device_memory(size_t num_of_byte)
{
    if (num_of_byte > gpu_size_)
    {
        gpu_size_ = num_of_byte;
        malloc_device_memory();
    }
}

void Memory::malloc_host_and_device_memory()
{
    malloc_host_memory();
    malloc_device_memory();
}

void Memory::free_device_memory()
{
    checkRuntime(cudaFree(device_ptr_));
    device_ptr_ = nullptr;
    gpu_size_   = 0;
    INFO("Free Device Memory.");
}

void Memory::free_host_memory()
{
    checkRuntime(cudaFreeHost(host_ptr_));
    host_ptr_ = nullptr;
    cpu_size_ = 0;
    INFO("Free Host Memory.");
}

void Memory::free_host_and_device_memory()
{
    free_device_memory();
    free_host_memory();
}

void Memory::to_gpu()
{
    assert_host();
    assert_device();
    checkRuntime(cudaMemcpyAsync(device_ptr_, host_ptr_, cpu_size_, H2D, stream_));
    sync();
}

void Memory::to_cpu()
{
    assert_host();
    assert_device();
    checkRuntime(cudaMemcpyAsync(host_ptr_, device_ptr_, gpu_size_, D2H, stream_));
    sync();
}

void Memory::to_other_gpu(void* out, int size, size_t offset)
{
    if (size <= 0)
        size = cpu_size_;

    assert_host();
    assert(size <= cpu_size_);

    void* in = offset_ptr(host_ptr_, offset);

    checkRuntime(cudaMemcpyAsync(out, in, size, D2H, stream_));
    sync();
}

void Memory::to_other_cpu(void* out, int size, size_t offset)
{
    if (size <= 0)
        size = gpu_size_;

    assert_device();

    assert(size <= gpu_size_);

    void* in = offset_ptr(device_ptr_, offset);

    checkRuntime(cudaMemcpyAsync(out, in, size, D2H, stream_));
    sync();
}

void Memory::cpu2cpu(void* out, int size, size_t offset)
{
    if (size <= 0)
        size = cpu_size_;

    assert_host();

    assert(size <= cpu_size_);

    void* in = offset_ptr(host_ptr_, offset);

    checkRuntime(cudaMemcpyAsync(out, in, size, H2H, stream_));
    sync();
};

void Memory::gpu2gpu(void* out, int size, size_t offset)
{
    if (size <= 0)
        size = gpu_size_;

    assert_device();

    assert(size <= gpu_size_);

    void* in = offset_ptr(device_ptr_, offset);

    checkRuntime(cudaMemcpyAsync(out, in, size, D2D, stream_));
    sync();
}

float Memory::float16_to_float32(half value)
{
    return __half2float(*reinterpret_cast<__half*>(&value));
}

half Memory::float32_to_float16(float value)
{
    auto val = __float2half(value);
    return val;
}

void Memory::sync()
{
    checkRuntime(cudaStreamSynchronize(stream_));
}

void Memory::assert_host()
{
    assert(host_ptr_);
    assert(cpu_size_ != 0);
}

void Memory::assert_device()
{
    assert(device_ptr_);
    assert(gpu_size_ != 0);
}

void* Memory::offset_ptr(void* ptr, size_t offset)
{
    if (offset != 0)
        return (void*) ((char*) ptr + offset);
    else
        return device_ptr_;
}

size_t Memory::align_size(size_t sz, size_t n)
{
    assert(sz != 0);
    assert(n != 0);
    return (sz + n - 1) & -n;
}

} // namespace trt