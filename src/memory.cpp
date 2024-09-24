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
        malloc_cpu_and_gpu_memory();
}

Memory::Memory(int id, size_t num_of_byte, bool alloc)
{
    id_       = id;
    cpu_size_ = num_of_byte;
    gpu_size_ = num_of_byte;
    if (alloc)
        malloc_cpu_and_gpu_memory();
}

Memory::Memory(int id, size_t num_of_byte, bool alloc, cudaStream_t stream)
{
    id_       = id;
    cpu_size_ = num_of_byte;
    gpu_size_ = num_of_byte;
    stream_   = stream;
    if (alloc)
        malloc_cpu_and_gpu_memory();
}

void Memory::malloc_cpu_memory()
{
    if (cpu_size_ == 0)
    {
        INFO("Malloc num_of_byte == 0.");
        return;
    }
    if (cpu_ptr_ != nullptr)
    {
        free_cpu_memory();
    }

    CHECK_CUDA_RUNTIME(cudaMallocHost(&cpu_ptr_, cpu_size_));
    ASSERT_PTR(cpu_ptr_);
    INFO("Malloc Host Mem: %d Byte.", cpu_size_);
}

void Memory::malloc_cpu_memory(size_t num_of_byte)
{
    if (num_of_byte > cpu_size_)
    {
        cpu_size_ = num_of_byte;
        malloc_cpu_memory();
    }
}

void Memory::malloc_gpu_memory()
{
    if (gpu_size_ == 0)
    {
        INFO("Malloc num_of_byte == 0.");
    }
    if (gpu_ptr_ != nullptr)
    {
        free_gpu_memory();
    }

    CHECK_CUDA_RUNTIME(cudaMalloc(&gpu_ptr_, gpu_size_));
    ASSERT_PTR(gpu_ptr_);
    INFO("Malloc Device Mem: %d Byte.", gpu_size_);
}

void Memory::malloc_gpu_memory(size_t num_of_byte)
{
    if (num_of_byte > gpu_size_)
    {
        gpu_size_ = num_of_byte;
        malloc_gpu_memory();
    }
}

void Memory::malloc_cpu_and_gpu_memory()
{
    malloc_cpu_memory();
    malloc_gpu_memory();
}

void Memory::free_gpu_memory()
{
    CHECK_CUDA_RUNTIME(cudaFree(gpu_ptr_));
    gpu_ptr_  = nullptr;
    gpu_size_ = 0;
    INFO("Free Device Memory.");
}

void Memory::free_cpu_memory()
{
    CHECK_CUDA_RUNTIME(cudaFreeHost(cpu_ptr_));
    cpu_ptr_  = nullptr;
    cpu_size_ = 0;
    INFO("Free Host Memory.");
}

void Memory::free_all()
{
    free_gpu_memory();
    free_cpu_memory();
}

void Memory::to_gpu()
{
    assert_cpu();
    assert_gpu();
    ASSERT_TRUE(cpu_size_ == gpu_size_);
    CHECK_CUDA_RUNTIME(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, cpu_size_, cudaMemcpyHostToDevice, stream_));
    sync();
}

void Memory::to_cpu()
{
    assert_cpu();
    assert_gpu();
    ASSERT_TRUE(cpu_size_ == gpu_size_);
    CHECK_CUDA_RUNTIME(cudaMemcpyAsync(cpu_ptr_, gpu_ptr_, gpu_size_, cudaMemcpyDeviceToHost, stream_));
    sync();
}

void Memory::to_gpu(void* out, size_t size, MemcpyKind mode)
{
    ASSERT_PTR(out);
    ASSERT_TRUE(mode == MemcpyKind::CPU2GPU || mode == MemcpyKind::GPU2GPU);
    if (mode == MemcpyKind::CPU2GPU)
    {
        assert_cpu();
        ASSERT_TRUE(0 < size <= cpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, cpu_ptr_, size, cudaMemcpyHostToDevice, stream_));
    }
    else if (mode == MemcpyKind::GPU2GPU)
    {
        assert_gpu();
        ASSERT_TRUE(0 < size <= gpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, gpu_ptr_, size, cudaMemcpyDeviceToDevice, stream_));
    }
    sync();
}

void Memory::to_cpu(void* out, size_t size, MemcpyKind mode)
{
    ASSERT_PTR(out);
    ASSERT_TRUE(mode == MemcpyKind::GPU2CPU || mode == MemcpyKind::CPU2CPU);
    if (mode == MemcpyKind::GPU2CPU)
    {
        assert_gpu();
        ASSERT_TRUE(0 <= size <= gpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, gpu_ptr_, size, cudaMemcpyDeviceToHost, stream_));
    }
    else if (mode == MemcpyKind::CPU2CPU)
    {
        assert_cpu();
        ASSERT_TRUE(0 <= size <= cpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, cpu_ptr_, size, cudaMemcpyHostToHost, stream_));
    }
    sync();
}

void Memory::sync()
{
    CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream_));
}

void Memory::assert_cpu()
{
    ASSERT_PTR(cpu_ptr_);
    ASSERT_TRUE(cpu_size_ != 0);
}

void Memory::assert_gpu()
{
    ASSERT_PTR(gpu_ptr_);
    ASSERT_TRUE(gpu_size_ != 0);
}

void* Memory::offset_cpu_ptr(size_t offset)
{
    assert_cpu();
    ASSERT_TRUE(offset <= cpu_size_);
    return (void*) ((char*) cpu_ptr_ + offset);
}

void* Memory::offset_gpu_ptr(size_t offset)
{
    assert_gpu();
    ASSERT_TRUE(offset <= gpu_size_);
    return (void*) ((char*) gpu_ptr_ + offset);
}

} // namespace trt