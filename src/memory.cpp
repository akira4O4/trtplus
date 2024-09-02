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
    gpu_size_ = 0;
    gpu_ptr_  = nullptr;
    INFO("Free Device Memory.");
}

void Memory::free_cpu_memory()
{
    CHECK_CUDA_RUNTIME(cudaFreeHost(cpu_ptr_));
    cpu_size_ = 0;
    cpu_ptr_  = nullptr;
    INFO("Free Host Memory.");
}

void Memory::free_all()
{
    free_gpu_memory();
    free_cpu_memory();
}

// inner cpu -> inner gpu
void Memory::to_gpu()
{
    assert_cpu();
    assert_gpu();
    ASSERT_OP(cpu_size_ == gpu_size_);
    CHECK_CUDA_RUNTIME(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, cpu_size_, cudaMemcpyHostToDevice, stream_));
    sync();
}

// inner gpu-> inner cpu
void Memory::to_cpu()
{
    assert_cpu();
    assert_gpu();
    ASSERT_OP(cpu_size_ == gpu_size_);
    CHECK_CUDA_RUNTIME(cudaMemcpyAsync(cpu_ptr_, gpu_ptr_, gpu_size_, cudaMemcpyDeviceToHost, stream_));
    sync();
}

// inner cpu -> gpu
// inner gpu-> gpu
void Memory::to_gpu(void* out, size_t size, MemcpyKind mode)
{
    ASSERT_PTR(out);
    if (mode == MemcpyKind::CPU2GPU)
    {
        assert_cpu();
        ASSERT_OP(0 < size <= cpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, cpu_ptr_, size, cudaMemcpyHostToDevice, stream_));
    }
    else if (mode == MemcpyKind::GPU2GPU)
    {
        assert_gpu();
        ASSERT_OP(0 < size <= gpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, gpu_ptr_, size, cudaMemcpyDeviceToDevice, stream_));
    }
    sync();
}
// inner gpu->cpu
// inner cpu->cpu
void Memory::to_cpu(void* out, size_t size, MemcpyKind mode)
{
    ASSERT_PTR(out);
    if (mode == MemcpyKind::GPU2CPU)
    {
        assert_gpu();
        ASSERT_OP(0 <= size <= gpu_size_);
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(out, gpu_ptr_, size, cudaMemcpyDeviceToHost, stream_));
    }
    else if (mode == MemcpyKind::CPU2CPU)
    {
        assert_cpu();
        ASSERT_OP(0 <= size <= cpu_size_);
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
    ASSERT_OP(cpu_size_ != 0);
}

void Memory::assert_gpu()
{
    ASSERT_PTR(gpu_ptr_);
    ASSERT_OP(gpu_size_ != 0);
}

void* Memory::offset_cpu_ptr(size_t offset)
{
    assert_cpu();
    ASSERT_OP(offset <= cpu_size_);
    return (void*) ((char*) cpu_ptr_ + offset);
}

void* Memory::offset_gpu_ptr(size_t offset)
{
    assert_gpu();
    ASSERT_OP(offset <= gpu_size_);
    return (void*) ((char*) gpu_ptr_ + offset);
}

size_t Memory::align_size(size_t sz, size_t n)
{
    ASSERT_OP(sz != 0);
    ASSERT_OP(n != 0);
    return (sz + n - 1) & -n;
}

} // namespace trt