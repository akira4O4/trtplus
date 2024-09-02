#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "iostream"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <utils.h>

namespace trt
{

constexpr uchar kAlignSize = 128;

enum class MemcpyKind
{
    CPU2CPU = 0,
    CPU2GPU = 1,
    GPU2CPU = 2,
    GPU2GPU = 3
};

class Memory
{
  private:
    int          id_       = -1;
    void*        cpu_ptr_  = nullptr;
    void*        gpu_ptr_  = nullptr;
    size_t       cpu_size_ = 0;
    size_t       gpu_size_ = 0;
    cudaStream_t stream_   = nullptr;

  public:
    Memory() = default;

    explicit Memory(size_t num_of_byte, bool alloc = false);

    explicit Memory(int id, size_t num_of_byte, bool alloc = false);

    explicit Memory(int id, size_t num_of_byte, bool alloc = false, cudaStream_t stream = 0);

    Memory(const Memory&) = delete;

    Memory& operator=(const Memory&) = delete;

    ~Memory() { free_all(); };

    inline int get_id() const { return id_; };
    inline int set_id(int id) { id_ = id; };

    size_t get_cpu_size() const { return cpu_size_; };
    size_t get_gpu_size() const { return gpu_size_; };

    void set_cpu_size(size_t num_of_byte) { cpu_size_ = num_of_byte; };
    void set_gpu_size(size_t num_of_byte) { gpu_size_ = num_of_byte; };

    cudaStream_t get_stream() { return stream_; };
    void         set_stream(cudaStream_t stream) { stream_ = stream; };

    void* get_cpu_ptr() { return cpu_ptr_; };
    void* get_gpu_ptr() { return gpu_ptr_; };

    template <typename T>
    T* get_cpu_ptr()
    {
        return (T*) cpu_ptr_;
    };

    template <typename T>
    T* get_gpu_ptr()
    {
        return (T*) gpu_ptr_;
    };

    void malloc_cpu_memory();

    void malloc_gpu_memory();

    void malloc_cpu_memory(size_t num_of_byte);

    void malloc_gpu_memory(size_t num_of_byte);

    void malloc_cpu_and_gpu_memory();

    void to_cpu();

    void to_gpu();

    void to_cpu(void* out, size_t size, MemcpyKind mode);

    void to_gpu(void* out, size_t size, MemcpyKind mode);

    void free_gpu_memory();

    void free_cpu_memory();

    void free_all();

    static size_t align_size(size_t sz, size_t n);

    void sync();

    void assert_cpu();

    void assert_gpu();

    void* offset_cpu_ptr(size_t offset);

    void* offset_gpu_ptr(size_t offset);
};

} // namespace trt
#endif // MEMORY_MANAGER_H