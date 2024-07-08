#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <utils.h>

#include "iostream"

#define H2H cudaMemcpyHostToHost
#define D2D cudaMemcpyDeviceToDevice
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define ALIGN_SIZE 128

namespace trt
{

class Memory
{
  private:
    int          id_       = -1;
    void*        host_     = nullptr;
    void*        device_   = nullptr;
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

    ~Memory() { free_host_and_device_memory(); };

    inline int get_id() const { return id_; };
    inline int set_id(int id) { id_ = id; };

    size_t get_cpu_size() const { return cpu_size_; };
    size_t get_gpu_size() const { return gpu_size_; };

    void set_cpu_size(size_t num_of_byte) { cpu_size_ = num_of_byte; };
    void set_gpu_size(size_t num_of_byte) { gpu_size_ = num_of_byte; };

    cudaStream_t get_stream() { return stream_; };

    void set_stream(cudaStream_t stream) { stream_ = stream; };

    void* get_host_ptr() { return host_; };

    void* get_device_ptr() { return device_; };

    template <typename T>
    T* get_host_ptr()
    {
        return (T*) host_;
    };

    template <typename T>
    T* get_device_ptr()
    {
        return (T*) device_;
    };

    void malloc_host_memory();

    void malloc_device_memory();

    void malloc_host_memory(size_t num_of_byte);

    void malloc_device_memory(size_t num_of_byte);

    void malloc_host_and_device_memory();

    void to_cpu();

    void to_other_cpu(void* out, int size = -1, size_t offset = 0);

    void to_gpu();

    void to_other_gpu(void* out, int size = -1, size_t offset = 0);

    void cpu2cpu(void* out, int size = -1, size_t offset = 0);

    void gpu2gpu(void* out, int size = -1, size_t offset = 0);

    void free_device_memory();

    void free_host_memory();

    void free_host_and_device_memory();

    static float float16_to_float32(half value);

    static half float32_to_float16(float value);

    static size_t align_size(size_t sz, size_t n);

    void sync();

    void assert_host();

    void assert_device();

    void* offset_ptr(void* ptr, size_t offset);
};

} // namespace trt
#endif // MEMORY_MANAGER_H