#ifndef ENGINE_H
#define ENGINE_H

#include <utility>

#include "NvInfer.h"
#include "fstream"
#include "iostream"
#include "map"
#include "memory"
#include "result.hpp"
#include "utils.h"
#include "vector"

namespace trt
{

namespace nv = nvinfer1;

//-----------------------------------------------------------

template <typename T>
void delete_pointer(T* ptr)
{
    delete ptr;
}

struct PointerDeleter
{
    template <typename T>
    void operator()(T* ptr) const
    {
        delete ptr;
    }
};

template <typename T>
using make_unique = std::unique_ptr<T, PointerDeleter>;

template <typename T>
std::shared_ptr<T> make_shared(T* ptr)
{
    return std::shared_ptr<T>(ptr, PointerDeleter());
}

//-----------------------------------------------------------

class NVLogger : public nv::ILogger
{
  public:
    static NVLogger& instance()
    {
        static NVLogger instance;
        return instance;
    }

  private:
    NVLogger() {}

    ~NVLogger() override = default;

    using Severity = nv::ILogger::Severity;

    void log(Severity severity, const char* msg) noexcept override
    {
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

//-----------------------------------------------------------

class Model
{
  private:
    bool fp16_   = false;
    bool int8_   = false;
    int  device_ = 0;

    cudaStream_t stream_ = nullptr;

    int  num_of_bindings_{0};
    bool is_dynamic_{false};

    std::string model_path_;

    std::vector<uint8_t> data_{};

    std::shared_ptr<nv::IRuntime>          runtime_ = nullptr;
    std::shared_ptr<nv::ICudaEngine>       engine_  = nullptr;
    std::shared_ptr<nv::IExecutionContext> context_ = nullptr;

    std::vector<result::NCHW> inputs_{};
    std::vector<result::NCHW> output_{};

  public:
    Model() = default;
    ~Model();

    explicit Model(std::string engine_path) { model_path_ = std::move(engine_path); };

    explicit Model(std::string model_path, int device, const std::string& mode);

    void init();

    void load_engine();

    void build_model();

    void create_stream();

    void check_dynamic();

    inline std::vector<result::NCHW> get_inputs() { return inputs_; };

    inline std::vector<result::NCHW> get_outputs() { return output_; };

    inline void set_model_path(std::string path) { model_path_ = std::move(path); };

    inline cudaStream_t get_stream() { return stream_; };

    inline void set_stream(cudaStream_t stream) { stream_ = stream; };

    inline int get_device() const { return device_; };

    void set_device(int device);

    void reset();

    static std::vector<int> dims_to_vector(nv::Dims dims);

    static nv::Dims vector_to_dims(const std::vector<int>& data);

    bool set_binding_dims(const std::string& name, nv::Dims dims);

    nv::Dims get_binding_dims(const std::string& name);

    nv::DataType get_binding_datatype(const std::string& name);

    bool forward(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed);

    inline int get_num_of_binding() const { return num_of_bindings_; };

    inline bool is_dynamic() const { return is_dynamic_; };

    inline void use_fp16()
    {
        fp16_ = true;
        int8_ = false;
    };

    inline void use_int8()
    {
        fp16_ = false;
        int8_ = true;
    };

    static void print_dims(nv::Dims dims);

    void decode_input();

    void decode_output();

    void decode_binding();
};

} // namespace trt
#endif