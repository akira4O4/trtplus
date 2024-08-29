#ifndef MODEL_H
#define MODEL_H

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
    bool use_fp16_ = false;
    bool use_int8_ = false;

    uchar device_          = 0;
    bool  is_int8_         = false;
    bool  is_fp16_         = false;
    bool  is_fp32_         = true;
    bool  is_dynamic_      = false;
    uchar num_of_bindings_ = 0;

    std::string          model_path_;
    std::vector<uint8_t> data_{};

    cudaStream_t                           stream_  = nullptr;
    std::shared_ptr<nv::IRuntime>          runtime_ = nullptr;
    std::shared_ptr<nv::ICudaEngine>       engine_  = nullptr;
    std::shared_ptr<nv::IExecutionContext> context_ = nullptr;

    std::vector<result::NCHW> inputs_{};
    std::vector<result::NCHW> output_{};

  public:
    Model() = default;
    ~Model();

    explicit Model(const std::string& model_path, int device, const std::string& mode);

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

    inline int get_num_of_binding() const { return num_of_bindings_; };

    inline bool is_dynamic() const { return is_dynamic_; };

    void set_device(uchar device);

    void reset();

    bool set_binding_dims(const std::string& name, nv::Dims dims);

    bool set_binding_dims(uchar index, nv::Dims dims);

    nv::Dims get_binding_dims(const std::string& name);

    nv::Dims get_binding_dims(uchar index);

    bool forward(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed);

    inline void use_fp16()
    {
        use_fp16_ = true;
        use_int8_ = false;
    };

    inline void use_int8()
    {
        use_fp16_ = false;
        use_int8_ = true;
    };

    void decode_model_input();

    void decode_model_output();

    void decode_model_binding();

    void check_model_type();

    void set_model_input_shape(nvinfer1::Dims dims);
};

} // namespace trt
#endif