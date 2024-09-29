#ifndef MODEL_H
#define MODEL_H

#include "NvInfer.h"
#include "fstream"
#include "iostream"
#include "map"
#include "memory"
#include "unordered_map"
#include "utils.h"
#include "vector"
#include <utility>

namespace trt
{

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

class NVLogger : public nvinfer1::ILogger
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

    using Severity = nvinfer1::ILogger::Severity;

    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity)
            return;

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

  private:
    Severity reportableSeverity = Severity::kWARNING;
};

//-----------------------------------------------------------

class Model
{
  private:
    uchar device_          = 0;
    bool  is_int8_         = false;
    bool  is_fp16_         = false;
    bool  is_fp32_         = true;
    bool  is_dynamic_      = false;
    uchar num_of_bindings_ = 0;

    std::string          model_path_;
    std::vector<uint8_t> data_{};

    cudaStream_t                                 stream_  = nullptr;
    std::shared_ptr<nvinfer1::IRuntime>          runtime_ = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_  = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

    std::unordered_map<uchar, nvinfer1::Dims> bindings_{};
    std::unordered_map<uchar, nvinfer1::Dims> inputs_{};
    std::unordered_map<uchar, nvinfer1::Dims> outputs_{};

  public:
    Model() = default;
    ~Model();

    explicit Model(const std::string& model_path);

    explicit Model(const std::string& model_path, uchar device);

    void init();

    void load_engine();

    void build_model();

    void create_stream();

    inline nvinfer1::Dims get_input(uchar binding_index) { return inputs_[ binding_index ]; };

    inline nvinfer1::Dims get_output(uchar binding_index) { return outputs_[ binding_index ]; };

    inline nvinfer1::Dims get_binding(uchar binding_index) { return bindings_[ binding_index ]; };

    inline std::unordered_map<uchar, nvinfer1::Dims> get_inputs() const { return inputs_; };

    inline std::unordered_map<uchar, nvinfer1::Dims> get_outputs() const { return outputs_; };

    inline std::unordered_map<uchar, nvinfer1::Dims> get_bindings() const { return bindings_; };

    inline uchar get_input_size() const { return inputs_.size(); };

    inline uchar get_output_size() const { return outputs_.size(); };

    inline void set_model_path(const std::string& path) { model_path_ = path; };

    inline cudaStream_t get_stream() const { return stream_; };

    inline void set_stream(cudaStream_t stream) { stream_ = stream; };

    inline int get_device() const { return device_; };

    void set_device(uchar device);

    inline int get_num_of_binding() const { return num_of_bindings_; };

    inline bool is_dynamic() const { return is_dynamic_; };

    void reset();

    nvinfer1::Dims get_binding_dims(uchar binding_index);

    bool set_binding_dims(uchar binding_index, nvinfer1::Dims dims);

    bool forward(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed);

    DEPRECATED
    void decode_model_inputs();

    DEPRECATED
    void decode_model_outputs();

    void decode_model_bindings();

    char const* idx2name(uchar binding_index);

    void decode_model_status();
};

} // namespace trt
#endif