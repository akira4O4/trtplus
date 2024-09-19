#include "NvInfer.h"
#include "fstream"
#include "iostream"
#include "memory"
#include <model.h>
#include <utility>
#include <utils.h>

namespace trt
{

Model::Model(const std::string& model_path)
{
    model_path_ = model_path;
}

Model::Model(const std::string& model_path, uchar device)
{
    model_path_ = model_path;
    device_     = device;
}

Model::~Model()
{
    reset();
}

void Model::init()
{

    INFO("Begin Init Model.");

    if (model_path_.empty())
    {
        INFO("Model path is empty -> exit(0)");
        std::exit(0);
    }

    create_stream();
    load_engine();
    build_model();
    decode_model_status();
    decode_model_bindings();

    INFO("Model Init Done.");
}

void Model::create_stream()
{
    CHECK_CUDA_RUNTIME(cudaStreamCreate(&stream_));
    assert(stream_);
    INFO("Init CUDA Stream");
}

void Model::load_engine()
{
    std::ifstream in(model_path_, std::ios::in | std::ios::binary);

    if (!in.is_open())
    {
        INFO("Model file is not open.");
        std::exit(-1);
    }

    in.seekg(0, std::ios::end);
    size_t file_length = in.tellg();
    if (file_length > 0)
    {
        in.seekg(0, std::ios::beg);
        data_.resize(file_length);
        in.read((char*) &data_[ 0 ], file_length);
    }
    in.close();
    INFO("Load engine file: %s", model_path_.c_str());
}

void Model::build_model()
{
    if (data_.data() == nullptr || data_.empty())
    {
        INFO("Engine_data == nullptr or engine_size == 0");
        return;
    }
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(NVLogger::instance()),
                                                   delete_pointer<nvinfer1::IRuntime>);
    assert(runtime_);

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(data_.data(), data_.size(), nullptr), delete_pointer<nvinfer1::ICudaEngine>);
    assert(engine_);

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                            delete_pointer<nvinfer1::IExecutionContext>);
    assert(context_);
}

void Model::decode_model_status()
{

#if NV_TENSORRT_MINOR < 5
    num_of_bindings_ = engine_->getNbBindings();
    auto data_type   = engine_->getBindingDataType(0);

#elif NV_TENSORRT_MINOR >= 5
    num_of_bindings_ = engine_->getNbIOTensors();
    auto data_type   = engine_->getTensorDataType(engine_->getIOTensorName(0));
#endif

    if (data_type == nvinfer1::DataType::kFLOAT)
        is_fp32_ = true;
    else if (data_type == nvinfer1::DataType::kHALF)
        is_fp16_ = true;
    else if (data_type == nvinfer1::DataType::kINT8)
        is_int8_ = true;

    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto dims = get_binding_dims(i);
        if (dims.d[ 0 ] == -1)
        {
            is_dynamic_ = true;
            break;
        }
    }
};

bool Model::forward(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
{
    return context_->enqueueV2(bindings, stream, inputConsumed);
}

void Model::reset()
{
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

nvinfer1::Dims Model::get_binding_dims(uchar index)
{
#if NV_TENSORRT_MINOR < 5
    return context_->getBindingDimensions(index);
#elif NV_TENSORRT_MINOR >= 5
    return context_->getTensorShape(idx2name(index));
#endif
}

bool Model::set_binding_dims(uchar index, nvinfer1::Dims dims)
{
    auto item = inputs_.find(index);
    if (item == inputs_.end())
    {
        return false;
    }
    else
    {
        auto name = idx2name(item->first);

#if NV_TENSORRT_MINOR < 5
        return context_->setBindingDimensions(index, dims);
#elif NV_TENSORRT_MINOR >= 5
        return context_->setInputShape(name, dims);
#endif
    }
}

void Model::set_device(uchar device)
{
    assert(device >= 0);
    device_ = device;
    cudaSetDevice(device_);
    INFO("Set Device: %d.", device_);
}

#if NV_TENSORRT_MINOR < 5
void Model::decode_model_inputs()
{
    inputs_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        if (engine_->bindingIsInput(i))
        {
            auto         name = engine_->getBindingName(i);
            auto         dims = context_->getBindingDimensions(i);
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            inputs_[ i ]      = nchw;
        }
    }
}

#elif NV_TENSORRT_MINOR >= 5
void Model::decode_model_inputs()
{
    inputs_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name  = engine_->getIOTensorName(i);
        auto dims  = context_->getTensorShape(name);
        auto mode  = engine_->getTensorIOMode(name);
        auto input = nvinfer1::TensorIOMode::kINPUT;
        if (mode == input)
        {
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            inputs_[ i ]      = nchw;
        }
    }
}
#endif

#if NV_TENSORRT_MINOR < 5
void Model::decode_model_outputs()
{
    outputs_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        if (!engine_->bindingIsInput(i))
        {

            auto         name = engine_->getBindingName(i);
            auto         dims = context_->getBindingDimensions(i);
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            outputs_[ i ]     = nchw;
        }
    }
}
#elif NV_TENSORRT_MINOR >= 5
void Model::decode_model_outputs()
{
    outputs_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = context_->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            outputs_[ i ]     = nchw;
        }
    }
}
#endif
void Model::decode_model_bindings()
{
    decode_model_inputs();
    decode_model_outputs();
}

char const* Model::idx2name(uchar index)
{
#if NV_TENSORRT_MINOR < 5

#elif NV_TENSORRT_MINOR >= 5
    return engine_->getIOTensorName(index);
#endif
}

void Model::set_input_dims(uchar index, nvinfer1::Dims dims)
{
    if (is_dynamic_)
    {
        assert(set_binding_dims(index, dims));
        decode_model_inputs();
    }
    else
    {
        INFO("Model is static. can not set the input dims.");
    }
}

void Model::show_model_info()
{
    INFO("Model path: %s", model_path_.c_str());
    INFO("Device: %d", device_);

    if (is_dynamic_)
        INFO("Model is dynamic");
    else
        INFO("Model is static");

    if (is_int8_)
        INFO("Model Int8: True");
    else if (is_fp16_)
        INFO("Model FP16: True");
    else if (is_fp32_)
        INFO("Model FP32: True");

    INFO("Num of bindings: %d", num_of_bindings_);
    for (const auto& item : inputs_)
        item.second.info();
    for (const auto& item : outputs_)
        item.second.info();
}

} // namespace trt
