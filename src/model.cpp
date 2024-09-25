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
    ASSERT_PTR(stream_);
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
    ASSERT_PTR(runtime_);

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(data_.data(), data_.size(), nullptr), delete_pointer<nvinfer1::ICudaEngine>);
    ASSERT_PTR(engine_);

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                            delete_pointer<nvinfer1::IExecutionContext>);
    ASSERT_PTR(context_);
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
// TODO:FIX:TRT_DEPRECATED FUNCTION
bool Model::forward(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
{

    // #if NV_TENSORRT_MINOR < 5
    return context_->enqueueV2(bindings, stream, inputConsumed);
    // #elif NV_TENSORRT_MINOR >= 5
    //     return context_->enqueueV3(stream_);
    // #endif
}

void Model::reset()
{
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

nvinfer1::Dims Model::get_binding_dims(uchar binding_index)
{
#if NV_TENSORRT_MINOR < 5
    return context_->getBindingDimensions(index);
#elif NV_TENSORRT_MINOR >= 5
    return context_->getTensorShape(idx2name(binding_index));
#endif
}

bool Model::set_binding_dims(uchar binding_index, nvinfer1::Dims dims)
{
    auto item = inputs_.find(binding_index);
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
    ASSERT_TRUE(device >= 0);
    device_ = device;
    cudaSetDevice(device_);
    INFO("Set Device: %d.", device_);
}

#if NV_TENSORRT_MINOR < 5
void Model::decode_model_bindings()
{
    inputs_.clear();
    outputs_.clear();
    bindings_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        if (engine_->bindingIsInput(i))
        {
            auto name      = engine_->getBindingName(i);
            auto dims      = context_->getBindingDimensions(i);
            inputs_[ i ]   = dims;
            bindings_[ i ] = dims;
        }
        else
        {

            auto name      = engine_->getBindingName(i);
            auto dims      = context_->getBindingDimensions(i);
            outputs_[ i ]  = dims;
            bindings_[ i ] = dims;
        }
    }
}
#elif NV_TENSORRT_MINOR >= 5
void Model::decode_model_bindings()
{
    bindings_.clear();
    inputs_.clear();
    outputs_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = context_->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            inputs_[ i ]   = dims;
            bindings_[ i ] = dims;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputs_[ i ]  = dims;
            bindings_[ i ] = dims;
        }
    }
}
#endif

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
            //            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            inputs_[ i ] = dims;
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
            //            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            outputs_[ i ] = dims;
        }
    }
}
#endif

char const* Model::idx2name(uchar binding_index)
{
#if NV_TENSORRT_MINOR < 5
    return engine_->getBindingName(index);
#elif NV_TENSORRT_MINOR >= 5
    return engine_->getIOTensorName(binding_index);
#endif
}

} // namespace trt