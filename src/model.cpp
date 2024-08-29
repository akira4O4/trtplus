#include "NvInfer.h"
#include "fstream"
#include "iostream"
#include "memory"
#include "vector"
#include <model.h>
#include <utility>
#include <utils.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace trt
{
namespace nv = nvinfer1;

Model::Model(const std::string& model_path, int device, const std::string& mode)
{
    model_path_ = std::move(model_path);
    device_     = device;

    if (mode == "fp16")
        use_fp16();
    else if (mode == "int8")
        use_int8();
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
    check_dynamic();

    INFO("Model Init Done.");
}

void Model::create_stream()
{
    checkRuntime(cudaStreamCreate(&stream_));
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
    runtime_ =
        std::shared_ptr<nv::IRuntime>(nv::createInferRuntime(NVLogger::instance()), delete_pointer<nv::IRuntime>);
    assert(runtime_);

    engine_ = std::shared_ptr<nv::ICudaEngine>(runtime_->deserializeCudaEngine(data_.data(), data_.size(), nullptr),
                                               delete_pointer<nv::ICudaEngine>);
    assert(engine_);

    context_ = std::shared_ptr<nv::IExecutionContext>(engine_->createExecutionContext(),
                                                      delete_pointer<nv::IExecutionContext>);
    assert(context_);

    num_of_bindings_ = engine_->getNbIOTensors();

    set_device(device_);

    INFO("Build Model Done.");
}

void Model::check_dynamic(check_dynamic)
{
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto dims = get_binding_dims(i);
        if (dims.d[ 0 ] == -1)
        {
            is_dynamic_ = true;
            break;
        }
    }
}

void Model::check_model_type()
{
    auto name = engine_->getIOTensorName(0);
    if (engine_->getTensorDataType(name) == nvinfer1::DataType::kFLOAT)
        is_fp32_ = true;
    else if (engine_->getTensorDataType(name) == nvinfer1::DataType::kHALF)
        is_fp16_ = true;
    else if (engine_->getTensorDataType(name) == nvinfer1::DataType::kINT8)
        is_int8_ = true;
}

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

nv::Dims Model::get_binding_dims(const std::string& name)
{
    return context_->getTensorShape(name.c_str());
}

nv::Dims Model::get_binding_dims(uchar index)
{
    auto name = engine_->getIOTensorName(index);
    return context_->getTensorShape(name);
}

bool Model::set_binding_dims(const std::string& name, nv::Dims dims)
{
    return context_->setInputShape(name.c_str(), dims);
}

bool Model::set_binding_dims(uchar index, nv::Dims dims)
{
    auto name = engine_->getIOTensorName(index);
    return context_->setInputShape(name, dims);
}

void Model::set_device(uchar device)
{
    device_ = device;
    cudaSetDevice(device_);
    INFO("Set Device: %d.", device_);
}

void Model::decode_model_input()
{
    inputs_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = context_->getTensorShape(name);
        if (mode == nv::TensorIOMode::kINPUT)
        {
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            inputs_.emplace_back(nchw);
        }
    }
    INFO("Num Of Input: %d", inputs_.size());
}

void Model::decode_model_output()
{
    output_.clear();
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = context_->getTensorShape(name);
        if (mode == nv::TensorIOMode::kOUTPUT)
        {
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            output_.emplace_back(nchw);
        }
    }
    INFO("Num Of Output: %d", output_.size());
}

void Model::decode_model_binding()
{
    inputs_.clear();
    output_.clear();

    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = context_->getTensorShape(name);

        if (mode == nv::TensorIOMode::kINPUT)
        {
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            inputs_.emplace_back(nchw);
        }
        else if (mode == nv::TensorIOMode::kOUTPUT)
        {
            result::NCHW nchw = {name, i, dims.d[ 0 ], dims.d[ 1 ], dims.d[ 2 ], dims.d[ 3 ]};
            output_.emplace_back(nchw);
        }
    }
    INFO("Num Of Input  : %d", inputs_.size());
    INFO("Num Of Output : %d", output_.size());
}

} // namespace trt
