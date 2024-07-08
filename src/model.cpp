#include <utility>

#include "NvInfer.h"
#include "config.h"
#include "fstream"
#include "iostream"
#include "memory"
#include "vector"
#include <model.h>
#include <utils.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace trt
{
namespace nv = nvinfer1;

Model::Model(std::string model_path, int device, const std::string &mode)
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
        in.read((char *) &data_[ 0 ], file_length);
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

void Model::check_dynamic()
{
    for (int i = 0; i < num_of_bindings_; ++i)
    {
        auto name = engine_->getIOTensorName(i);
        auto dims = get_binding_dims(name);
        if (dims.d[ 0 ] == -1)
            is_dynamic_ = true;
    }

    if (is_dynamic_)
        INFO("Model Is Dynamic.");
    else
        INFO("Model Is Static.");
}

std::vector<int> Model::dims_to_vector(const nv::Dims dims)
{
    std::vector<int> vec(dims.d, dims.d + dims.nbDims);
    return vec;
}

nv::Dims Model::vector_to_dims(const std::vector<int> &data)
{
    nv::Dims d;
    std::memcpy(d.d, data.data(), sizeof(int) * data.size());
    d.nbDims = data.size();
    return d;
}

bool Model::forward(void *const *bindings, cudaStream_t stream, cudaEvent_t *inputConsumed)
{
    return context_->enqueueV2(bindings, stream, inputConsumed);
}

void Model::reset()
{
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

nv::Dims Model::get_binding_dims(const std::string &name)
{
    return context_->getTensorShape(name.c_str());
}

bool Model::set_binding_dims(const std::string &name, nv::Dims dims)
{
    return context_->setInputShape(name.c_str(), dims);
}

nv::DataType Model::get_binding_datatype(const std::string &name)
{
    return engine_->getTensorDataType(name.c_str());
}

void Model::print_dims(nv::Dims dims)
{
    std::cout << "[ ";

    for (int i = 0; i < dims.nbDims; i++)
    {
        std::cout << dims.d[ i ] << " ";
    }
    std::cout << " ]\n";
}

void Model::set_device(const int device)
{
    device_ = device;
    cudaSetDevice(device_);
    INFO("Set Device: %d.", device_);
}

void Model::decode_input()
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

void Model::decode_output()
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

void Model::decode_binding()
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
