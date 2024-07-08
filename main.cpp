#include <opencv2/core/core.hpp>

#include "3rdparty/yaml-cpp/include/yaml.h"
#include "chrono"
#include "iostream"
#include "opencv2/opencv.hpp"
#include "src/config.h"
#include "src/memory.h"
#include "src/model.h"
#include "src/postprocess.h"
#include "src/prepprocess.h"
#include "src/result.hpp"
#include "src/utils.h"
#include "unistd.h"
#include "vector"

namespace nv = nvinfer1;

void mem_test()
{
    std::string images_dir = "/home/seeking/llf/code/tensorRT-lab/temp/images/imagenet";

    std::vector<cv::String> image_paths = get_image_paths(images_dir, "jpg");
    std::vector<cv::Mat>    images      = get_images(image_paths);

    std::vector<cv::Mat>    batch_images;
    std::vector<cv::String> batch_images_path;

    auto batch = 4;
    auto c     = 3;
    auto h     = 224;
    auto w     = 224;

    size_t img_size = c * h * w;
    size_t mem_size = batch * img_size * FLOAT32;

    trt::Memory mem_i(mem_size, true);
    trt::Memory mem_o(mem_size, true);

    for (int i = 0; i < images.size(); ++i)
    {
        INFO("Read Image: %s", image_paths.at(i).c_str());
        batch_images_path.emplace_back(image_paths.at(i));
        batch_images.emplace_back(images.at(i));

        if (batch_images.size() != batch)
            continue;

        for (int j = 0; j < batch; ++j)
        {

            cv::Mat img = batch_images.at(j);
            img         = preprocess::resize(img, cv::Size(w, h));
            std::cout << img.step[ 0 ] * img.rows << std::endl;
            std::cout << img_size << std::endl;
            img.convertTo(img, CV_32FC3, 1.f / 255.f);

            std::cout << img.isContinuous() << std::endl;
            std::cout << img.step[ 0 ] * img.rows << std::endl;
            std::cout << img_size << std::endl;

            //            size_t offset = j * c * h * w*FLOAT32;
            size_t offset = j * img_size;
            float* ptr    = mem_i.get_host_ptr<float>() + offset;
            std::memcpy(ptr, img.ptr<float>(0), img_size * FLOAT32);
        }

        mem_i.to_gpu();
        mem_i.gpu2gpu(mem_o.get_device_ptr());
        mem_o.to_cpu();

        for (int j = 0; j < batch; ++j)
        {
            size_t  offset = j * c * h * w;
            auto*   ptr    = mem_o.get_host_ptr<float>() + offset;
            cv::Mat out    = cv::Mat(h, w, CV_32FC3);
            std::memcpy(out.ptr<float>(0), ptr, img_size);
            out.convertTo(out, CV_8UC3, 255.0);

            std::string save_dir  = "/home/seeking/llf/code/tensorRT-lab/temp/output";
            auto        save_path = save_dir + "/bs" + std::to_string(i) + "_i" + std::to_string(j) + ".jpg";
            cv::imwrite(save_path, out);
        }
        batch_images.clear();
        batch_images_path.clear();
    }
}

int main(int argc, char const* argv[])
{
    Config config("/home/seeking/llf/code/tensorRT-lab/configs/config.yaml", true);
    auto   task        = config.get<std::string>("task");
    auto   model_name  = config.get<std::string>("model");
    auto   images_dir  = config.get<std::string>("images");
    auto   output_dir  = config.get<std::string>("output");
    auto   device      = config.get<int>("device");
    auto   batch       = config.get<int>("batch");
    auto   thr         = config.get<std::vector<float>>("thr");
    auto   labels_file = config.get<std::string>("labels_file");
    auto   mode        = config.get<std::string>("mode");

    std::cout << "Task        : " << task << std::endl;
    std::cout << "Model       : " << model_name << std::endl;
    std::cout << "Images Dir  : " << images_dir << std::endl;
    std::cout << "Output Dir  : " << output_dir << std::endl;
    std::cout << "Device      : " << device << std::endl;
    std::cout << "Batch       : " << batch << std::endl;
    std::cout << "Mode        : " << mode << std::endl;
    //-------------------------------------------------------------------------

    auto model = trt::Model(model_name, device, mode);
    model.init();

    if (model.is_dynamic())
    {
        model.decode_input();
        result::NCHW input = model.get_inputs()[ 0 ];
        input.info();
        nv::Dims4 dims(batch, input.c, input.h, input.w);
        model.set_binding_dims(input.name, dims);
    }

    auto stream = model.get_stream();
    model.decode_binding();

    result::NCHW input_shape  = model.get_inputs()[ 0 ];
    result::NCHW output_shape = model.get_outputs()[ 0 ];
    input_shape.info();
    output_shape.info();

    //----------------------------------------------------------------------
    size_t input_size          = input_shape.NxCxHxW(FLOAT32);
    size_t output_size         = output_shape.NxC(FLOAT32);
    auto   model_input_memory  = std::make_shared<trt::Memory>(input_shape.idx, input_size, true, stream);
    auto   model_output_memory = std::make_shared<trt::Memory>(output_shape.idx, output_size, true, stream);

    //-----------------------------------------------------------------------
    auto input_wh = cv::Size(input_shape.w, input_shape.h);

    std::vector<cv::String> image_paths = get_image_paths(images_dir, "jpg");
    std::vector<cv::Mat>    images      = get_images(image_paths);

    std::vector<cv::Mat>    batch_images;
    std::vector<cv::String> batch_images_path;

    std::vector<std::string> labels = load_txt(labels_file);
    for (int i = 0; i < images.size(); ++i)
    {

        INFO("Read Image: %s", image_paths.at(i).c_str());
        batch_images_path.emplace_back(image_paths.at(i));
        batch_images.emplace_back(images.at(i));

        if (batch_images.size() != input_shape.bs)
            continue;

        cv::Mat out;

        // pre-process
        auto host_ptr = model_input_memory->get_host_ptr<float>();
        for (int n = 0; n < input_shape.bs; ++n)
        {
            cv::Mat img = batch_images.at(n);

            out = preprocess::resize(img, input_wh);
            out = preprocess::bgr2rgb(out);
            out = preprocess::normalize_image(out);

            size_t offset = n * input_shape.CxHxW();
            //            host_ptr += n * input_shape.CxHxW();
            preprocess::hwc2chw(out, host_ptr + offset);
        }

        // Infer--------------------------------------------------------------------------------------------------------

        model_input_memory->to_gpu();
        void* bindings[ 2 ]{model_input_memory->get_device_ptr(), model_output_memory->get_device_ptr()};
        model.forward(bindings, stream, nullptr);
        model_output_memory->to_cpu();

        //--------------------------------------------------------------------------------------------------------------

        auto  output_host_ptr = model_output_memory->get_host_ptr<float>();
        auto  nc              = output_shape.c;
        float cls_output[ nc ];

        if (output_host_ptr == nullptr)
        {
            INFO("Output Is Null.");
            return 0;
        }

        for (int k = 0; k < input_shape.bs; k++)
        {
            //            output_host_ptr += k * nc;
            //            postprocess::softmax(output_host_ptr, cls_output, nc, true);
            postprocess::classification(output_host_ptr + k * nc, nc, thr, labels, output_dir);
        }
        std::cout << std::endl;

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}