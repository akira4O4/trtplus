//
// Created by seeking on 9/23/24.
//
#include "iostream"
#include "src/cpu/postprocess.h"
#include "src/cpu/preprocess.h"
#include "src/memory.h"
#include "src/model.h"
#include "src/result.hpp"
#include "src/utils.h"
#include "unistd.h"
#include "vector"
#include <opencv2/core/core.hpp>

int main(int argc, char const* argv[])
{
    auto device      = kDefaultDevice;
    auto model_path  = "/home/seeking/llf/code/trtplus/assets/D2/models/1x3x256x256.fp32.static.engine";
    auto images_dir  = "/home/seeking/llf/code/trtplus/assets/D2/images";
    auto labels_file = "/home/seeking/llf/code/trtplus/assets/D2/label.txt";
    //-------------------------------------------------------------------------

    auto model = trt::Model(model_path, device);
    model.init();
    auto stream = model.get_stream();

    nvinfer1::Dims input_dims  = model.get_binding_dims(0);
    nvinfer1::Dims output_dims = model.get_binding_dims(1);

    result::NCHW input_shape = {"input", 0, input_dims.d[ 0 ], input_dims.d[ 1 ], input_dims.d[ 2 ], input_dims.d[ 3 ]};
    result::YOLOv8Output output_shape = {"output", 1, output_dims.d[ 0 ], output_dims.d[ 1 ], output_dims.d[ 2 ]};
    input_shape.info();
    output_shape.info();

    size_t input_mem_size  = dims_volume(input_dims) * kFLOAT32;
    size_t output_mem_size = dims_volume(output_dims) * kFLOAT32;
    INFO("Input memory size: %d Byte", input_mem_size);
    INFO("Output memory size: %d Byte", output_mem_size);

    auto input_memory  = std::make_shared<trt::Memory>(0, input_mem_size, true, stream);
    auto output_memory = std::make_shared<trt::Memory>(0, output_mem_size, true, stream);

    auto input_wh = cv::Size(input_shape.w, input_shape.h);

    std::vector<cv::String> image_paths = get_image_paths(images_dir, "png");
    std::vector<cv::Mat>    images      = get_images(image_paths);

    std::vector<cv::Mat>    batch_images;
    std::vector<cv::String> batch_images_path;

    std::vector<std::string> labels = load_label_from_txt(labels_file);
    for (int i = 0; i < images.size(); ++i)
    {

        INFO("Read Image: %s", image_paths.at(i).c_str());
        batch_images_path.emplace_back(image_paths.at(i));
        batch_images.emplace_back(images.at(i));

        if (batch_images.size() != input_shape.bs)
            continue;

        cv::Mat out;

        // pre-process
        auto host_ptr = input_memory->get_cpu_ptr<float>();
        for (int n = 0; n < input_shape.bs; ++n)
        {
            cv::Mat img = batch_images.at(n);

            out = cpu::resize(img, input_wh);
            out = cpu::bgr2rgb(out);
            out = cpu::normalize(out);

            size_t offset = n * input_shape.CxHxW();
            cpu::hwc2chw(out, host_ptr + offset);
        }

        // Infer--------------------------------------------------------------------------------------------------------
        input_memory->to_gpu();
        void* bindings[ 2 ]{input_memory->get_gpu_ptr(), output_memory->get_gpu_ptr()};
        model.forward(bindings, stream, nullptr);
        output_memory->to_cpu();

        //--------------------------------------------------------------------------------------------------------------

        auto output_host_ptr = output_memory->get_cpu_ptr<float>();
        ASSERT_PTR(output_host_ptr);
        auto nc = labels.size();

        for (int k = 0; k < input_shape.bs; k++)
        {
            size_t offset = k * output_shape.rows * output_shape.dimensions;
            //            auto   max_idx = cpu::argmax(output_host_ptr + offset, nc);
            //            INFO("Prediction-> label: %s | score:%f", labels[ max_idx ].c_str(), output_host_ptr[ max_idx
            //            ]);
        }
        INFO("Done.\n");

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}