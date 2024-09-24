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
    auto model_path  = "";
    auto images_dir  = "";
    auto labels_file = "";
    //-------------------------------------------------------------------------

    auto model = trt::Model(model_path, device);
    model.init();
    auto stream = model.get_stream();

    result::NCHW input_shape  = model.get_binding(0);
    result::NCHW output_shape = model.get_binding(1);
    input_shape.info();
    output_shape.info();

    //----------------------------------------------------------------------
    size_t input_size  = input_shape.NxCxHxW(kFLOAT32);
    size_t output_size = output_shape.NxC(kFLOAT32);

    auto model_input_memory  = std::make_shared<trt::Memory>(input_shape.idx, input_size, true, stream);
    auto model_output_memory = std::make_shared<trt::Memory>(output_shape.idx, output_size, true, stream);

    //-----------------------------------------------------------------------
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
        auto host_ptr = model_input_memory->get_cpu_ptr<float>();
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
        model_input_memory->to_gpu();
        void* bindings[ 2 ]{model_input_memory->get_gpu_ptr(), model_output_memory->get_gpu_ptr()};
        model.forward(bindings, stream, nullptr);
        model_output_memory->to_cpu();

        //--------------------------------------------------------------------------------------------------------------

        auto output_host_ptr = model_output_memory->get_cpu_ptr<float>();
        ASSERT_PTR(output_host_ptr);
        auto nc = output_shape.c;

        for (int k = 0; k < input_shape.bs; k++)
        {
            auto max_idx = cpu::argmax(output_host_ptr, nc);
            INFO("Prediction-> label: %s | score:%f", labels[ max_idx ].c_str(), output_host_ptr[ max_idx ]);
        }
        INFO("Done.\n");

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}