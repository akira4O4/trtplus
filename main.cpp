#include "NvInfer.h"
#include "chrono"
#include "cuda_runtime.h"
#include "iostream"
#include "src/cpu/postprocess.h"
#include "src/cpu/preprocess.h"
#include "src/memory.h"
#include "src/model.h"
#include "src/precision.h"
#include "src/utils.h"
#include "unistd.h"
#include "vector"
#include <opencv2/core/core.hpp>

int main()
{
    std::string model_path  = "/home/seeking/llf/code/trtplus/assets/mnist/mnist-1x3x28x28.static.fp32.static.engine";
    std::string images_dir  = "/home/seeking/llf/code/trtplus/assets/mnist/test";
    std::string labels_file = "/home/seeking/llf/code/trtplus/assets/mnist/mnist-lables.txt";
    int         device      = kDefaultDevice;

    // Prepare Model ---------------------------------------------------------------------------------------------------
    auto model = trt::Model(model_path, device);
    model.init();
    model.show_model_info();

    // Prepare input and output memory ---------------------------------------------------------------------------------
    result::NCHW input_shape  = model.get_input(0);
    result::NCHW output_shape = model.get_output(1);

    size_t input_size  = input_shape.NxCxHxW(kFLOAT32);
    size_t output_size = output_shape.NxC(kFLOAT32);

    auto stream              = model.get_stream();
    auto model_input_memory  = std::make_shared<trt::Memory>(input_shape.idx, input_size, true, stream);
    auto model_output_memory = std::make_shared<trt::Memory>(output_shape.idx, output_size, true, stream);

    // Prepare images
    // ---------------------------------------------------------------------------------------------------

    auto input_wh = cv::Size(input_shape.w, input_shape.h);

    std::vector<cv::String> image_paths = get_image_paths(images_dir, "png");
    std::vector<cv::Mat>    images      = get_images(image_paths);

    std::vector<cv::Mat>     batch_images;
    std::vector<cv::String>  batch_images_path;
    std::vector<std::string> labels = load_label_from_txt(labels_file);

    for (int i = 0; i < images.size(); ++i)
    {

        INFO("Read Image: %s", image_paths.at(i).c_str());
        batch_images_path.emplace_back(image_paths.at(i));
        batch_images.emplace_back(images.at(i));

        if (batch_images.size() != input_shape.bs)
            continue;

        cv::Mat out;

        // Pre-process
        //[CxHxW][CxHxW][CxHxW]....
        auto host_ptr = model_input_memory->get_cpu_ptr<float>();
        for (int n = 0; n < input_shape.bs; ++n)
        {
            cv::Mat img = batch_images.at(n);

            out = cpu::resize(img, input_wh);
            out = cpu::image2rgb(out);
            out = cpu::normalize(out); // 1/255

            // change channel
            size_t offset = n * input_shape.CxHxW();
            cpu::hwc2chw_v1(out, host_ptr + offset);
        }

        // Forward -----------------------------------------------------------------------------------------------------
        model_input_memory->to_gpu(); // move data from cpu to gpu
        void* bindings[ 2 ]{model_input_memory->get_gpu_ptr(), model_output_memory->get_gpu_ptr()};
        model.forward(bindings, stream, nullptr);
        model_output_memory->to_cpu();

        //--------------------------------------------------------------------------------------------------------------

        auto output_cpu_ptr = model_output_memory->get_cpu_ptr<float>();
        ASSERT_PTR(output_cpu_ptr);
        auto nc = output_shape.c;

        for (int n = 0; n < input_shape.bs; ++n)
        {
            size_t             offset     = n * nc;
            std::vector<float> cls_output = cpu::softmax(output_cpu_ptr + offset, nc, true);
            size_t             max_index  = cpu::argmax(cls_output);

            INFO("Prediction >>> %s:%f", labels[ max_index ].c_str(), cls_output[ max_index ]);
        }
        INFO("Infer Done.\n");

        batch_images.clear();
        batch_images_path.clear();
    }
}