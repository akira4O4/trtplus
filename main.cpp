#include <opencv2/core/core.hpp>

#include "chrono"
#include "cpu/postprocess.h"
#include "cpu/preprocess.h"
#include "iostream"
#include "opencv2/opencv.hpp"
#include "src/memory.h"
#include "src/model.h"
#include "src/result.hpp"
#include "src/utils.h"
#include "unistd.h"
#include "vector"

int main(int argc, char const* argv[])
{
    auto task        = "";
    auto model_path  = "";
    auto images_dir  = "";
    auto output_dir  = "";
    auto device      = kDefaultDevice;
    auto batch       = 1;
    auto thr         = std::vector<float>{0};
    auto labels_file = "";
    auto mode        = kDefaultMode;

    std::cout << "Task        : " << task << std::endl;
    std::cout << "Model       : " << model_path << std::endl;
    std::cout << "Images Dir  : " << images_dir << std::endl;
    std::cout << "Output Dir  : " << output_dir << std::endl;
    std::cout << "Device      : " << device << std::endl;
    std::cout << "Batch       : " << batch << std::endl;
    std::cout << "Mode        : " << mode << std::endl;
    //-------------------------------------------------------------------------

    auto model = trt::Model(model_path, device, mode);
    model.init();

    if (model.is_dynamic())
    {
        model.decode_input();
        result::NCHW input = model.get_inputs()[ 0 ];
        input.info();
        nvinfer1::Dims4 dims(batch, input.c, input.h, input.w);
        model.set_binding_dims(input.name, dims);
    }

    auto stream = model.get_stream();
    model.decode_binding();

    result::NCHW input_shape  = model.get_inputs()[ 0 ];
    result::NCHW output_shape = model.get_outputs()[ 0 ];
    input_shape.info();
    output_shape.info();

    //----------------------------------------------------------------------
    size_t input_size  = input_shape.NxCxHxW(FLOAT32);
    size_t output_size = 0;

    if (task == "classification")
    {
        output_size = output_shape.NxC(FLOAT32);
    }
    else if (task == "segmentation")
    {
        output_size = output_shape.NxCxHxW(FLOAT32);
    }
    else
    {
        INFO("Error: Task must be [classification or segmentation]");
        return 1;
    }

    auto model_input_memory  = std::make_shared<trt::Memory>(input_shape.idx, input_size, true, stream);
    auto model_output_memory = std::make_shared<trt::Memory>(output_shape.idx, output_size, true, stream);

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

            out = cpu::resize(img, input_wh);
            out = cpu::image2rgb(out);
            out = cpu::normalize(out);

            size_t offset = n * input_shape.CxHxW();
            //            host_ptr += n * input_shape.CxHxW();
            cpu::hwc2chw_v1(out, host_ptr + offset);
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
        else
        {
            INFO("Done.");
        }

        //        for (int k = 0; k < input_shape.bs; k++)
        //        {
        //            //            output_host_ptr += k * nc;
        //            //            postprocess::softmax(output_host_ptr, cls_output, nc, true);
        //
        //            // TODO:Bad function
        //            postprocess::classification(output_host_ptr + k * nc, nc, thr, labels, output_dir);
        //        }
        //        std::cout << std::endl;

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}