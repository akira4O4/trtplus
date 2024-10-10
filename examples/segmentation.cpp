//
// Created by seeking on 9/23/24.
//
#include "iostream"
#include "src/cpu/postprocess.h"
#include "src/cpu/preprocess.h"
#include "src/io.h"
#include "src/memory.h"
#include "src/model.h"
#include "src/utils.h"
#include "unistd.h"
#include "vector"
#include <opencv2/core/core.hpp>

int main(int argc, char const* argv[])
{

    uchar            device              = 0;
    cv::Scalar       mean                = {0.485, 0.456, 0.406}; // RGB
    cv::Scalar       std                 = {0.229, 0.224, 0.225}; // RGB
    std::vector<int> thresholds          = {-1, 10, 10, 100, 100, 100};
    std::vector<int> dynamic_input_shape = {2, 3, 256, 256}; // if your model is dynamic

    std::string model_path  = "";
    std::string images_dir  = "";
    std::string output_dir  = "";
    std::string labels_file = "";
    //-------------------------------------------------------------------------

    auto model = trt::Model(model_path, device);
    model.init();
    auto stream = model.get_stream();
    if (model.is_dynamic())
    {
        if (model.set_binding_dims(0, vector2dims(dynamic_input_shape)))
        {
            model.decode_model_bindings();
            INFO("Setting Successful .");
        }
        else
        {
            ERROR("Setup Failure.");
        }
    }

    nvinfer1::Dims input_dims  = model.get_binding_dims("images");
    nvinfer1::Dims output_dims = model.get_binding_dims("output0");

    input::NCHW input_shape = {"images", input_dims.d[ 0 ], input_dims.d[ 1 ], input_dims.d[ 2 ], input_dims.d[ 3 ]};
    output::Segmentation output_shape = {"output0", output_dims.d[ 0 ], output_dims.d[ 1 ], output_dims.d[ 2 ],
                                         output_dims.d[ 3 ]};

    input_shape.print();
    output_shape.print();

    size_t input_mem_size  = input_shape.volume() * kFLOAT32;
    size_t output_mem_size = output_shape.volume() * kFLOAT32;
    INFO("Input memory size: %d Byte", input_mem_size);
    INFO("Output memory size: %d Byte", output_mem_size);

    auto input_memory  = std::make_shared<trt::Memory>(0, input_mem_size, true, stream);
    auto output_memory = std::make_shared<trt::Memory>(1, output_mem_size, true, stream);

    auto input_wh = cv::Size(input_shape.w, input_shape.h);

    std::vector<cv::String> image_paths = get_image_paths(images_dir, "jpg");
    std::vector<cv::Mat>    images      = get_images(image_paths);

    std::vector<cv::Mat>    batch_images;
    std::vector<cv::String> batch_images_path;

    std::vector<std::string> labels = load_label_from_txt(labels_file);

    int nc = labels.size();

    std::vector<cv::Scalar> color_list = generate_color_list(nc);

    for (int i = 0; i < images.size(); ++i)
    {

        INFO("Read Image: %s", image_paths.at(i).c_str());
        batch_images_path.emplace_back(image_paths.at(i));
        batch_images.emplace_back(images.at(i));

        if (batch_images.size() != input_shape.bs)
            continue;

        // Preprocess --------------------------------------------------------------------------------------------------
        cv::Mat out;
        auto    host_ptr = input_memory->get_cpu_ptr<float>();
        for (int n = 0; n < input_shape.bs; ++n)
        {
            cv::Mat img = batch_images.at(n);

            //            out=cpu::letterbox(img,input_wh);
            out = cpu::resize(img, input_wh);
            out = cpu::bgr2rgb(out);
            out = cpu::normalize(out);
            out = cpu::hwc2chw(out);
            out = cpu::standardize(out, mean, std);

            size_t offset = n * input_shape.CxHxW();
            std::memcpy(host_ptr + offset, out.ptr<float>(0), input_shape.CxHxW() * sizeof(float));
        }

        // Infer--------------------------------------------------------------------------------------------------------
        input_memory->to_gpu();
        void* bindings[ 2 ]{input_memory->get_gpu_ptr(), output_memory->get_gpu_ptr()};
        model.forward(bindings, stream, nullptr);
        output_memory->to_cpu();

        // Decode ------------------------------------------------------------------------------------------------------

        auto output_cpu_ptr = output_memory->get_cpu_ptr<float>();
        ASSERT_PTR(output_cpu_ptr);

        for (int n = 0; n < input_shape.bs; n++)
        {
            cv::Mat curr_image = batch_images[ n ];
            int     iw         = curr_image.cols;
            int     ih         = curr_image.rows;

            size_t offset     = n * output_shape.c * output_shape.w * output_shape.h;
            auto   batch_data = output_cpu_ptr + offset;

            // CHW(ptr)->HWC(Mat)
            // mask.shape=[h,w]
            cv::Mat mask(output_shape.h, output_shape.w, CV_32F, batch_data);
            mask.convertTo(mask, CV_8UC1);
            // Resize
            if (iw != output_shape.w || ih != output_shape.h)
                mask = cpu::resize(mask, cv::Size(iw, ih));

            cv::Mat draw_image(output_shape.h, output_shape.w, CV_8UC3, cv::Scalar(0, 0, 0));
            for (uchar label_id = 0; label_id < labels.size(); ++label_id)
            {

                std::string label = labels[ label_id ];
                int         thr   = thresholds[ label_id ];

                // Skip some label
                if (thr == -1)
                    continue;

                cv::Mat temp;
                cv::compare(mask, label_id, temp, cv::CMP_EQ);
                int num_of_pixel = cv::countNonZero(temp);
                if (num_of_pixel >= thr)
                {
                    draw_image.setTo(color_list[ label_id ], temp);
                }
            }
            // Draw and save image
            cv::Mat     draw_img             = merge_image(curr_image, draw_image);
            std::string basename             = get_basename(batch_images_path[ n ]);
            std::string draw_image_save_path = output_dir + "/" + basename;
            cv::imwrite(draw_image_save_path, draw_img);
            INFO("Save: %s", draw_image_save_path.c_str());
        }
        INFO("Done.\n");

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}