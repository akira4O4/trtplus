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
    uchar              device              = 0;
    cv::Scalar         mean                = {0.485, 0.456, 0.406};
    cv::Scalar         std                 = {0.229, 0.224, 0.225};
    std::vector<float> thr                 = {-1, 100, 100, 100, 100, 100};
    std::vector<int>   dynamic_input_shape = {2, 3, 640, 640}; // if your model is dynamic

    std::string model_path =
        "/home/seeking/llf/code/trtplus/assets/coco8/models/v8n-seg-1x3x640x640.fp32.static.engine";
    std::string images_dir  = "/home/seeking/llf/code/trtplus/assets/coco8/images";
    std::string output_dir  = "/home/seeking/llf/code/trtplus/assets/coco8/output";
    std::string labels_file = "/home/seeking/llf/code/trtplus/assets/coco8/labels.txt";
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

    nvinfer1::Dims input_dims      = model.get_binding_dims("images");
    nvinfer1::Dims output_det_dims = model.get_binding_dims("output0");
    nvinfer1::Dims output_seg_dims = model.get_binding_dims("output1");

    input::NCHW input_shape = {0, input_dims.d[ 0 ], input_dims.d[ 1 ], input_dims.d[ 2 ], input_dims.d[ 3 ]};

    output::YoloDetection output_det_shape = {1};
    output_det_shape.bs                    = output_det_dims.d[ 0 ];
    output_det_shape.dimensions            = output_det_dims.d[ 1 ];
    output_det_shape.rows                  = output_det_dims.d[ 2 ];

    output::Segmentation output_seg_shape = {2, output_seg_dims.d[ 0 ], output_seg_dims.d[ 1 ], output_seg_dims.d[ 2 ],
                                             output_seg_dims.d[ 3 ]};

    input_shape.print();
    output_det_shape.print();
    output_seg_shape.print();

    size_t input_mem_size      = input_shape.volume() * kFLOAT32;
    size_t output_det_mem_size = output_det_shape.volume() * kFLOAT32;
    size_t output_seg_mem_size = output_seg_shape.volume() * kFLOAT32;
    INFO("Input memory size: %d Byte", input_mem_size);
    INFO("Output det memory size: %d Byte", output_det_mem_size);
    INFO("Output seg memory size: %d Byte", output_seg_mem_size);

    auto input_memory      = std::make_shared<trt::Memory>(0, input_mem_size, true, stream);
    auto output_det_memory = std::make_shared<trt::Memory>(1, output_det_mem_size, true, stream);
    auto output_seg_memory = std::make_shared<trt::Memory>(2, output_seg_mem_size, true, stream);

    auto input_wh = cv::Size(input_shape.w, input_shape.h);

    std::vector<cv::String>  image_paths = get_image_paths(images_dir, "jpg");
    std::vector<cv::Mat>     images      = get_images(image_paths);
    std::vector<std::string> labels      = load_label_from_txt(labels_file);

    std::vector<cv::Mat>    batch_images;
    std::vector<cv::String> batch_images_path;

    auto nc         = labels.size();
    auto color_list = generate_color_list(nc);

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

            size_t offset = n * input_shape.CxHxW();
            std::memcpy(host_ptr + offset, out.ptr<float>(0), input_shape.CxHxW() * sizeof(float));
        }

        // Infer--------------------------------------------------------------------------------------------------------
        input_memory->to_gpu();
        void* bindings[ 3 ]{input_memory->get_gpu_ptr(), output_det_memory->get_gpu_ptr(),
                            output_seg_memory->get_gpu_ptr()};
        model.forward(bindings, stream, nullptr);
        output_det_memory->to_cpu();
        output_seg_memory->to_cpu();

        // Decode ------------------------------------------------------------------------------------------------------

        auto output_det_ptr = output_det_memory->get_cpu_ptr<float>();
        auto output_seg_ptr = output_seg_memory->get_cpu_ptr<float>();

        ASSERT_PTR(output_det_ptr);
        ASSERT_PTR(output_seg_ptr);

        // bbox.dimensions=[(x,y,w,h)+conf+nc+mask_info],mask_info=32

        // bbox output.shape[bs,dimensions,rows] dimensions=[xywh,con,nc,mask_info]
        // mask output.shape[bs,c,h,w]

        for (int n = 0; n < input_shape.bs; n++)
        {
            cv::Mat mask;
            cv::Mat curr_image = batch_images[ n ];

            size_t det_offset = n * output_det_shape.rows * output_det_shape.dimensions;
            size_t seg_offset = n * output_seg_shape.c * output_seg_shape.h * output_seg_shape.w;
            auto   det_data   = output_det_ptr + det_offset;
            auto   seg_data   = output_seg_ptr + seg_offset;

            // Draw and save image.
            //            cv::Mat     draw_img             = merge_image(curr_image, mask);
            //            std::string basename             = get_basename(batch_images_path[ n ]);
            //            std::string draw_image_save_path = output_dir + "/" + basename;
            //            cv::imwrite(draw_image_save_path, draw_img);
            //            INFO("Save: %s", draw_image_save_path.c_str());
        }
        INFO("Done.\n");

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}