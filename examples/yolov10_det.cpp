#include "iostream"
#include "src/cpu/postprocess.h"
#include "src/cpu/preprocess.h"
#include "src/memory.h"
#include "src/model.h"
#include "src/io.h"
#include "src/utils.h"
#include "unistd.h"
#include "vector"
#include <opencv2/core/core.hpp>

int main(int argc, char const* argv[])
{
    uchar device   = 0;
    float conf_thr = 0.5;
    float iou_thr  = 0.6;

    std::vector<int> dynamic_input_shape = {2, 3, 256, 256};

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

    nvinfer1::Dims input_dims  = model.get_binding_dims(0);
    nvinfer1::Dims output_dims = model.get_binding_dims(1);

    // yolov10 output shape=[bs,rows,dimensions]
    input::NCHW           input_shape = {0, input_dims.d[ 0 ], input_dims.d[ 1 ], input_dims.d[ 2 ], input_dims.d[ 3 ]};
    output::YoloDetection output_shape = {1, output_dims.d[ 0 ], output_dims.d[ 1 ], output_dims.d[ 2 ]};

    input_shape.print();
    output_shape.print();

    size_t input_mem_size  = input_shape.volume() * kFLOAT32;
    size_t output_mem_size = output_shape.volume() * kFLOAT32;
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

        // Decode ------------------------------------------------------------------------------------------------------

        auto output_cpu_ptr = output_memory->get_cpu_ptr<float>();
        ASSERT_PTR(output_cpu_ptr);

        for (int n = 0; n < input_shape.bs; n++)
        {

            size_t offset = n * output_shape.rows * output_shape.dimensions;
            float* data   = output_cpu_ptr + offset;

            // Scale factor
            cv::Mat curr_image = batch_images[ n ];
            auto    x_factor   = curr_image.cols / input_shape.w;
            auto    y_factor   = curr_image.rows / input_shape.h;

            std::vector<result::Detection> detections{};

            for (int row = 0; row < output_shape.rows; ++row)
            {
                float scores   = data[ 4 ];
                int   label_id = int(data[ 5 ]);

                if (scores >= conf_thr)
                {
                    result::Detection det;

                    int x1 = int(data[ 0 ] * x_factor);
                    int y1 = int(data[ 1 ] * y_factor);
                    int x2 = int(data[ 2 ] * x_factor);
                    int y2 = int(data[ 3 ] * y_factor);

                    std::vector<int> xywh = xyxy2xywh(std::vector<int>{x1, y1, x2, y2});

                    det.label_id = label_id;
                    det.label    = labels[ label_id ];
                    det.conf     = scores;
                    det.box      = cv::Rect(xywh[ 0 ], xywh[ 1 ], xywh[ 2 ], xywh[ 3 ]);

                    detections.push_back(det);
                }
                data += output_shape.dimensions;
            }

            cv::Mat     draw_img = draw_box(curr_image, detections, color_list);
            std::string basename = get_basename(batch_images_path[ n ]);

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