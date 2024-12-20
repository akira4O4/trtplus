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
    float            conf_thr            = 0.5;
    float            iou_thr             = 0.6;
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
        if (model.set_binding_dims("images", vector2dims(dynamic_input_shape)))
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

    // yolov8 output shape=[bs,dimensions,rows]
    // e.g. output shape: [84, 8400]; 84 means: [cx, cy, w, h, prob * 80]
    input::NCHW input_shape = {"images", input_dims.d[ 0 ], input_dims.d[ 1 ], input_dims.d[ 2 ], input_dims.d[ 3 ]};
    output::YoloDetection output_shape = {"output0"};
    output_shape.bs                    = output_dims.d[ 0 ];
    output_shape.rows                  = output_dims.d[ 2 ];
    output_shape.dimensions            = output_dims.d[ 1 ];

    input_shape.print();
    output_shape.print();

    size_t input_mem_size  = input_shape.volume() * kFLOAT32;
    size_t output_mem_size = output_shape.volume() * kFLOAT32;
    INFO("Input memory size: %d Byte", input_mem_size);
    INFO("Output memory size: %d Byte", output_mem_size);

    auto input_memory  = std::make_shared<trt::Memory>(0, input_mem_size, true, stream);
    auto output_memory = std::make_shared<trt::Memory>(1, output_mem_size, true, stream);

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
            out = cpu::hwc2chw(out);

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

        //  yolov8 output of shape (bs, dimensions, rows) (dimensions=[xywh+nc])
        for (int n = 0; n < input_shape.bs; n++)
        {

            size_t offset     = n * output_shape.rows * output_shape.dimensions;
            auto   batch_data = output_cpu_ptr + offset;

            // [dims, rows] -> [rows, dims]
            cv::Mat temp_data(output_shape.dimensions, output_shape.rows, CV_32FC1, batch_data);
            cv::Mat transposed_data;
            cv::transpose(temp_data, transposed_data);

            auto data = transposed_data.ptr<float>();

            std::vector<int>      label_ids;
            std::vector<float>    confidences;
            std::vector<cv::Rect> boxes;

            // Scale factor
            cv::Mat curr_image = batch_images[ n ];
            auto    x_factor   = curr_image.cols / input_shape.w;
            auto    y_factor   = curr_image.rows / input_shape.h;

            for (int row = 0; row < output_shape.rows; ++row)
            {
                float* classes_scores = data + 4;
                size_t max_score_idx  = cpu::argmax(classes_scores, nc);
                float  max_score      = classes_scores[ max_score_idx ];

                if (max_score >= conf_thr)
                {
                    confidences.emplace_back(max_score);
                    label_ids.push_back(max_score_idx);

                    float cx = data[ 0 ];
                    float cy = data[ 1 ];
                    float w  = data[ 2 ];
                    float h  = data[ 3 ];

                    int real_x = int((cx - 0.5 * w) * x_factor);
                    int real_y = int((cy - 0.5 * h) * y_factor);
                    int real_w = int(w * x_factor);
                    int real_h = int(h * y_factor);

                    boxes.emplace_back(cv::Rect(real_x, real_y, real_w, real_h));
                }
                data += output_shape.dimensions;
            }

            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, confidences, conf_thr, iou_thr, nms_result);

            std::vector<output::Detection> detections{};
            for (unsigned long i = 0; i < nms_result.size(); ++i)
            {
                int idx = nms_result[ i ];

                output::Detection det;
                det.label_id = label_ids[ idx ];
                det.label    = labels[ det.label_id ];
                det.conf     = confidences[ idx ];
                det.box      = boxes[ idx ];
                detections.push_back(det);
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