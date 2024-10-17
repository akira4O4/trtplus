#include "iostream"
#include "src/cpu/postprocess.h"
#include "src/cpu/preprocess.h"
#include "src/io.h"
#include "src/memory.h"
#include "src/model.h"
#include "src/utils.h"
#include "unistd.h"
#include "vector"
#include <algorithm>
#include <opencv2/core/core.hpp>

int main(int argc, char const* argv[])
{
    uchar              device              = 0;
    float              conf_thr            = 0.2;
    float              iou_thr             = 0.6;
    cv::Scalar         mean                = {0.485, 0.456, 0.406};
    cv::Scalar         std                 = {0.229, 0.224, 0.225};
    std::vector<float> thr                 = {-1, 100, 100, 100, 100, 100};
    std::vector<int>   dynamic_input_shape = {2, 3, 640, 640}; // if your model is dynamic

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
    auto images_binding_idx  = int(model.name2idx("images"));
    auto output0_binding_idx = int(model.name2idx("output0"));
    auto output1_binding_idx = int(model.name2idx("output1"));
    std::cout << model.idx2name(0) << std::endl;
    std::cout << model.idx2name(1) << std::endl;
    std::cout << model.idx2name(2) << std::endl;

    std::cout << "images binding idx:" << images_binding_idx << std::endl;
    std::cout << "output0 binding idx:" << output0_binding_idx << std::endl;
    std::cout << "output1 binding idx:" << output1_binding_idx << std::endl;

    nvinfer1::Dims input_dims      = model.get_binding_dims("images");
    nvinfer1::Dims output_det_dims = model.get_binding_dims("output0");
    nvinfer1::Dims output_seg_dims = model.get_binding_dims("output1");

    input::NCHW input_shape = {"images", input_dims.d[ 0 ], input_dims.d[ 1 ], input_dims.d[ 2 ], input_dims.d[ 3 ]};

    output::YoloDetection output_det_shape = {"output0"};
    output_det_shape.bs                    = output_det_dims.d[ 0 ];
    output_det_shape.dimensions            = output_det_dims.d[ 1 ];
    output_det_shape.rows                  = output_det_dims.d[ 2 ];

    output::Segmentation output_seg_shape = {"output1", output_seg_dims.d[ 0 ], output_seg_dims.d[ 1 ],
                                             output_seg_dims.d[ 2 ], output_seg_dims.d[ 3 ]};

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

    auto                    nc         = labels.size();
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

            size_t offset = n * input_shape.CxHxW();
            std::memcpy(host_ptr + offset, out.ptr<float>(0), input_shape.CxHxW() * sizeof(float));
        }

        // Infer--------------------------------------------------------------------------------------------------------
        input_memory->to_gpu();
        void* bindings[ 3 ]{
            input_memory->get_gpu_ptr(),
            output_seg_memory->get_gpu_ptr(),
            output_det_memory->get_gpu_ptr(),
        };
        model.forward(bindings, stream, nullptr);
        output_det_memory->to_cpu();
        output_seg_memory->to_cpu();

        // Decode ------------------------------------------------------------------------------------------------------

        auto output_det_ptr = output_det_memory->get_cpu_ptr<float>();
        auto output_seg_ptr = output_seg_memory->get_cpu_ptr<float>();

        ASSERT_PTR(output_det_ptr);
        ASSERT_PTR(output_seg_ptr);

        // bbox.dimensions=[(x,y,w,h)+nc+mask_info],mask_info=32
        // bbox output.shape[bs,dimensions,rows] dimensions=[xywh,con,nc,mask_info]
        // mask output.shape[bs,c,h,w]

        for (int n = 0; n < input_shape.bs; n++)
        {
            cv::Mat curr_image = batch_images[ n ];
            auto    ih         = curr_image.rows;
            auto    iw         = curr_image.cols;

            size_t det_offset = n * output_det_shape.rows * output_det_shape.dimensions;
            size_t seg_offset = n * output_seg_shape.c * output_seg_shape.h * output_seg_shape.w;
            auto   det_data   = output_det_ptr + det_offset;
            auto   seg_data   = output_seg_ptr + seg_offset;

            // Decode det_head data
            //==========================================================================================================
            // [dims, rows] -> [rows, dims]
            cv::Mat temp_data(output_det_shape.dimensions, output_det_shape.rows, CV_32FC1, det_data);
            cv::Mat det_data_T = cpu::transpose(temp_data);

            std::vector<int>      indices;
            std::vector<int>      classes;
            std::vector<float>    confidences;
            std::vector<cv::Rect> boxes;
            cv::Mat               mask_features;

            for (int i = 0; i < det_data_T.rows; ++i)
            {
                cv::Mat row = det_data_T.row(i);

                // det_head.shape=[xywh(4),conf(nc),mask_feat(32)]
                cv::Mat raw_box_data      = row.colRange(0, 4);
                cv::Mat raw_conf_data     = row.colRange(4, 4 + nc);
                cv::Mat raw_features_data = row.colRange(4 + nc, row.cols);

                // Get the max conf
                cv::Point max_loc;
                double    max_conf;

                cv::minMaxLoc(raw_conf_data, nullptr, &max_conf, nullptr, &max_loc);

                if (max_conf >= conf_thr)
                {
                    confidences.emplace_back(max_conf);
                    classes.emplace_back(max_loc.x);

                    auto cxcywh = cv::Rect2f{raw_box_data.at<float>(0, 0), raw_box_data.at<float>(0, 1),
                                             raw_box_data.at<float>(0, 2), raw_box_data.at<float>(0, 3)};

                    auto factors = cv::Point2f(float(iw) / float(input_shape.w), float(ih) / float(input_shape.h));

                    auto xywh = cpu::cxcywh2xywh(cxcywh, factors);

                    boxes.push_back(xywh);
                    mask_features.push_back(raw_features_data);
                }
            }

            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(boxes, confidences, conf_thr, iou_thr, nms_indices);

            cv::Mat                        nms_mask_features;
            std::vector<int>               nms_classes;
            std::vector<float>             nms_confidences;
            std::vector<cv::Rect>          nms_boxes;
            std::vector<output::Detection> detections{};

            for (int idx : nms_indices)
            {
                nms_boxes.push_back(boxes[ idx ]);
                nms_confidences.push_back(confidences[ idx ]);
                nms_classes.push_back(classes[ idx ]);
                nms_mask_features.push_back(mask_features.row(idx));

                output::Detection det;
                det.label_id = classes[ idx ];
                det.label    = labels[ idx ];
                det.conf     = confidences[ idx ];
                det.box      = boxes[ idx ];
                detections.push_back(det);
            }

            cv::Mat     draw_img = draw_box(curr_image, detections, color_list);
            std::string basename = get_basename(batch_images_path[ n ]);

            std::string draw_image_save_path = output_dir + "/det/" + basename;
            cv::imwrite(draw_image_save_path, draw_img);

            // Decode set.seg_head data
            // =========================================================================================================
            int mc = output_seg_shape.c;
            int mh = output_seg_shape.h;
            int mw = output_seg_shape.w;

            // protos.shape=(mc,mh*mw)
            cv::Mat protos(mc, mh * mw, CV_32F);
            for (int c = 0; c < mc; ++c)
            {
                cv::Mat channel(mh, mw, CV_32F, seg_data + c * mh * mw);
                cv::Mat row_channel = channel.reshape(1, 1); //(h,w)->(1,h*w)
                row_channel.copyTo(protos.row(c));
            }

            ASSERT_TRUE(nms_mask_features.cols == protos.rows);
            cv::Mat masks = nms_mask_features * protos;
            masks.convertTo(masks, CV_8U);
            ASSERT_TRUE(masks.isContinuous());

            // Merge mask [n,h*w]->[h,w]
            cv::Mat merged_mask = cv::Mat::zeros(mh, mw, CV_8U); // shape=[mh,mw]
            for (int nm = 0; nm < masks.rows; ++nm)
            {
                for (int row = 0; row < mh; ++row)
                {
                    for (int col = 0; col < mw; ++col)
                    {
                        auto label = masks.at<uchar>(nm, row * mw + col);
                        if (label > 0 and label != merged_mask.at<uchar>(row, col))
                        {
                            merged_mask.at<uchar>(row, col) = label;
                        }
                    }
                }
            }

            cv::Mat draw_image(mh, mw, CV_8UC3, cv::Scalar(0, 0, 0));
            for (int row = 0; row < mh; ++row)
            {
                for (int col = 0; col < mw; ++col)
                {
                    int label = int(merged_mask.at<uchar>(row, col));
                    if (label > 0)
                    {
                        auto color                         = color_list[ label ];
                        draw_image.at<cv::Vec3b>(row, col) = cv::Vec3b(color[ 0 ], color[ 1 ], color[ 2 ]);
                    }
                }
            }

            cv::Mat out;
            cv::resize(draw_image, out, cv::Size(iw, ih), 0, 0, cv::INTER_LINEAR);
            //            out = cpu::gray2bgr(out);

            // Draw and save image.
            draw_img             = merge_image(curr_image, out);
            basename             = get_basename(batch_images_path[ n ]);
            draw_image_save_path = output_dir + "/seg/" + basename;
            cv::imwrite(draw_image_save_path, draw_img);
            INFO("Save: %s", draw_image_save_path.c_str());
        }
        INFO("Done.\n");

        batch_images.clear();
        batch_images_path.clear();
    }

    return 0;
}