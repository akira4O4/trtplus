#include "../src/cpu/prepprocess.h"
#include "../src/memory.h"
#include "iostream"
#include "unistd.h"
#include "vector"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace nv = nvinfer1;

int memory()
{
    std::string images_dir = "/home/seeking/llf/code/trtplus/temp/images/imagenet";

    std::vector<cv::String> image_paths = get_image_paths(images_dir, "jpg");
    std::vector<cv::Mat>    images      = get_images(image_paths);

    std::vector<cv::Mat>    batch_images;
    std::vector<cv::String> batch_images_path;

    auto batch = kDefaultBatch;
    auto c     = kDefaultChannel;
    auto h     = kDefaultHeight;
    auto w     = kDefaultWidth;

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
            img         = cpu::resize(img, cv::Size(w, h));
            img.convertTo(img, CV_32FC3, 1.f / 255.f);

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

            std::string save_dir  = "/home/seeking/llf/code/trtplus/temp/output";
            auto        save_path = save_dir + "/bs" + std::to_string(i) + "_i" + std::to_string(j) + ".jpg";
            cv::imwrite(save_path, out);
        }
        batch_images.clear();
        batch_images_path.clear();
    }
    return 0;
}