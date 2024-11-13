#include "utils.h"
#include "opencv2/opencv.hpp"
#include "vector"
#include <cstdarg>

void info(const char* file, int line, const char* format, ...)
{
    std::string basename = get_basename(file);
    fprintf(stdout, "INFO:[%s:%d]: ", basename.c_str(), line);
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fprintf(stdout, "\n");
}

void error(const char* file, int line, const char* format, ...)
{
    // 打印文件名和行号
    std::string basename = get_basename(file);
    fprintf(stderr, "ERROR:[%s:%d]: ", basename.c_str(), line);

    // 可变参数处理
    va_list args;
    va_start(args, format);

    // 打印格式化的错误信息
    vfprintf(stderr, format, args);

    // 结束变参的处理
    va_end(args);

    // 换行
    fprintf(stderr, "\n");
}

std::vector<cv::String> get_image_paths(const std::string& path, const std::string& pattern)
{
    cv::String              path_pattern = path + "/*." + pattern;
    std::vector<cv::String> image_paths;
    cv::glob(path_pattern, image_paths);
    return image_paths;
}

std::vector<cv::Mat> get_images(const std::vector<cv::String>& image_paths)
{
    std::vector<cv::Mat> images;
    for (const auto& image_path : image_paths)
    {
        cv::Mat im = cv::imread(image_path);
        images.emplace_back(im);
    }
    return images;
}

std::vector<std::string> load_label_from_txt(const std::string& file_name)
{
    std::vector<std::string> classes;
    std::ifstream            ifs(file_name, std::ios::in);
    if (!ifs.is_open())
    {
        ERROR("%s is not found, pls refer to README and download it.", file_name.c_str());
        assert(0);
    }
    std::string s;
    while (std::getline(ifs, s))
    {
        classes.push_back(s);
    }
    ifs.close();
    return classes;
}

std::vector<int> dims2vector(const nvinfer1::Dims dims)
{
    std::vector<int> vec(dims.d, dims.d + dims.nbDims);
    return vec;
}

nvinfer1::Dims vector2dims(const std::vector<int>& data)
{
    ASSERT_TRUE(1 <= data.size() <= 8);
    nvinfer1::Dims d = {};
    std::memcpy(d.d, data.data(), sizeof(int) * data.size());
    d.nbDims = data.size();
    return d;
}

void print_dims(nvinfer1::Dims dims)
{
    std::cout << "[ ";

    for (int i = 0; i < dims.nbDims; i++)
    {
        std::cout << dims.d[ i ] << " ";
    }
    std::cout << " ]\n";
}

size_t dims_volume(nvinfer1::Dims dims)
{
    size_t volume = 1;
    for (int i : dims.d)
    {
        if (i <= 0)
            continue;
        volume *= i;
    }
    return volume;
}

std::vector<cv::Scalar> generate_color_list(int nc)
{
    std::vector<cv::Scalar> color_list;

    for (int i = 0; i < nc; ++i)
    {
        // 生成不同的颜色，HSV 色相值从 0 到 180，饱和度和明亮度固定为 255
        int        hue   = static_cast<int>((i * 180.0) / nc) % 180; // 色相
        cv::Scalar color = cv::Scalar(0, 0, 0);                      // 初始化为黑色
        cv::Mat    hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));    // HSV颜色
        cv::Mat    bgr;

        // 将 HSV 转换为 BGR
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        color = cv::Scalar(bgr.at<cv::Vec3b>(0, 0)[ 0 ], bgr.at<cv::Vec3b>(0, 0)[ 1 ], bgr.at<cv::Vec3b>(0, 0)[ 2 ]);

        color_list.push_back(color);
    }

    return color_list;
}

std::vector<cv::Scalar> generate_gray_color_list(int numColors)
{
    std::vector<cv::Scalar> color_list;

    for (int i = 0; i < numColors; ++i)
    {
        // 生成不同的灰度值，灰度值范围为 [0, 255]
        int gray_value = static_cast<int>((i * 255.0) / numColors) % 256; // 生成均匀分布的灰度值

        // RGB 三个通道保持一致，即灰度颜色
        cv::Scalar color = cv::Scalar(gray_value, gray_value, gray_value); // 灰度颜色

        color_list.push_back(color);
    }

    return color_list;
}

std::string get_basename(const std::string& filePath)
{
    size_t pos = filePath.find_last_of("/\\");
    return filePath.substr(pos + 1);
}

cv::Mat draw_obb_box(const cv::Mat& image, const std::vector<output::Detection>& detections,
                     std::vector<cv::Scalar> colors)
{
    cv::Mat imageCopy = image.clone();

    for (int n = 0; n < detections.size(); ++n)
    {
        output::Detection det = detections[ n ];

        auto x     = det.box.x;
        auto y     = det.box.y;
        auto color = colors[ det.label_id ];
        auto angle = radians2degrees(det.angle);

        if (angle != 0.0)
        {
            auto w = det.box.width;
            auto h = det.box.height;

            INFO("%d,%d,%d,%d,%f", x, y, w, h, angle);

            // Get the center x and y.
            cv::Point2f     center(x + w / 2, y + h / 2);
            cv::RotatedRect rotatedRect(center, cv::Size2f(w, h), angle);

            cv::Point2f vertices[ 4 ];
            rotatedRect.points(vertices);

            for (int i = 0; i < 4; ++i)
            {
                cv::line(imageCopy, vertices[ i ], vertices[ (i + 1) % 4 ], color, 2);
            }
        }
        else
        {
            cv::rectangle(imageCopy, det.box, color, 1);
        }

        float conf = floor(100 * det.conf) / 100;
        std::cout << std::fixed << std::setprecision(2);
        // label:conf:angle
        std::string label = det.label + ":" + std::to_string(conf).substr(0, std::to_string(conf).size() - 4) + ":" +
                            std::to_string(angle);

        cv::putText(imageCopy, label, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
    }

    return imageCopy;
}

cv::Mat draw_box(const cv::Mat& image, const std::vector<output::Detection>& detections, std::vector<cv::Scalar> colors)
{
    cv::Mat imageCopy = image.clone();

    for (int n = 0; n < detections.size(); ++n)
    {
        output::Detection det = detections[ n ];

        auto x     = det.box.x;
        auto y     = det.box.y;
        auto color = colors[ det.label_id ];

        cv::rectangle(imageCopy, det.box, color, 1);

        float conf = floor(100 * det.conf) / 100;
        std::cout << std::fixed << std::setprecision(2);
        std::string label = det.label + " " + std::to_string(conf).substr(0, std::to_string(conf).size() - 4);

        cv::putText(imageCopy, label, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    return imageCopy;
}

auto merge_image(const cv::Mat& image, const cv::Mat& mask) -> cv::Mat
{
    cv::Mat resized_mask;

    if (mask.size() != image.size())
        cv::resize(mask, resized_mask, image.size());
    else
        resized_mask = mask;

    cv::Mat output;
    cv::addWeighted(image, 0.3, resized_mask, 0.7, 0, output);
    return output;
}

double radians2degrees(double radians)
{
    return radians * (180.0 / CV_PI);
}