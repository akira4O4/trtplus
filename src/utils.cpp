#include "utils.h"

#include <cstdarg>
#include <opencv2/core/core.hpp>

#include "iostream"
#include "opencv2/opencv.hpp"

std::string file_name(const std::string& path, bool include_suffix)
{
    if (path.empty())
    {
        return "";
    }

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p     = std::max(p, e);
    p += 1;

    // include suffix
    if (include_suffix)
    {
        return path.substr(p);
    }

    int u = path.rfind('.');
    if (u == -1)
    {
        return path.substr(p);
    }

    if (u <= p)
    {
        u = path.size();
    }
    return path.substr(p, u - p);
}

void info_(const char* file, int line, const char* fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    char        buffer[ 2048 ];
    std::string filename = file_name(file, true);
    int         n        = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    fprintf(stdout, "%s\n", buffer);
}

void info(const char* file, int line, const char* format, ...)
{
    fprintf(stdout, "INFO:[%s:%d]: ", file, line);
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fprintf(stdout, "\n");
}

void error(const char* file, int line, const char* format, ...)
{
    // 打印文件名和行号
    fprintf(stderr, "ERROR:[%s:%d]: ", file, line);

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

void save_image(const std::string& save_dir, const std::vector<cv::Mat>& images, const std::string& pattern)
{
    for (int i = 0; i < images.size(); ++i)
    {
        std::string save_path = save_dir + "/" + std::to_string(i) + "." + pattern;
        cv::imwrite(save_path, images[ i ]);
    }
}

bool is_exists(std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
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
    nvinfer1::Dims d;
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
