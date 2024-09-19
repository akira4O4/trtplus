#ifndef UTILS_H
#define UTILS_H

#include "NvInfer.h"
#include "cuda_fp16.h"
#include "iostream"
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <utility>

constexpr uchar   kDefaultDevice  = 0;
constexpr uchar   kDefaultBatch   = 1;
constexpr uchar   kDefaultChannel = 3;
constexpr uchar   kDefaultHeight  = 224;
constexpr uchar   kDefaultWidth   = 224;
constexpr float   kDefaultIoU     = 0.5;
constexpr float   kDefaultConf    = 0.5;
const std::string kDefaultMode    = "fp32";

#define DEPRECATED [[deprecated]]

#define INFO(...) info(__FILE__, __LINE__, __VA_ARGS__)

#define ERROR(...)

#define CHECK_CUDA_RUNTIME(call)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret_code = (call);                                                                                        \
        if (ret_code != cudaSuccess)                                                                                   \
        {                                                                                                              \
            INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call, cudaGetErrorString(ret_code),                \
                 cudaGetErrorName(ret_code), ret_code);                                                                \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_OP(op)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        bool state = !(!(op));                                                                                         \
        if (!state)                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "[%s:%d]: Assert failed: %s", __FILE__, __LINE__, #op);                                    \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_PTR(ptr)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (ptr == nullptr)                                                                                            \
        {                                                                                                              \
            fprintf(stderr, "[%s:%d]: Null pointer detected: %s", __FILE__, __LINE__, #ptr);                           \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

void info(const char* file, int line, const char* fmt, ...);

std::string file_name(const std::string& path, bool include_suffix);

std::vector<cv::String> get_image_paths(const std::string& path, const std::string& pattern = "jpg");

std::vector<cv::Mat> get_images(const std::vector<cv::String>& image_paths);

void save_image(const std::string&, const std::vector<cv::Mat>& images, const std::string& pattern = "jpg");

bool is_exists(std::string& name);

std::vector<std::string> load_label_from_txt(const std::string& file_name);

std::vector<int> dims2vector(nvinfer1::Dims dims);

nvinfer1::Dims vector2dims(const std::vector<int>& data);

void print_dims(nvinfer1::Dims dims);

#endif