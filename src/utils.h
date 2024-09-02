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
const std::string kDefaultMode    = "fp32";

constexpr float kDefaultIoU  = 0.5;
constexpr float kDefaultConf = 0.5;

#define DEPRECATED [[deprecated]]

#define INFO(...) info(__FILE__, __LINE__, __VA_ARGS__)

#define checkRuntime(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        auto __call_ret_code__ = (call);                                                                               \
        if (__call_ret_code__ != cudaSuccess)                                                                          \
        {                                                                                                              \
            INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call, cudaGetErrorString(__call_ret_code__),       \
                 cudaGetErrorName(__call_ret_code__), __call_ret_code__);                                              \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define checkKernel(...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        {                                                                                                              \
            (__VA_ARGS__);                                                                                             \
        }                                                                                                              \
        checkRuntime(cudaPeekAtLastError());                                                                           \
    } while (0)

#define Assert(op)                                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        bool cond = !(!(op));                                                                                          \
        if (!cond)                                                                                                     \
        {                                                                                                              \
            INFO("Assert failed, " #op);                                                                               \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define Assertf(op, ...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        bool cond = !(!(op));                                                                                          \
        if (!cond)                                                                                                     \
        {                                                                                                              \
            INFO("Assert failed, " #op " : " __VA_ARGS__);                                                             \
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