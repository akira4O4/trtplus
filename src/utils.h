#ifndef UTILS_H
#define UTILS_H

#include "NvInfer.h"
#include "iostream"
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <utility>

const uchar       kDefaultChannel = 3;
const uchar       kDefaultDevice  = 0;
const uchar       kDefaultBatch   = 1;
const uchar       kDefaultHeight  = 224;
const uchar       kDefaultWidth   = 224;
const std::string kDefaultMode    = "fp32";
const float       kDefaultIOU     = 0.5;
const float       kDefaultCONF    = 0.5;

#define DEPRECATED [[deprecated]]

#define INT8 sizeof(char)
#define UINT8 sizeof(uchar)

#define FLOAT16 sizeof(half)
#define FLOAT32 sizeof(float)
#define HALF FLOAT16

#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)

#define KERNEL_CHECK(call) kernel_check(__FILE__, __LINE__)

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

static void cuda_check(cudaError_t err, const char* file, const int line);

static void kernel_check(const char* file, const int line);

std::vector<std::string> load_txt(const std::string& file_name);
#endif