#ifndef UTILS_H
#define UTILS_H

#include "NvInfer.h"
#include "io.h"
#include "iostream"
#include "utils.h"
#include <experimental/filesystem>
#include <opencv2/core/core.hpp>
#include <utility>

typedef unsigned short bfloat16_t;
typedef unsigned short float16_t;
typedef float          float32_t;

constexpr uint8_t kINT8     = sizeof(uint8_t);    // size=1
constexpr uint8_t kFLOAT16  = sizeof(float16_t);  // size=2
constexpr uint8_t kBFLOAT16 = sizeof(bfloat16_t); // size=2
constexpr uint8_t kFLOAT32  = sizeof(float32_t);  // size=4

#define DEPRECATED [[deprecated]]

#define INFO(format, ...) info(__FILE__, __LINE__, format, ##__VA_ARGS__)

#define ERROR(format, ...) error(__FILE__, __LINE__, format, ##__VA_ARGS__)

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

#define ASSERT_TRUE(op)                                                                                                \
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
        if ((ptr) == nullptr)                                                                                          \
        {                                                                                                              \
            fprintf(stderr, "[%s:%d]: Null pointer detected: %s", __FILE__, __LINE__, #ptr);                           \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

void info(const char* file, int line, const char* format, ...);

void error(const char* file, int line, const char* format, ...);

std::vector<cv::String> get_image_paths(const std::string& path, const std::string& pattern = "jpg");

std::vector<cv::Mat> get_images(const std::vector<cv::String>& image_paths);

std::vector<std::string> load_label_from_txt(const std::string& file_name);

std::vector<int> dims2vector(nvinfer1::Dims dims);

nvinfer1::Dims vector2dims(const std::vector<int>& data);

void print_dims(nvinfer1::Dims dims);

size_t dims_volume(nvinfer1::Dims dims);

std::vector<cv::Scalar> generate_color_list(int nc);

std::vector<cv::Scalar> generate_color_list_v1(int nc);

std::string get_basename(const std::string& filePath);

cv::Mat draw_box(const cv::Mat& image, const std::vector<output::Detection>& detections,
                 std::vector<cv::Scalar> colors);

cv::Mat draw_obb_box(const cv::Mat& image, const std::vector<output::Detection>& detections,
                     std::vector<cv::Scalar> colors);

auto merge_image(const cv::Mat& image, const cv::Mat& mask) -> cv::Mat;

double radians2degrees(double radians);

#endif