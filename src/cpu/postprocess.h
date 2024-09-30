#ifndef CPU_POSTPROCESS_H
#define CPU_POSTPROCESS_H

#include "algorithm"
#include "iostream"
#include <cmath>
#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>

namespace cpu
{

// input: std::vector<T> list;
template <typename T>
inline size_t argmax(const T& data)
{
    if (data.begin() == data.end())
        throw std::invalid_argument("Input data is empty.");

    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

// input:T num[n]
template <typename T>
inline size_t argmax(const T* data, size_t len)
{
    if (len == 0)
        throw std::invalid_argument("Input data is empty");
    return std::distance(data, std::max_element(data, data + len));
}

template <typename T>
std::vector<T> softmax(const T* src, const int num_of_label, bool safe = false)
{
    T max_val = 0;
    if (safe)
    {
        max_val = *std::max_element(src, src + num_of_label);
    }

    std::vector<T> dst(num_of_label);
    T              denominator{0};

    for (auto i = 0; i < num_of_label; i++)
    {
        dst[ i ] = std::exp(src[ i ] - max_val);
        denominator += dst[ i ];
    }

    for (auto i = 0; i < num_of_label; i++)
    {
        dst[ i ] /= denominator;
    }

    return dst;
}

cv::Mat de_normalize(const cv::Mat& input);

cv::Mat mask2mat(const float* ptr, const cv::Size& wh);

template <typename T>
std::vector<T> xyxy2xywh(std::vector<T> xyxy)
{
    T x = xyxy[ 0 ];
    T y = xyxy[ 1 ];
    T w = xyxy[ 2 ] - xyxy[ 0 ];
    T h = xyxy[ 3 ] - xyxy[ 1 ];

    std::vector<T> xywh{x, y, w, h};
    return xywh;
}

template <typename T>
std::vector<T> xywh2xyxy(std::vector<T> xywh)
{
    T x1 = xywh[ 0 ];
    T y1 = xywh[ 1 ];
    T x2 = xywh[ 0 ] + xywh[ 2 ];
    T y2 = xywh[ 1 ] + xywh[ 3 ];

    std::vector<T> xyxy{x1, y1, x2, y2};
    return xyxy;
}

std::vector<int> xyxy2xywh(std::vector<int> xyxy);

std::vector<int> xywh2xyxy(std::vector<int> xywh);

} // namespace cpu
#endif