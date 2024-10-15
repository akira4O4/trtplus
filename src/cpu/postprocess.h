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

template <typename T>
inline size_t argmax(const T& data)
{
    if (data.begin() == data.end())
        throw std::invalid_argument("Input data is empty.");

    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

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

template <typename T>
cv::Rect_<T> xyxy2xywh(std::vector<T> xyxy)
{
    cv::Rect_<T> out{xyxy[ 0 ], xyxy[ 0 ], xyxy[ 2 ] - xyxy[ 0 ], xyxy[ 3 ] - xyxy[ 1 ]};
    return out;
}

template <typename T>
std::vector<T> xywh2xyxy(cv::Rect_<T> xywh)
{
    std::vector<T> out{xywh.x, xywh.y, xywh.x + xywh.width, xywh.y + xywh.height};
    return out;
}

template <typename T>
cv::Rect_<T> cxcywh2xywh(cv::Rect_<T> data)
{
    cv::Rect_<T> out{data.x - data.width / 2, data.y - data.height / 2, data.width, data.height};
    return out;
}

template <typename T>
cv::Rect_<T> cxcywh2xywh(cv::Rect_<T> data, cv::Point_<T> factors)
{
    cv::Rect_<T> out;
    out.x      = (data.x - data.width / 2) * factors.x;
    out.y      = (data.y - data.height / 2) * factors.y;
    out.width  = data.width * factors.x;
    out.height = data.height * factors.y;
    return out;
}

cv::Mat transpose(const cv::Mat& data);

template <typename T>
cv::Rect_<T> vec2rect(std::vector<T> xywh)
{
    cv::Rect_<T> out{xywh[ 0 ], xywh[ 1 ], xywh[ 2 ], xywh[ 3 ]};
    return out;
}

// output=[x,y,w,h]
template <typename T>
std::vector<T> rect2vec(cv::Rect_<T> data)
{
    std::vector<T> out{data.x, data.y, data.width, data.height};
    return out;
}

template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (v > hi ? hi : v);
}

// data.shape=[n,h*w]
//cv::Mat merge_mat(const cv::Mat& data, int h, int w)
//{
//    cv::Mat out(h, w, CV_8U);
//    for (int n = 0; n < data.rows; ++n)
//    {
//    }
//    return out;
//}

} // namespace cpu

#endif