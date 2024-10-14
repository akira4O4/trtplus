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
// cv::Mat scale_mask(const cv::Mat& masks, const cv::Size& image_shape)
//{
//     cv::Size mask_shape = masks.size();
//
//     float       r;
//     cv::Point2f pad;
//
//     auto ih = image_shape.height;
//     auto iw = image_shape.width;
//     auto mh = mask_shape.height;
//     auto mw = mask_shape.width;
//
//     r = std::min(static_cast<float>(mh) / ih, static_cast<float>(mw) / iw);
//     pad  = cv::Point2f((mw - iw * r) / 2, (mh - ih * r) / 2);
//
//     int top    = static_cast<int>(round(pad.y - 0.1));
//     int left   = static_cast<int>(round(pad.x - 0.1));
//     int bottom = static_cast<int>(round(mh - pad.y + 0.1));
//     int right  = static_cast<int>(round(mw - pad.x + 0.1));
//
//     cv::Mat cropped_masks = masks(cv::Rect(left, top, right - left, bottom - top));
//
//     cv::Mat resized_masks;
//     cv::resize(cropped_masks, resized_masks, image_shape, 0, 0, cv::INTER_LINEAR);
//
//     // 如果是二维 mask，调整到三维 (h, w, 1)
//     if (resized_masks.channels() == 1)
//     {
//         cv::Mat reshaped_masks;
//         cv::cvtColor(resized_masks, reshaped_masks, cv::COLOR_GRAY2BGR); // 处理单通道 mask
//         resized_masks = reshaped_masks;
//     }
//
//     return resized_masks;
// }
//
// cv::Mat crop_mask(const cv::Mat& masks, const cv::Mat& boxes)
//{
//     int n = masks.size[ 0 ]; // Number of masks
//     int h = masks.size[ 1 ]; // Height
//     int w = masks.size[ 2 ]; // Width
//
//     // Initialize a new mask for storing cropped results
//     cv::Mat cropped_masks = cv::Mat::zeros(masks.size(), masks.type());
//
//     // Split boxes into x1, y1, x2, y2 (assumes boxes shape is [n, 4])
//     std::vector<cv::Mat> box_coords;
//     cv::split(boxes, box_coords);
//     cv::Mat x1 = box_coords[ 0 ];
//     cv::Mat y1 = box_coords[ 1 ];
//     cv::Mat x2 = box_coords[ 2 ];
//     cv::Mat y2 = box_coords[ 3 ];
//
//     // Create row and column ranges
//     cv::Mat r(1, w, CV_32F); // Range for width
//     cv::Mat c(1, h, CV_32F); // Range for height
//
//     for (int i = 0; i < w; ++i)
//     {
//         r.at<float>(0, i) = static_cast<float>(i);
//     }
//     for (int i = 0; i < h; ++i)
//     {
//         c.at<float>(0, i) = static_cast<float>(i);
//     }
//
//     // Iterate through each mask
//     for (int i = 0; i < n; ++i)
//     {
//         cv::Mat mask = masks.row(i); // Single mask
//
//         // Get bounding box coordinates for current mask
//         float x1_val = x1.at<float>(i);
//         float y1_val = y1.at<float>(i);
//         float x2_val = x2.at<float>(i);
//         float y2_val = y2.at<float>(i);
//
//         // Create condition matrices for cropping
//         cv::Mat r_cond = (r >= x1_val) & (r < x2_val);
//         cv::Mat c_cond = (c >= y1_val) & (c < y2_val);
//
//         // Perform cropping
//         for (int y = 0; y < h; ++y)
//         {
//             for (int x = 0; x < w; ++x)
//             {
//                 if (r_cond.at<float>(0, x) && c_cond.at<float>(0, y))
//                 {
//                     cropped_masks.at<float>(i, y, x) = mask.at<float>(y, x);
//                 }
//             }
//         }
//     }
//
//     return cropped_masks;
// }

} // namespace cpu

#endif