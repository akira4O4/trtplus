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

//std::vector<int> xyxy2xywh(std::vector<int> xyxy);
//
//std::vector<int> xywh2xyxy(std::vector<int> xywh);

template <typename T>
std::vector<T> cxcywh2xyxy(std::vector<T> cxcywh)
{
    std::vector<T> xyxy(4);
    xyxy[ 0 ] = cxcywh[ 0 ] - cxcywh[ 2 ] / 2; // x_min = center_x - width / 2
    xyxy[ 1 ] = cxcywh[ 1 ] - cxcywh[ 3 ] / 2; // y_min = center_y - height / 2
    xyxy[ 2 ] = cxcywh[ 0 ] + cxcywh[ 2 ] / 2; // x_max = center_x + width / 2
    xyxy[ 3 ] = cxcywh[ 1 ] + cxcywh[ 3 ] / 2; // y_max = center_y + height / 2
    return xyxy;
}

template <typename T>
std::vector<T> cxcywh2xyxy(std::vector<T> cxcywh, cv::Point2f factors)
{
    std::vector<T> xyxy(4);
    xyxy[ 0 ] = (cxcywh[ 0 ] - cxcywh[ 2 ] / 2) * factors.x;
    xyxy[ 1 ] = (cxcywh[ 1 ] - cxcywh[ 3 ] / 2) * factors.y;
    xyxy[ 2 ] = (cxcywh[ 0 ] + cxcywh[ 2 ] / 2) * factors.x;
    xyxy[ 3 ] = (cxcywh[ 1 ] + cxcywh[ 3 ] / 2) * factors.y;
    return xyxy;
}

template <typename T>
std::vector<T> cxcywh2xywh(std::vector<T> cxcywh)
{
    std::vector<T> xywh(4);
    xywh[ 0 ] = (cxcywh[ 0 ] - cxcywh[ 2 ] / 2); // x_min = center_x - width / 2
    xywh[ 1 ] = (cxcywh[ 1 ] - cxcywh[ 3 ] / 2); // y_min = center_y - height / 2
    xywh[ 2 ] = (cxcywh[ 2 ]);                   // width
    xywh[ 3 ] = (cxcywh[ 3 ]);                   // height
    return xywh;
}

template <typename T>
std::vector<T> cxcywh2xywh(std::vector<T> cxcywh, cv::Point2f factors)
{
    std::vector<T> xywh(4);
    xywh[ 0 ] = (cxcywh[ 0 ] - cxcywh[ 2 ] / 2) * factors.x; // x_min = center_x - width / 2
    xywh[ 1 ] = (cxcywh[ 1 ] - cxcywh[ 3 ] / 2) * factors.y; // y_min = center_y - height / 2
    xywh[ 2 ] = (cxcywh[ 2 ]) * factors.x;                   // width
    xywh[ 3 ] = (cxcywh[ 3 ]) * factors.y;                   // height
    return xywh;
}

template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (v > hi ? hi : v);
}

cv::Mat transpose(const cv::Mat& data);

template <typename T>
cv::Rect vec2rect(std::vector<T> vec)
{
    return cv::Rect(vec[ 0 ], vec[ 1 ], vec[ 2 ], vec[ 3 ]);
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