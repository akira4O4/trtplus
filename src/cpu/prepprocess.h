//
// Created by main on 24-4-11.
//

#ifndef MAIN_PREPPROCESS_H
#define MAIN_PREPPROCESS_H

#include "result.hpp"
#include <opencv2/opencv.hpp>
namespace cpu
{

cv::Mat hwc2chw(cv::Mat& image);
void    hwc2chw(cv::Mat& img, float* out);
void    hwc2chw(cv::Mat& img, float* output, int bs, int dst_width, int dst_height);
void    hwc2chw(const result::NCHW& input_shape, const cv::Mat& data, float* ptr, int bs_offset);
cv::Mat normalize_image(const cv::Mat& input);

cv::Mat standardize_image(const cv::Mat& input);

cv::Mat bgr2rgb(const cv::Mat& input);

cv::Mat split_channel(const cv::Mat& input);

cv::Mat letterbox(const cv::Mat& input, const cv::Size2i& wh);

cv::Mat resize(const cv::Mat& input, const cv::Size& wh);
// void preprocess(cv::Mat &img, float *data, const cv::Size &model_input_wh);

// yolo preprocess pipline
// 1.bgr2rgb
// 2.resize
// 3.1/255
// 4.hwc2chw

} // namespace cpu
#endif // MAIN_PREPPROCESS_H
