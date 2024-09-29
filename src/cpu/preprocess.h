#ifndef CPU_PREPROCESS_H
#define CPU_PREPROCESS_H

#include <opencv2/opencv.hpp>
namespace cpu
{
bool is_gray(const cv::Mat& image);

void hwc2chw(cv::Mat& img, float* out);

cv::Mat hwc2chw(const cv::Mat& img);

void chw2hwc(cv::Mat& img, float* out);

cv::Mat bgr2rgb(const cv::Mat& input);

cv::Mat rgb2bgr(const cv::Mat& input);

cv::Mat gray2rgb(const cv::Mat& input);

cv::Mat gray2bgr(const cv::Mat& input);

cv::Mat normalize(const cv::Mat& input);

cv::Mat standardize(const cv::Mat& input, const cv::Scalar& mean, const cv::Scalar& std);

cv::Mat letterbox(const cv::Mat& input, const cv::Size& wh);

cv::Mat resize(const cv::Mat& input, const cv::Size& wh);

} // namespace cpu
#endif