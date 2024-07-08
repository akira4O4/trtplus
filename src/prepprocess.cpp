//
// Created by main on 24-4-11.
//

#include "prepprocess.h"

#include "memory.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
namespace preprocess
{
cv::Mat hwc2chw(cv::Mat img)
{
    // split output=Vec(Mat(b),Mat(g),Mat(r))
    std::vector<cv::Mat> chw(3);
    split(img, chw);

    cv::Mat out;
    merge(chw, out);
    return out;
}

void hwc2chw(cv::Mat& img, float* out)
{
    assert(kDefaultChannel == 3);

    // vec(mat(r),mat(g),mat(b))
    std::vector<cv::Mat> channels(kDefaultChannel);
    cv::split(img, channels);

    //    size_t h            = img.rows;
    //    size_t w            = img.cols;
    auto channel_size = img.total() * FLOAT32;
    for (int c = 0; c < kDefaultChannel; ++c)
    {
        cv::Mat channel = channels[ c ];

        // plan1
        auto offset = c * channel.total();
        std::memcpy(out + offset, channel.ptr<float>(0), channel_size);

        // plan2
        //        int i = 0; // num of pixel
        //        for (int row = 0; row < h; ++row)
        //        {
        //            for (int col = 0; col < w; ++col)
        //            {
        //                size_t idx = i + c * channel.total();
        //                out[ idx ] = channel.at<float>(row, col);
        //                ++i;
        //            }
        //        }
    }
}

// Split Channel : [RGB][RGB][RGB]->[R,R,R][G,G,G][B,B,B]
void hwc2chw(const result::NCHW& input_shape, const cv::Mat& data, float* ptr, int bs_offset)
{
    std::vector<cv::Mat> chw;
    auto                 input_wh = cv::Size(input_shape.w, input_shape.h);
    for (int i = 0; i < input_shape.c; ++i)
    {
        auto data = ptr + (i + bs_offset) * input_shape.HxW(FLOAT32);
        chw.emplace_back(input_wh, CV_32FC1, data);
    }
    split(data, chw);
}

cv::Mat bgr2rgb(const cv::Mat& input)
{
    cv::Mat output;
    if (input.channels() == 1)
        cv::cvtColor(input, output, cv::COLOR_GRAY2RGB);
    if (input.channels() == 3)
        cv::cvtColor(input, output, cv::COLOR_BGR2RGB);
    return output;
}

cv::Mat standardize_image(const cv::Mat& input)
{
    cv::Mat output;
    cv::subtract(input, cv::Scalar(0.485, 0.456, 0.406), output);
    cv::divide(output, cv::Scalar(0.229, 0.224, 0.225), output);
    return output;
}
cv::Mat normalize_image(const cv::Mat& input)
{
    cv::Mat output;
    //    input.convertTo(output, CV_32FC3, 1.0 / 255);
    input.convertTo(output, CV_32FC3, 1.f / 255.f);
    return output;
}

cv::Mat letterbox(const cv::Mat& input, const cv::Size2i& wh)
{
    cv::Mat output;
    int     in_w     = input.cols; // width
    int     in_h     = input.rows; // height
    int     tar_w    = wh.width;
    int     tar_h    = wh.height;
    float   r        = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int     inside_w = round(in_w * r);
    int     inside_h = round(in_h * r);
    int     padd_w   = tar_w - inside_w;
    int     padd_h   = tar_h - inside_h;

    cv::resize(input, output, cv::Size(inside_w, inside_h));

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;

    int top    = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left   = int(round(padd_w - 0.1));
    int right  = int(round(padd_w + 0.1));
    cv::copyMakeBorder(output, output, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return output;
}
cv::Mat resize(const cv::Mat& input, const cv::Size& wh)
{
    cv::Mat out;
    if (input.cols != wh.width || input.rows != wh.height)
    {
        cv::resize(input, out, wh);
        return out;
    }
    else
    {
        return input;
    }
}
} // namespace preprocess