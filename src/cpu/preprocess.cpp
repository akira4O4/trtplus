#include "preprocess.h"
#include "memory.h"
#include <opencv2/opencv.hpp>

namespace cpu
{

bool is_gray(const cv::Mat& image)
{
    return image.channels() == 1;
}

void hwc2chw(cv::Mat& img, float* out)
{
    CV_Assert(img.type() == CV_32FC3);
    int ic = img.channels();

    // vec(R,G,B)
    std::vector<cv::Mat> channels(ic);
    cv::split(img, channels);

    auto channel_size = img.total() * kFLOAT32;
    for (int c = 0; c < ic; ++c)
    {
        cv::Mat channel = channels[ c ];
        auto    offset  = c * channel.total();
        std::memcpy(out + offset, channel.ptr<float>(0), channel_size);
    }
}

cv::Mat hwc2chw(const cv::Mat& input)
{
    CV_Assert(input.type() == CV_32FC3);
    int height   = input.rows;
    int width    = input.cols;
    int channels = input.channels();

    cv::Mat output(3, new int[ 3 ]{channels, height, width}, CV_32F);

    std::vector<cv::Mat> input_channels(channels);
    cv::split(input, input_channels);

    for (int c = 0; c < channels; ++c)
    {
        std::memcpy(output.ptr<float>(c), input_channels[ c ].ptr<float>(0), height * width * sizeof(float));
    }

    return output;
}

cv::Mat bgr2rgb(const cv::Mat& input)
{

    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_BGR2RGB);

    return output;
}
cv::Mat rgb2bgr(const cv::Mat& input)
{

    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_RGB2BGR);

    return output;
}
cv::Mat gray2rgb(const cv::Mat& input)
{

    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_GRAY2RGB);

    return output;
}
cv::Mat gray2bgr(const cv::Mat& input)
{

    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);

    return output;
}

cv::Mat standardize(const cv::Mat& input, const cv::Scalar& mean, const cv::Scalar& std)
{
    cv::Mat output;
    cv::subtract(input, mean, output);
    cv::divide(output, std, output);
    return output;
}

cv::Mat normalize(const cv::Mat& input)
{
    cv::Mat output;
    input.convertTo(output, CV_32FC3, 1.f / 255.f);
    return output;
}

cv::Mat letterbox(const cv::Mat& input, const cv::Size& wh)
{
    if (input.cols == wh.width && input.rows == wh.height)
        return input;

    int iw = input.cols;
    int ih = input.rows;
    int ow = wh.width;
    int oh = wh.height;

    if (iw == ow && ih == oh)
        return input;

    float r = std::min(static_cast<float>(ow) / iw, static_cast<float>(oh) / ih);

    int inside_w = static_cast<int>(round(iw * r));
    int inside_h = static_cast<int>(round(ih * r));

    int padd_w = (ow - inside_w) / 2;
    int padd_h = (oh - inside_h) / 2;

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(inside_w, inside_h));

    cv::Mat output;
    cv::copyMakeBorder(resized, output, padd_h, oh - inside_h - padd_h, padd_w, ow - inside_w - padd_w,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return output;
}

cv::Mat resize(const cv::Mat& input, const cv::Size& wh)
{
    if (input.cols == wh.width && input.rows == wh.height)
        return input;

    cv::Mat out;
    cv::resize(input, out, wh);
    return out;
}

} // namespace cpu