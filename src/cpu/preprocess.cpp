#include "preprocess.h"
#include "memory.h"
#include <opencv2/opencv.hpp>

namespace cpu
{

bool is_gray(const cv::Mat& image)
{
    return image.channels() == 1;
}

void hwc2chw_v1(const cv::Mat& img, float* out)
{
    assert(img.channels() == 3); // 确保图像有3个通道

    int    height       = img.rows;
    int    width        = img.cols;
    int    channels     = img.channels();
    size_t channel_size = height * width; // 每个通道的像素总数

    // 逐像素处理，将HWC转换为CHW
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                out[ c * channel_size + h * width + w ] = static_cast<float>(img.at<cv::Vec3b>(h, w)[ c ]);
            }
        }
    }
}

void hwc2chw_v2(cv::Mat& img, float* out)
{
    assert(img.channels() == 3);
    int ic = img.channels();

    // vec(mat(r),mat(g),mat(b))
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

void chw2hwc(cv::Mat& img, float* out)
{
    // 检查图像是否为三通道彩色图像
    if (img.channels() != 3)
    {
        std::cerr << "图像必须是三通道 (BGR/RGB) 彩色图像！" << std::endl;
        return;
    }

    int height   = img.rows;
    int width    = img.cols;
    int channels = img.channels();

    // 逐通道拷贝数据
    for (int c = 0; c < channels; ++c)
    {
        const uchar* src = img.ptr<uchar>(0) + c;
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                out[ h * width * channels + w * channels + c ] = static_cast<float>(src[ h * img.step + w * channels ]);
            }
        }
    }
}

cv::Mat image2rgb(const cv::Mat& input)
{
    cv::Mat output;
    if (input.channels() == 1)
        cv::cvtColor(input, output, cv::COLOR_GRAY2RGB);
    else if (input.channels() == 3)
        cv::cvtColor(input, output, cv::COLOR_BGR2RGB);
    return output;
}

cv::Mat image2bgr(const cv::Mat& input)
{
    cv::Mat output;
    if (input.channels() == 1)
        cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
    else if (input.channels() == 3)
        cv::cvtColor(input, output, cv::COLOR_RGB2BGR);
    return output;
}

cv::Mat standardize(const cv::Mat& input)
{
    cv::Mat output;
    cv::subtract(input, cv::Scalar(0.485, 0.456, 0.406), output);
    cv::divide(output, cv::Scalar(0.229, 0.224, 0.225), output);
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

    int iw = input.cols; // 输入图像的宽度
    int ih = input.rows; // 输入图像的高度
    int ow = wh.width;   // 目标宽度
    int oh = wh.height;  // 目标高度

    // 如果输入图像已经是目标尺寸，直接返回
    if (iw == ow && ih == oh)
        return input;

    // 计算缩放比例
    float r = std::min(static_cast<float>(ow) / iw, static_cast<float>(oh) / ih);

    // 计算缩放后的尺寸
    int inside_w = static_cast<int>(round(iw * r));
    int inside_h = static_cast<int>(round(ih * r));

    // 计算填充的边界
    int padd_w = (ow - inside_w) / 2;
    int padd_h = (oh - inside_h) / 2;

    // 调整图像尺寸
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(inside_w, inside_h));

    // 添加填充边界
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
