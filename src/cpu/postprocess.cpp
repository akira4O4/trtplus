#include "postprocess.h"
namespace cpu
{
// output.shape[3,h,w]
cv::Mat mask2mat(const float* data, const std::array<int, 3> chw)
{
    int channel = chw[ 0 ];
    int height  = chw[ 1 ];
    int width   = chw[ 2 ];

    cv::Mat output(3, std::vector<int>{channel, height, width}.data(), CV_32F);

    for (int c = 0; c < channel; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                output.at<float>(c * height + h, w) = data[c * height * width + h * width + w];
            }
        }
    }

    return output;
}

cv::Mat de_normalize(const cv::Mat& input)
{
    cv::Mat output;
    input.convertTo(output, CV_8UC3, 255.0);
    return output;
}

cv::Mat transpose(const cv::Mat& data)
{
    cv::Mat out;
    cv::transpose(data, out);
    return out;
}

} // namespace cpu