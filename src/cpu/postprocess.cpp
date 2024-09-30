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

std::vector<int> xyxy2xywh(std::vector<int> xyxy)
{
    int x = xyxy[ 0 ];
    int y = xyxy[ 1 ];
    int w = xyxy[ 2 ] - xyxy[ 0 ];
    int h = xyxy[ 3 ] - xyxy[ 1 ];

    std::vector<int> xywh{x, y, w, h};
    return xywh;
}

std::vector<int> xywh2xyxy(std::vector<int> xywh)
{
    int x1 = xywh[ 0 ];
    int y1 = xywh[ 1 ];
    int x2 = xywh[ 0 ] + xywh[ 2 ];
    int y2 = xywh[ 1 ] + xywh[ 3 ];

    std::vector<int> xyxy{x1, y1, x2, y2};
    return xyxy;
}

} // namespace cpu