#ifndef MAIN_RESULT_HPP
#define MAIN_RESULT_HPP

// #include "iostream"
// #include <chrono>
// #include <ctime>
#include <opencv2/core/core.hpp>

namespace result
{

// Result output
struct Detection
{
    int         label_id = -1;
    std::string label    = "None";
    float       conf     = 0.0;
    cv::Rect    box      = {};
};

struct ClassificationOutput
{
    int id = -1;
    int bs = 0;
    int nc = 0;

    size_t NxNC() const { return bs * nc; };

    void info() const { std::cout << "Id: " << id << "\tShape: [ bs:" << bs << " nc: " << nc << " ]" << std::endl; }
};

struct YoloDetectionOutput
{
    int id         = -1;
    int bs         = 0;
    int rows       = 0;
    int dimensions = 0;

    size_t volume() const { return bs * rows * dimensions; };

    void info() const
    {
        std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " rows: " << rows << " dimensions: " << dimensions
                  << " ]" << std::endl;
    }
};

struct YoloSegmentationOutput
{
    int id = -1;
    int bs = 0;
    int nb = 0; // num of box
    int h  = 0;
    int w  = 0;

    size_t volume() const { return bs * nb * w * h; };

    void info() const
    {
        std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " nbox: " << nb << " h: " << h << " w: " << w << " ]"
                  << std::endl;
    }
};
struct NCHW
{
    int id = -1;
    int bs = 0;
    int c  = 0; // channels
    int h  = 0;
    int w  = 0;

    size_t NxC() const { return bs * c; };

    size_t HxW() const { return w * h; };

    size_t CxHxW() const { return c * w * h; };

    size_t NxCxHxW() const { return bs * c * w * h; };

    void info() const
    {
        std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " c: " << c << " h: " << h << " w: " << w << " ]"
                  << std::endl;
    }
};

} // namespace result

#endif // MAIN_RESULT_HPP
