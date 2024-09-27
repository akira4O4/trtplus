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

// Support v5 v8 v10
struct YOLOOutput
{
    int id         = -1;
    int bs         = 0;
    int rows       = 0;
    int dimensions = 0;

    void info() const
    {
        std::cout << "Id: " << id << "\tShape: [ " << bs << " " << rows << " " << dimensions << " ]" << std::endl;
    }
};
struct NCHW
{
    int id = -1;
    int bs = 0;
    int c  = 0;
    int h  = 0;
    int w  = 0;

    size_t NxC() const { return bs * c; };

    size_t HxW() const { return w * h; };

    size_t CxHxW() const { return c * w * h; };

    size_t NxCxHxW() const { return bs * c * w * h; };

    void info() const
    {
        std::cout << "Id: " << id << "\tShape: [ " << bs << " " << c << " " << h << " " << w << " " << " ]"
                  << std::endl;
    }
};

} // namespace result

#endif // MAIN_RESULT_HPP
