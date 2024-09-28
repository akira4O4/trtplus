#ifndef IO_H
#define IO_H

// #include "iostream"
// #include <chrono>
// #include <ctime>
#include <opencv2/core/core.hpp>

namespace input
{

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

    size_t volume() const { return bs * c * w * h; };

    void print() const
    {
        std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " c: " << c << " h: " << h << " w: " << w << " ]"
                  << std::endl;
    }
};
} // namespace input

namespace output
{

// Result output
struct Detection
{
    int         label_id = -1;
    std::string label    = "None";
    float       conf     = 0.0;
    cv::Rect    box      = {0, 0, 0, 0};
};

struct Classification
{
    int id = -1;
    int bs = 0;
    int nc = 0;

    size_t volume() const { return bs * nc; };

    void print() const { std::cout << "Id: " << id << "\tShape: [ bs:" << bs << " nc: " << nc << " ]" << std::endl; }
};

struct YoloDetection
{
    int id         = -1;
    int bs         = 0;
    int rows       = 0;
    int dimensions = 0;

    size_t volume() const { return bs * rows * dimensions; };

    void print() const
    {
        std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " rows: " << rows << " dimensions: " << dimensions
                  << " ]" << std::endl;
    }
};

struct YoloSegmentation
{
    int id = -1;
    int bs = 0;
    int nb = 0; // num of box
    int h  = 0;
    int w  = 0;

    size_t volume() const { return bs * nb * w * h; };

    void print() const
    {
        std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " nbox: " << nb << " h: " << h << " w: " << w << " ]"
                  << std::endl;
    }
};

} // namespace output

#endif // IO_H
