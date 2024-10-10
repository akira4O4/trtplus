#ifndef IO_H
#define IO_H

#include "iostream"
// #include <chrono>
// #include <ctime>
#include <opencv2/core/core.hpp>

namespace input
{

struct NCHW
{
    std::string name;
    int         bs = 0;
    int         c  = 0; // channels
    int         h  = 0;
    int         w  = 0;

    size_t NxC() const { return bs * c; };

    size_t HxW() const { return w * h; };

    size_t CxHxW() const { return c * w * h; };

    size_t NxCxHxW() const { return bs * c * w * h; };

    size_t volume() const { return bs * c * w * h; };

    void print() const
    {
        std::cout << "Name: " << name << "\tShape: [ " << bs << "," << c << "," << h << "," << w << " ]" << std::endl;
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
    std::string name;
    int         bs = 0;
    int         nc = 0;

    size_t volume() const { return bs * nc; };

    void print() const { std::cout << "Name: " << name << "\tShape: [ " << bs << "," << nc << " ]" << std::endl; }
};

struct Segmentation
{
    std::string name;
    int         bs = 0;
    int         c  = 0; // channels or num_classes+1
    int         h  = 0;
    int         w  = 0;

    size_t volume() const { return bs * c * w * h; };

    void print() const
    {
        std::cout << "Name: " << name << "\tShape: [ " << bs << "," << c << "," << h << "," << w << " ]" << std::endl;
    }
};

struct YoloDetection
{
    std::string name;
    int         bs         = 0;
    int         rows       = 0;
    int         dimensions = 0;

    size_t volume() const { return bs * rows * dimensions; };

    void print() const
    {
        std::cout << "Name: " << name << "\tShape: [ " << bs << "," << rows << "," << dimensions << " ]" << std::endl;
    }
};

// struct YoloSegmentation
//{
//     int id = -1;
//     int bs = 0;
//     int c  = 0;
//     int h  = 0;
//     int w  = 0;
//
//     size_t volume() const { return bs * c * h * w; };
//
//     void print() const
//     {
//         std::cout << "Id: " << id << "\tShape: [ bs: " << bs << " c: " << c << " h: " << h << " w: " << w << " ]"
//                   << std::endl;
//     }
// };

} // namespace output

#endif // IO_H
