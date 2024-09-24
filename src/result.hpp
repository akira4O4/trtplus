//
// Created by seeking on 6/5/24.
//

#ifndef MAIN_RESULT_HPP
#define MAIN_RESULT_HPP
#include "iostream"
#include "utils.h"
#include <chrono>
#include <ctime>
#include <opencv2/core/core.hpp>
namespace result
{
struct XYXY
{
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    size_t area() const { return (x2 - x1) * (y2 - y1); };
};
struct XYWH
{
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    size_t area() const { return w * h; };
};

struct Box
{
    size_t      idx = -1;
    XYWH        xywh;
    XYXY        xyxy;
    float       score = 0.0;
    std::string label;
};

struct Detection
{
    size_t           idx = -1;
    std::string      name;
    std::vector<Box> boxes;
    bool             is_empty() const { return boxes.empty(); };
    size_t           len() const { return boxes.size(); };
};
struct NCHW
{
    std::string name;

    int    idx = -1;
    int    bs  = 0;
    int    c   = 0;
    int    h   = 0;
    int    w   = 0;
    size_t NxC(size_t byte = kINT8) const { return bs * c * byte; };

    size_t HxW(size_t byte = kINT8) const { return w * h * byte; };

    size_t CxHxW(size_t byte = kINT8) const { return c * w * h * byte; };

    size_t NxCxHxW(size_t byte = kINT8) const { return bs * c * w * h * byte; };

    void info() const
    {
        std::cout << "Idx: " << idx << "\tName: " << name << "\tShape: [ " << bs << " " << c << " " << h << " " << w
                  << " " << " ]" << std::endl;
    }
};

} // namespace result

#endif // MAIN_RESULT_HPP
