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

struct BoundingBox
{
    size_t      id = -1;
    XYWH        xywh;
    float       score = 0.0;
    std::string label_id;
};

struct Detection
{
    size_t                   id    = -1;
    std::string              name  = "";
    std::vector<BoundingBox> boxes = {};
    bool                     is_empty() const { return boxes.empty(); };
    size_t                   len() const { return boxes.size(); };
};

// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
struct YOLOv8Output
{

    std::string name       = "";
    int         id         = -1;
    int         bs         = 0;
    int         rows       = 0;
    int         dimensions = 0;

    size_t size(size_t byte = kINT8) const { return bs * rows * dimensions * byte; };
    void   info() const
    {
        std::cout << "Id: " << id << "\tName: " << name << "\tShape: [ " << bs << " " << rows << " " << dimensions
                  << " ]" << std::endl;
    }
};

struct NCHW
{
    std::string name;

    int id = -1;
    int bs = 0;
    int c  = 0;
    int h  = 0;
    int w  = 0;

    size_t NxC(size_t byte = kINT8) const { return bs * c * byte; };

    size_t HxW(size_t byte = kINT8) const { return w * h * byte; };

    size_t CxHxW(size_t byte = kINT8) const { return c * w * h * byte; };

    size_t NxCxHxW(size_t byte = kINT8) const { return bs * c * w * h * byte; };

    void info() const
    {
        //        INFO("ID:%d\tName:%s\tShape:\t[%d %d %d %d]", id, name, bs, c, h, w);
        std::cout << "Id: " << id << "\tName: " << name << "\tShape: [ " << bs << " " << c << " " << h << " " << w
                  << " " << " ]" << std::endl;
    }
};

} // namespace result

#endif // MAIN_RESULT_HPP
