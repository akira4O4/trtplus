//
// Created by main on 24-4-11.
//

#ifndef MAIN_CPU_POSTPROCESS_H
#define MAIN_CPU_POSTPROCESS_H

#include "algorithm"
#include "iostream"
#include <cmath>
#include <functional>
#include <vector>

namespace cpu
{

// input: std::vector<T> list;
template <typename T>
inline size_t argmax(const T& data)
{
    if (data.begin() == data.end())
        throw std::invalid_argument("Input data is empty.");

    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

// input:T num[n]
template <typename T>
inline size_t argmax(const T* data, size_t len)
{
    if (len == 0)
        throw std::invalid_argument("Input data is empty");
    return std::distance(data, std::max_element(data, data + len));
}

template <typename T>
std::vector<T> softmax(const T* src, const int num_of_label, bool safe = false)
{
    T max_val = 0;
    if (safe)
    {
        max_val = *std::max_element(src, src + num_of_label);
    }

    std::vector<T> dst(num_of_label);
    T              denominator{0};

    for (auto i = 0; i < num_of_label; i++)
    {
        dst[ i ] = std::exp(src[ i ] - max_val);
        denominator += dst[ i ];
    }

    for (auto i = 0; i < num_of_label; i++)
    {
        dst[ i ] /= denominator;
    }

    return dst;
}

void nms(...);

void fast_nms(...);

} // namespace cpu
#endif