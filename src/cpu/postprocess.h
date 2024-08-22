//
// Created by main on 24-4-11.
//

#ifndef MAIN_CPU_POSTPROCESS_H
#define MAIN_CPU_POSTPROCESS_H

#include "algorithm"
#include "iostream"
#include "utils.h"
#include <cmath>
#include <functional>
#include <vector>

namespace cpu
{

template <typename T>
void softmax(const T* src, T* dst, const int num_of_label, bool safe = false)
{
    T max_val = 0;
    if (safe)
    {
        max_val = *std::max_element(src, src + num_of_label);
    }

    T denominator{0};
    for (auto i = 0; i < num_of_label; i++)
    {
        dst[ i ] = std::exp(src[ i ] - max_val);
        denominator += dst[ i ];
    }
    for (auto i = 0; i < num_of_label; i++)
    {
        dst[ i ] /= denominator;
    }
}

template <typename T>
inline size_t argmin_impl(T begin, T end)
{
    return std::distance(begin, std::min_element(begin, end));
}

template <typename T>
inline size_t argmax_impl(T begin, T end)
{
    return std::distance(begin, std::max_element(begin, end));
}

} // namespace cpu
#endif // MAIN_POSTPROCESS_H
