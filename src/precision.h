#ifndef TRTPLUS_PRECISION_H
#define TRTPLUS_PRECISION_H

#include "iostream"
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

// FP32=sign:8bits:23bits
// FP16=sign:5bits:10bits
// TF32=sign:8bits:10bits
// BF32=sign:8bits:7bits

typedef unsigned short half;

constexpr uint8_t kINT8    = sizeof(uint8_t); // size=1
constexpr uint8_t kFLOAT16 = sizeof(half);    // size=2
constexpr uint8_t kFLOAT32 = sizeof(float);   // size=4

half fp32_to_bf16(float value);

float bf16_to_fp32(half value);

half fp32_to_fp16(float value);

uint8_t fp32_to_int8(float value);

float fp16_to_fp32(half value);

uint8_t fp16_to_int8(half value);

#endif // TRTPLUS_PRECISION_CONVERSION_H
