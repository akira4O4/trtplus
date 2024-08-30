#ifndef TRTPLUS_PRECISION_H
#define TRTPLUS_PRECISION_H

// #include "cuda_fp16.h"
#include "iostream"
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

// FP32=sign:8bits:23bits
// FP16=sign:5bits:10bits
// TF32=sign:8bits:10bits
// BF16=sign:8bits:7bits

typedef unsigned short bfloat16_t;
typedef unsigned short float16_t;
typedef float          float32_t;

constexpr uint8_t kINT8     = sizeof(uint8_t);    // size=1
constexpr uint8_t kFLOAT16  = sizeof(float16_t);  // size=2
constexpr uint8_t kBFLOAT16 = sizeof(bfloat16_t); // size=2
constexpr uint8_t kFLOAT32  = sizeof(float32_t);  // size=4

bfloat16_t fp32_to_bf16(float value);

float bf16_to_fp32(bfloat16_t value);

float16_t fp32_to_fp16(float value);

uint8_t fp32_to_int8(float value);

float fp16_to_fp32(float16_t value);

uint8_t fp16_to_int8(float16_t value);

#endif // TRTPLUS_PRECISION_CONVERSION_H
