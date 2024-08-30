//
// Created by seeking on 8/29/24.
//

#include "precision.h"

bfloat16_t fp32_to_bf16(float value)
{
    union {
        unsigned int u;
        float        f;
    } out{};
    out.f = value;
    return out.u >> 16;

    //    uint32_t temp;
    //    std::memcpy(&temp, &value, sizeof(temp));
    //     右移 16 位，将高 16 位作为 bfloat16
    //    return static_cast<unsigned short>(temp >> 16);
}

float bf16_to_fp32(bfloat16_t value)
{
    union {
        unsigned int u;
        float        f;
    } out{};
    out.u = value << 16;
    return out.f;

    //    uint32_t temp = static_cast<uint32_t>(value) << 16; // 恢复到32位格式
    //    float    result;
    //    std::memcpy(&result, &temp, sizeof(result));
    //    return result;
}

float16_t fp32_to_fp16(float value)
{
    uint32_t temp;
    std::memcpy(&temp, &value, sizeof(temp));

    // 提取符号、指数和尾数
    uint32_t sign     = (temp >> 16) & 0x8000;
    uint32_t exponent = ((temp >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (temp >> 13) & 0x3FF;

    if (exponent <= 0)
    {
        // Underflow to zero
        return static_cast<short>(sign);
    }
    else if (exponent >= 31)
    {
        // Overflow to infinity
        return static_cast<short>(sign | 0x7C00);
    }

    return static_cast<short>(sign | (exponent << 10) | mantissa);
}

uint8_t fp32_to_int8(float value)
{
    if (value > 127.0f)
        return 127;
    if (value < -128.0f)
        return -128;
    return static_cast<int>(std::round(value));
}

float fp16_to_fp32(float16_t value)
{
    uint32_t sign     = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7C00) >> 10;
    uint32_t mantissa = value & 0x3FF;

    if (exponent == 0x1F)
    {
        // NaN or Inf
        exponent = 0xFF;
        mantissa <<= 13;
    }
    else if (exponent == 0)
    {
        // Subnormal or Zero
        if (mantissa != 0)
        {
            // Normalize the mantissa
            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            exponent = 1;
        }
    }
    else
    {
        exponent = exponent - 15 + 127;
        mantissa <<= 13;
    }

    uint32_t temp = sign | (exponent << 23) | mantissa;
    float    result;
    std::memcpy(&result, &temp, sizeof(result));
    return result;
}

uint8_t fp16_to_int8(short value)
{
    float fp32_value = fp16_to_fp32(value);
    return fp32_to_int8(fp32_value);
}