#ifndef THREADS_FIXED_POINT_H
#define THREADS_FIXED_POINT_H

#include <stdint.h>

// Define a fixed-point data type as an int.
typedef int fixed_point;

// Define a constant with value 16384 as f_1714.
static int f_1714 = 16384;

// Convert an integer to fixed-point.
static inline fixed_point itof(int n)
{
return f_1714 * n;
}

// Convert a fixed-point number to an integer by dividing by f_1714.
static inline int ftoi(fixed_point x)
{
return x / f_1714;
}

// Convert a fixed-point number to an integer with rounding by adding or subtracting f_1714/2.
static inline int ftoi_round(fixed_point x)
{
int ftoi_r;
if (x >= 0)
{
ftoi_r = (x + f_1714 / 2) / f_1714;
}
else
{
ftoi_r = (x - f_1714 / 2) / f_1714;
}
return ftoi_r;
}

// Add an integer to a fixed-point number.
static inline fixed_point add_fi(fixed_point x, int n)
{
return x + n * f_1714;
}

// Subtract an integer from a fixed-point number.
static inline fixed_point sub_fi(fixed_point x, int n)
{
return x - n * f_1714;
}

// Multiply two fixed-point numbers by first converting them to 64-bit integers.
static inline fixed_point mul_ff(fixed_point x, fixed_point y)
{
return ((int64_t)x) * y / f_1714;
}

// Divide two fixed-point numbers by first converting them to 64-bit integers.
static inline fixed_point div_ff(fixed_point x, fixed_point y)
{
return ((int64_t)x) * f_1714 / y;
}

#endif