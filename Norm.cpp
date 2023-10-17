/**
 * Norm.cpp
 */

#include "cuben.hpp"

cuben::Norm::Norm() {
    mean = 0.0f;
    variance = 1.0f;
}

float cuben::Norm::roll() {
    float u1 = cuben::rand::stdRoll();
    float u2 = cuben::rand::stdRoll();
    return mean + std::sqrt(variance) * std::sqrt(-2.0f * std::log(1.0f - u1)) * std::cos(2 * M_PI * u2);
}

float cuben::Norm::cdf(float x) {
    // Use 1.5e-7 approximation from Abramowitz and Stugen
    x = x / std::sqrt(2);
    bool isNeg = x < 0;
    if (isNeg) { x = -x; }
    float p = 0.3275911f;
    float a1 = 0.254829592f;
    float a2 = -0.284496736f;
    float a3 = 1.421413741f;
    float a4 = -1.453152027f;
    float a5 = 1.061405429f;
    float t = 1 / (1 + p * x);
    float N = 1 - (a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t) * std::exp(-(x * x));
    if (isNeg) { N = -N; }
    return 0.5f * (1.0f + N);
}
