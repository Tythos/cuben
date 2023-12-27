/**
 * Lgc.cpp
 */

#include "cuben.hpp"

cuben::Lcg::Lcg() {
    state = 1;
    multiplier = 13;
    offset = 0;
    modulus = 31;
}

cuben::Lcg::Lcg(unsigned int s, unsigned int m, unsigned int o, unsigned int mod) {
    state = s;
    multiplier = m;
    offset = o;
    modulus = mod;
}

float cuben::Lcg::roll() {
    nRolls++;
    state = (multiplier * state + offset) % modulus;
    return std::abs((float)state / modulus);
}
