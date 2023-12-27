/**
 * MinStd.cpp
 */

#include "cuben.hpp"

cuben::MinStd::MinStd() {
    state = 1;
    multiplier = (unsigned int)std::pow(7, 5);
    offset = 0;
    modulus = (unsigned int)std::pow(2, 31) - 1;
}

cuben::MinStd::MinStd(unsigned int s) {
    state = s;
    multiplier = (unsigned int)std::pow(7, 5);
    offset = 0;
    modulus = (unsigned int)std::pow(2, 31) - 1;
}
