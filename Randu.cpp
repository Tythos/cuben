/**
 * Randu.cpp
 */

#include "cuben.hpp"

cuben::Randu::Randu() {
    multiplier = 65539;
    offset = 0;
    modulus = std::pow(2, 31);
}

cuben::Randu::Randu(unsigned int s) {
    state = s;
    multiplier = 65539;
    offset = 0;
    modulus = std::pow(2, 31);
}
