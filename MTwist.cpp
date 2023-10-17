/**
 * MTwist.cpp
 */

#include "cuben.hpp"

cuben::MTwist::MTwist(unsigned int seed) {
    w = 32;
    n = 624;
    m = 397;
    r = 31;
    a = 0x9908B0DF;
    u = 11;
    s = 7;
    b = 0x9D2C5680;
    t = 15;
    c = 0xEFC60000;
    l = 18;
    mask = (unsigned int)std::pow(2, w) - 1;
    pow = (unsigned int)std::pow(2, r);
    spread = 1812433253;
    stateVec = std::vector<unsigned int>(n);
    ndx = 0;

    // Loop through state vector and initialize from seed
    stateVec[0] = seed;
    for (int i = 1; i < n; i += 1) {
        stateVec[i] = (spread * (stateVec[i-1] ^ (stateVec[i-1] >> 30)) + i) & mask;
    }
}

void cuben::MTwist::generate() {
    for (int i = 0; i < n; i++) {
        unsigned int y = (stateVec[i] & pow) + (stateVec[(i + 1) % n] & (pow - 1));
        stateVec[i] = stateVec[(i + m) % n] ^ (y >> 1);
        if (y % 2 != 0) {
            stateVec[i] = stateVec[i] ^ a;
        }
    }
}

float cuben::MTwist::roll() {
    if (ndx == 0) {
        this->generate();
    }
    unsigned int y = stateVec[ndx];
    y ^= y >> u;
    y ^= y << s & b;
    y ^= y << t & c;
    y ^= y >> l;
    ndx = (ndx + 1) % n;
    return (float)y / (float)mask;
}
