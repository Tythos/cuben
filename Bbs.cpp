/**
 * Bbs.cpp
 */

#include "cuben.hpp"

cuben::Bbs::Bbs() {
    p = 167;
    q = 251;
    xPrev = 15204;
}

cuben::Bbs::Bbs(unsigned int np, unsigned int nq, unsigned int nx) {
    p = np;
    q = nq;
    xPrev = nx;
}

float cuben::Bbs::roll() {
    xPrev = (xPrev * xPrev) % (p * q);
    return (float)xPrev / (float)(p * q - 1);
}
