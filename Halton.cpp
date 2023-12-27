/**
 * Halton.cpp
 */

#include "cuben.hpp"

cuben::Halton::Halton() {
    basePrime = 2;
}

cuben::Halton::Halton(unsigned int bp) {
    basePrime = bp;
}

float cuben::Halton::roll() {
    throw cuben::exceptions::xInvalidRoll();
}

Eigen::VectorXf cuben::Halton::rollAll(unsigned int numRolls) {
    float ratio = std::log((float)numRolls) / std::log((float)basePrime);
    unsigned int n = (unsigned int)std::ceil(ratio + cuben::fundamentals::relEps(ratio));
    Eigen::VectorXf b = Eigen::VectorXf::Zero(n);
    Eigen::VectorXf u = Eigen::VectorXf(numRolls);
    for (int j = 0; j < numRolls; j++) {
        int i = 0;
        b(0) = b(0) + 1;
        while (b(i) > (float)basePrime - 1.0f + cuben::fundamentals::machEps()) {
            b(i) = 0;
            i++;
            b(i) = b(i) + 1;
        }
        u(j) = 0.0f;
        for (int k = 0; k < b.rows(); k++) {
            //std::cout << u(j) << std::endl;
            u(j) = u(j) + b(k) * std::pow((float)basePrime, -(float)(k + 1.0f));
        }
    }
    return u;
}
