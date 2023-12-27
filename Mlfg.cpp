/**
 * Mlfg.cpp
 */

#include "cuben.hpp"

cuben::Mlfg::Mlfg() {
    j = 418;
    k = 1279;
    stateVector = initialize(j, k);
}

cuben::Mlfg::Mlfg(unsigned int nj, unsigned int nk) {
    j = nj;
    k = nk;
    stateVector = initialize(j, k);
}

Eigen::VectorXf cuben::Mlfg::initialize(unsigned int j, unsigned int k) {
    Eigen::VectorXi piDigits(100); piDigits << 1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9;
    Eigen::VectorXi eDigits(100); eDigits << 7,1,8,2,8,1,8,2,8,4,5,9,0,4,5,2,3,5,3,6,0,2,8,7,4,7,1,3,5,2,6,6,2,4,9,7,7,5,7,2,4,7,0,9,3,6,9,9,9,5,9,5,7,4,9,6,6,9,6,7,6,2,7,7,2,4,0,7,6,6,3,0,3,5,3,5,4,7,5,9,4,5,7,1,3,8,2,1,7,8,5,2,5,1,6,6,4,2,7,4;
    Eigen::VectorXf secretSauce(100);
    Eigen::VectorXf si(k);
    for (int i = 0; i < 100; i++) {
        secretSauce(i) = (float)(10 * piDigits(i) + eDigits(i)) / 99.0f;
    }
    if (k <= 100) {
        si = secretSauce.segment(0,k);
    } else {
        int jReduced = (int)((float)j / 2.0f);
        int kReduced = (int)((float)k / 2.0f);
        si.segment(0,kReduced) = initialize(jReduced, kReduced);
        for (int i = kReduced; i < k; i++) {
            si(i) = 1024.0f * si(i - jReduced) * si(i - kReduced);
            while (si(i) > 1.0f) { si(i) = si(i) - 1.0f; }
        }
    }
    return si;
}

float cuben::Mlfg::roll() {
    nRolls++;
    unsigned int svNdx = nRolls % k;
    unsigned int jNdx = (nRolls - j) % k;
    stateVector(svNdx) = 1024.0f * stateVector(jNdx) * stateVector(svNdx);
    while (stateVector(svNdx) > 1.0f) { stateVector(svNdx) = stateVector(svNdx) - 1.0f; }
    return stateVector(svNdx);
}
