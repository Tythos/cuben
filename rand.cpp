/**
 * rand.cpp
 */

#include "cuben.hpp"

float cuben::rand::stdRoll() {
    static unsigned int seed = (unsigned int)time(NULL);
    static MTwist mt = cuben::MTwist(seed);
    return mt.roll();
}

Eigen::VectorXf cuben::rand::eulerMaruyama(float(*f)(float, float), float(*g)(float, float), Eigen::VectorXf ti, float x0) {
    Eigen::VectorXf xi = Eigen::VectorXf(ti.rows());
    xi(0) = x0;
    Norm normPrng = cuben::Norm();
    for (int i = 1; i < ti.rows(); i++) {
        xi(i) = xi(i-1) + f(ti(i-1), xi(i-1)) * (ti(i) - ti(i-1)) + g(ti(i-1), xi(i-1)) * normPrng.roll() * std::sqrt(ti(i) - ti(i-1));
    }
    return xi;
}

Eigen::VectorXf cuben::rand::milstein(float(*f)(float, float), float(*g)(float,float), float(*dgdx)(float,float), Eigen::VectorXf ti, float x0) {
    Eigen::VectorXf xi = Eigen::VectorXf(ti.rows());
    xi(0) = x0;
    Norm normPrng = Norm();
    for (int i = 1; i < ti.rows(); i++) {
        float zi = normPrng.roll();
        xi(i) = xi(i-1) + f(ti(i-1), xi(i-1)) * (ti(i) - ti(i-1)) + g(ti(i-1), xi(i-1)) * zi * std::sqrt(ti(i) - ti(i-1)) + 0.5f * g(ti(i-1), xi(i-1)) * dgdx(ti(i-1), xi(i-1)) * (ti(i) - ti(i-1)) * (zi * zi - 1.0f);
    }
    return xi;
}
