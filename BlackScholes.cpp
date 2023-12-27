/**
 * BlackScholes.cpp
 */

#include "cuben.hpp"

cuben::BlackScholes::BlackScholes() {
    initPrice = 12.0f;
    strikePrice = 15.0f;
    interestRate = 0.05f;
    volatility = 0.25f;
}

cuben::BlackScholes::BlackScholes(float nip, float ncp, float nir, float nv) {
    initPrice = nip;
    strikePrice = ncp;
    interestRate = nir;
    volatility = nv;
}

Eigen::VectorXf cuben::BlackScholes::getWalk(float tf, float dt) {
    int nSteps = (int)std::ceil(tf / dt);
    float sqrtDt = std::sqrt(dt);
    Eigen::VectorXf xi = Eigen::VectorXf(nSteps + 1);
    cuben::Norm normPrng = cuben::Norm();
    xi(0) = initPrice;
    for (int i = 1; i <= nSteps; i++) {
        xi(i) = xi(i-1) + interestRate * xi(i-1) * dt + volatility * xi(i-1) * normPrng.roll() * sqrtDt;
    }
    return xi;
}

float cuben::BlackScholes::computeCallValue(float price, float tf) {
    float d1 = (std::log(price / strikePrice) + (interestRate + 0.5f * volatility * volatility) * tf) / (volatility * std::sqrt(tf));
    float d2 = (std::log(price / strikePrice) + (interestRate - 0.5f * volatility * volatility) * tf) / (volatility * std::sqrt(tf));
    cuben::Norm n = cuben::Norm();
    return price * n.cdf(d1) - strikePrice * std::exp(-interestRate * tf) * n.cdf(d2);
}

Eigen::Vector4f cuben::BlackScholes::getState() {
    Eigen::Vector4f state; state <<
        initPrice, strikePrice, interestRate, volatility;
    return state;
}
