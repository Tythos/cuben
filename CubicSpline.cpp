/**
 * CubicSpline.cpp
 */

#include "cuben.hpp"

cuben::CubicSpline::CubicSpline() {
    xi = Eigen::VectorXf();
    yi = Eigen::VectorXf();
    ec = cuben::interp::EndpointCondition::EC_NATURAL;
}

void cuben::CubicSpline::push(float x, float y) {
    int n = xi.rows();
    bool isInserted = false;
    Eigen::VectorXf xiNew(n+1);
    Eigen::VectorXf yiNew(n+1);
    for (int i = 0; i <= n; i++) {
        if (isInserted) {
            xiNew(i) = xi(i-1);
            yiNew(i) = yi(i-1);
        } else if (i == n || x < xi(i)) {
            xiNew(i) = x;
            yiNew(i) = y;
            isInserted = true;
        } else {
            xiNew(i) = xi(i);
            yiNew(i) = yi(i);
        }
    }
    xi = xiNew;
    yi = yiNew;
}

float cuben::CubicSpline::eval(float x) {
    static Eigen::VectorXf ai;
    static Eigen::VectorXf bi;
    static Eigen::VectorXf ci;
    static Eigen::VectorXf di;
    int n = xi.rows();
    int ndxLeft = 0;

    if (x < xi(0) || xi(n-1) < x) {
        throw cuben::exceptions::xOutOfInterpBounds();
    }

    if (ai.rows() != n) {
        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n,n);
        Eigen::VectorXf B = Eigen::VectorXf::Zero(n);
        float a, d, D;
        ai.resize(n);
        bi.resize(n);
        ci.resize(n);
        di.resize(n);
        
        // Apply endpoint conditions
        ai(0) = yi(0);
        ai(n-1) = yi(n-1);
        d = xi(1) - xi(0);
        D = yi(1) - yi(0);
        switch (ec) {
        case cuben::interp::EndpointCondition::EC_CLAMPED:
            A(0,0) = 2 * d;
            A(0,1) = d;
            A(n-1,n-2) = xi(n-1) - xi(n-2);
            A(n-1,n-1) = 2 * (xi(n-1) - xi(n-2));
            B(0) = 3 * D / d;
            B(n-1) = -3 * (yi(n-1) - yi(n-2)) / (xi(n-1) - xi(n-2));
            break;
        case cuben::interp::EndpointCondition::EC_PARABOLIC:
            A(0,0) = 1;
            A(0,1) = -1;
            A(n-1,n-2) = 1;
            A(n-1,n-1) = -1;
            B(0) = 0;
            B(n-1) = 0;
            break;
        case cuben::interp::EndpointCondition::EC_NOTAKNOT:
            A(0,0) = xi(3) - xi(2);
            A(0,1) = -(xi(2) - xi(1) + xi(3) - xi(2));
            A(0,2) = xi(2) - xi(1);
            A(n-1,n-3) = xi(n-1) - xi(n-2);
            A(n-1,n-2) = -(xi(n-2) - xi(n-3) + xi(n-1) - xi(n-2));
            A(n-1,n-1) = xi(n-2) - xi(n-3);
            B(0) = 0;
            B(n-1) = 0;
            break;
        case cuben::interp::EndpointCondition::EC_NATURAL:
        default:
            A(0,0) = 1.0f;
            A(n-1,n-1) = 1.0f;
            B(0) = 0;
            B(n-1) = 0;
            break;
        }

        // Loop through intermediate points
        for (int i = 1; i < n - 1; i++) {
            A(i,i-1) = d;
            A(i,i) = 2 * d + 2 * (xi(i+1) - xi(i));
            A(i,i+1) = xi(i+1) - xi(i);
            B(i) = 3 * ((yi(i+1) - yi(i)) / (xi(i+1) - xi(i)) - D / d);
            ai(i) = yi(i);
            d = xi(i+1) - xi(i);
            D = yi(i+1) - yi(i);
        }
        
        // For now, just invert to solve; then, compute other coeffs
        ci = A.inverse() * B;
        for (int i = 0; i < n - 1; i++) {
            di(i) = (ci(i+1) - ci(i)) / (3 * (xi(i+1) - xi(i)));
            bi(i) = (yi(i+1) - yi(i)) / (xi(i+1) - xi(i)) - (xi(i+1) - xi(i)) * (2 * ci(i) + ci(i+1)) / 3;
        }
    }
    while (ndxLeft < n - 1 && xi(ndxLeft+1) < x) {
        ndxLeft++;
    }
    return ai(ndxLeft) + bi(ndxLeft) * (x - xi(ndxLeft)) + ci(ndxLeft) * (x - xi(ndxLeft)) * (x - xi(ndxLeft)) + di(ndxLeft) * (x - xi(ndxLeft)) * (x - xi(ndxLeft)) * (x - xi(ndxLeft));
}
    
int cuben::CubicSpline::getNumPoints() {
    return xi.rows();
}
