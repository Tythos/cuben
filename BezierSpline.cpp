/**
 * BezierSpline.cpp
 */

#include "cuben.hpp"

cuben::BezierSpline::BezierSpline() {
    xi = Eigen::VectorXf();
    yi = Eigen::VectorXf();
    dydxi = Eigen::VectorXf();
}

void cuben::BezierSpline::push(float x, float y, float dydx) {
    // Store the current size of the vectors
    int currentSize = xi.size();

    // Resize the vectors to accommodate the new point and slope
    xi.conservativeResize(currentSize + 1);
    yi.conservativeResize(currentSize + 1);
    dydxi.conservativeResize(currentSize + 1);

    // Set the new values at the end of the vectors
    xi(currentSize) = x;
    yi(currentSize) = y;
    dydxi(currentSize) = dydx;
}

Eigen::Vector2f cuben::BezierSpline::eval(float t) {
    // Clamp t between 0 and the maximum index value (one less than number of points).
    int n = getNumPoints();
    t = std::max(0.0f, std::min(t, float(n - 1)));

    // Determine indices for control points
    int i0 = std::floor(t);
    int i1 = std::min(i0 + 1, n - 1);
    
    float local_t = t - i0;  // get the fractional part of t

    // If t is an exact integer, simply return the corresponding point
    if(local_t == 0) {
        return Eigen::Vector2f(xi(i0), yi(i0));
    }

    // Calculate control points for the segment
    Eigen::Vector2f P0(xi(i0), yi(i0));
    Eigen::Vector2f P3(xi(i1), yi(i1));
    Eigen::Vector2f P1 = P0 + dydxi(i0) * Eigen::Vector2f(1, dydxi(i0)) / 3.0; // using symmetric control points
    Eigen::Vector2f P2 = P3 - dydxi(i1) * Eigen::Vector2f(1, dydxi(i1)) / 3.0;

    // Cubic Bezier evaluation
    float one_minus_t = 1.0f - local_t;
    Eigen::Vector2f point = std::pow(one_minus_t, 3) * P0 
                            + 3 * std::pow(one_minus_t, 2) * local_t * P1 
                            + 3 * one_minus_t * std::pow(local_t, 2) * P2 
                            + std::pow(local_t, 3) * P3;

    return point;
}

int cuben::BezierSpline::getNumPoints() {
    return xi.rows();
}
