/**
 * tests/test_pde.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

float sinBound(float x) {
    return std::sin(2 * M_PI * x) * std::sin(2 * M_PI * x);
}

float zeroBound(float t) {
    return 0.0f;
}

float expHeatForce(float u, float x, float t) {
    return 0.0f;
}

float impHeatForce(float x, float t) {
    return 0.0f;
}

float zeroForce(float x, float y) {
    return 0.0f;
}

float bc1(float y) {
    return 2 * std::log(y);
}

float bc2(float y) {
    return std::log(y * y + 1);
}

float bc3(float x) {
    return std::log(x * x + 1);
}

float bc4(float x) {
    return std::log(x * x + 4);
}

float f(float u, float x, float t) {
    return 0.0f;
}

float f2(float x, float t) {
    return 0.0f;
}

float ul(float t) {
    return 1.0f;
}

float ur(float t) {
    return 1.0f;
}

float u0(float x) {
    return 2.0f;
}

float fElliptic(float x, float y) {
    return 2.0f * (x * x + y * y);
}

float uxbcLower(float y) {
    return 0.0f;
}

float uxbcUpper(float y) {
    return y * y;
}

float uybcLower(float x) {
    return 0.0f;
}

float uybcUpper(float x) {
    return x * x;
}

namespace cuben {
    namespace tests {
        namespace test_pde {
            TEST(TestPde, OldTest) {
                Eigen::VectorXf coeffs(6); coeffs <<
                    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f;
                Eigen::Vector2f xBounds; xBounds <<
                    0.0f, 1.0f;
                Eigen::Vector2f yBounds; yBounds <<
                    1.0f, 2.0f;
                Eigen::MatrixXf uij = cuben::pde::finDiffElliptic(coeffs, zeroForce, bc1, bc2, bc3, bc4, xBounds, yBounds, 0.25f, 0.25f);
                Eigen::VectorXf rhs_actual = uij.col(uij.cols()-1);
                Eigen::VectorXf rhs_expected(5); rhs_expected <<
                    1.38629f, 1.4018f, 1.44692f, 1.51787f, 1.60944f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(rhs_actual, rhs_expected, 1e-3, true));
            }

            TEST(TestPde, ExpParaEulerTest) {
                Eigen::VectorXf coeffs(5); coeffs <<
                    0.0, 1.0, 1.0, 1.0, 0.0;
                Eigen::Vector2f xBounds(0.0, 1.0);
                Eigen::Vector2f tBounds(0.0, 1.0);
                float dx = 0.1f;
                float dt = 0.1f;
                Eigen::MatrixXf result = cuben::pde::expParaEuler(coeffs, f, ul, ur, u0, xBounds, tBounds, dx, dt);
                Eigen::VectorXf expected(10); expected <<
                    0.942207f, 2.34794f, 2.05242f, 0.307038f, 0.643947f, 0.778088f, 0.517269f, 3.92183f, 1.04833f, 1.64533f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result.col(result.cols()-1), expected, 1e-3, true));
            }

            TEST(TestPde, ImpParaEulerTest) {
                Eigen::VectorXf coeffs(6); coeffs <<
                    0.0, 0.5, 1.0, 1.0, 0.5, 0.0;
                Eigen::Vector2f xBounds(0.0, 1.0);
                Eigen::Vector2f tBounds(0.0, 1.0);
                float dx = 0.1f;
                float dt = 0.1f;
                Eigen::MatrixXf result = cuben::pde::impParaEuler(coeffs, f2, ul, ur, u0, xBounds, tBounds, dx, dt);
                Eigen::VectorXf expected(10); expected <<
                    0.958299f, 1.02272f, 1.07293f, 1.17446f, 1.44086f, 1.63766f, 1.65043f, 2.19677f, 1.354f, 2.56175f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result.col(result.cols()-1), expected, 1e-3, true));
            }

            TEST(TestPde, HeatCrankNicolsonTest) {
                float c = 1.0f;
                Eigen::Vector2f xBounds(0.0, 1.0);
                Eigen::Vector2f tBounds(0.0, 1.0);
                float dx = 0.1f;
                float dt = 0.1f;
                Eigen::MatrixXf result = cuben::pde::heatCrankNicolson(c, ul, ur, u0, xBounds, tBounds, dx, dt);                
                Eigen::VectorXf expected(10); expected <<
                    1.0f, 0.93312f, 1.03375f, 1.00946f, 0.992885f, 0.992885f, 1.00946f, 1.03375f, 0.93312f, 1.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(result.col(result.cols()-1), expected, 1e-3, true));
            }

            TEST(TestPde, FinDiffEllipticTest) {
                Eigen::VectorXf coeffs(6); coeffs <<
                    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f;
                Eigen::Vector2f xBounds(0.0f, 1.0f);
                Eigen::Vector2f yBounds(0.0f, 1.0f);
                float dx = 0.1f;
                float dy = 0.1f;
                const int N = 10;
                Eigen::MatrixXf result = cuben::pde::finDiffElliptic(coeffs, fElliptic, uxbcLower, uxbcUpper, uybcLower, uybcUpper, xBounds, yBounds, dx, dy);
                Eigen::VectorXf actual(N); for (int n = 0; n < N; n += 1) {
                    actual(n) = result(n,n);
                }
                Eigen::VectorXf expected(N); expected <<
                    0.0f, 0.00984028f, 0.0382893f, 0.0826319f, 0.139926f, 0.209046f, 0.292957f, 0.401282f, 0.555228f, 0.81f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
