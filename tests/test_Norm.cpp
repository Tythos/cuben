/**
 * tests/test_Norm.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Norm {
            TEST(TestNorm, DefaultTest) {
                // assert default distribution
                cuben::Norm n;
                Eigen::Vector2f actual; actual <<
                    n.mean, n.variance;
                Eigen::Vector2f expected; expected <<
                    0.0f, 1.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestNorm, CustomTest) {
                cuben::Norm n(1.2f, 3.4f);
                Eigen::Vector2f actual; actual <<
                    n.mean, n.variance;
                Eigen::Vector2f expected; expected <<
                    1.2f, 3.4f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestNorm, CdfTest) {
                cuben::Norm n;
                Eigen::VectorXf actual(3); actual <<
                    n.cdf(-1.0f), n.cdf(0.0f), n.cdf(1.0f);
                Eigen::VectorXf expected(3); expected <<
                    0.158655f, 0.5f, 0.841345f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
