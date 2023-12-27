/**
 * tests/test_BrownianBridge.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_BrownianBridge {
            TEST(TestBrownianBridge, DefaultTest) {
                cuben::BrownianBridge bb;
                Eigen::Vector4f actual; actual <<
                    bb.t0, bb.x0, bb.tf, bb.xf;
                Eigen::Vector4f expected; expected <<
                    1.0f, 1.0f, 3.0f, 2.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestBrownianBridge, CustomTest) {
                cuben::BrownianBridge bb(2.0f, 3.0f, 5.0f, 7.0f);
                Eigen::Vector4f actual; actual <<
                    bb.t0, bb.x0, bb.tf, bb.xf;
                Eigen::Vector4f expected; expected <<
                    2.0f, 3.0f, 5.0f, 7.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestBrownianBridge, MultiplierTest) {
                cuben::BrownianBridge bb;
                Eigen::Vector2f actual; actual <<
                    bb.bbf(1.0f, 1.0f), bb.bbg(1.0f, 1.0f);
                Eigen::Vector2f expected; expected <<
                    0.5f, 1.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
