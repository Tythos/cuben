/**
 * tests/test_BlackScholes.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_BlackScholes {
            // TEST(TestBlackScholes, OldTest) {
            //     cuben::BlackScholes bs = cuben::BlackScholes();
            //     Eigen::VectorXf xi = bs.getWalk(0.5f, 0.01f);
            //     float cv = bs.computeCallValue(12.0f, 0.5f);
            //     std::cout << "Simulated stock price over time:" << std::endl << xi << std::endl;
            //     std::cout << "Computes call values: " << cv << std::endl;
            //     ASSERT_TRUE(true);
            // }

            // TEST(TestBlackScholes, WalkTest) {
            //     cuben::BlackScholes bs = cuben::BlackScholes();
            //     Eigen::VectorXf xi = bs.getWalk(0.5f, 0.01f);
            //     Eigen::VectorXf actual(3); actual <<
            //         xi[0], xi[1], xi[2];
            //     Eigen::VectorXf expected(3); expected <<
            //         12.0f, 12.1884f, 12.1532f;
            //     ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            // }

            TEST(TestBlackScholes, DefaultTest) {
                Eigen::Vector4f actual = cuben::BlackScholes().getState();
                Eigen::Vector4f expected; expected <<
                    12.0f, 15.0f, 0.05f, 0.25f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestBlackScholes, CustomTest) {
                Eigen::Vector4f actual = cuben::BlackScholes(2.0f, 3.0f, 0.4f, 0.5f).getState();
                Eigen::Vector4f expected; expected <<
                    2.0f, 3.0f, 0.4f, 0.5f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestBlackScholes, CallTest) {
                cuben::BlackScholes bs = cuben::BlackScholes();
                float cv = bs.computeCallValue(12.0f, 0.5f);
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(cv, 0.153805f, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
