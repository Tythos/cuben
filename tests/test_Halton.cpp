/**
 * tests/test_Halton.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Halton {
            TEST(TestHalton, DefaultTest) {
                cuben::Halton h;
                ASSERT_TRUE(h.getBasePrime(), 2);
            }

            TEST(TestHalton, CustomTest) {
                cuben::Halton h(2357);
                ASSERT_TRUE(h.getBasePrime(), 2357);
            }

            TEST(TestHalton, RollAllTest) {
                cuben::Halton h(2357);
                Eigen::VectorXf all = h.rollAll(3);
                Eigen::VectorXf actual(3); actual <<
                    all[0], all[1], all[2];
                Eigen::VectorXf expected(3); expected <<
                    0.000424268f, 0.000848536f, 0.0012728f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
