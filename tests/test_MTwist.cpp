/**
 * tests/test_MTwist.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_MTwist {
            TEST(TestMtwist, StateTest) {
                cuben::MTwist mt(123);
                const std::vector<unsigned int> sv = mt.getStateVec();\
                ASSERT_EQ(sv[0], 123);
                ASSERT_EQ(sv[1], 3885958024);
                ASSERT_EQ(sv[2], 930007257);
            }

            TEST(TestMtwist, RollTest) {
                cuben::MTwist mt(123);
                Eigen::VectorXf actual(3); actual <<
                    mt.roll(), mt.roll(), mt.roll();
                Eigen::VectorXf expected(3); expected <<
                    0.286139f, 0.712955f, 0.696469f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
