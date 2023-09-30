/**
 * tests/test_contants.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_constants {
            TEST(TestConstants, BasicAssertions) {
                EXPECT_EQ(cuben::constants::iterTol, (float)1e-8);
                EXPECT_EQ(cuben::constants::zeroTol, (float)1e-8);
                EXPECT_EQ(cuben::constants::adaptiveTol, (float)1e-4);
                EXPECT_EQ(cuben::constants::relDiffEqTol, (float)1e-6);
                EXPECT_EQ(cuben::constants::bvpZeroTol, (float)1e-6);
                EXPECT_EQ(cuben::constants::iterLimit, (int)1e4);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
