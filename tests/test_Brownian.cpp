/**
 * tests/test_Brownian.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Brownian {
            TEST(TestBrownian, LengthTest) {
                cuben::Brownian b;
                Eigen::VectorXf xi(4); xi <<
                    0.0f, 1.0f, 2.0f, 3.0f;
                Eigen::VectorXf walk = b.sampleWalk(xi);
                ASSERT_EQ(walk.size(), 4);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
