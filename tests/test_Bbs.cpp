/**
 * tests/test_Bbp.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Bbs {
            TEST(TestBbs, DefaultTest) {
                cuben::Bbs prng;
                Eigen::VectorXf actual(3); actual <<
                    prng.roll(), prng.roll(), prng.roll();
                Eigen::VectorXf expected(3); expected << 
                    0.636535f, 0.296331f, 0.746207f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestBbs, CustomTest) {
                cuben::Bbs prng(2, 3, 5);
                Eigen::VectorXf actual(3); actual <<
                    prng.roll(), prng.roll(), prng.roll();
                Eigen::VectorXf expected(3); expected << 
                    0.2f, 0.2f, 0.2f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
