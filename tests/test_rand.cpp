/**
 * tests/test_rand.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

float fTest(float t, float x) {
    return 0.1f * x;
}

float gTest(float t, float x) {
    return 0.3f * x;
}

float dgdxTest(float t, float x) {
    return 0.3f;
}

namespace cuben {
    namespace tests {
        namespace test_rand {
            TEST(TestRand, EmptyRandTest) {
                ASSERT_TRUE(true);
            }

            TEST(TestRand, StdRollTest) {
                float x = cuben::rand::stdRoll();
                std::cout << x << std::endl;
            }

            TEST(TestRand, EulerMaruTest) {
                Eigen::VectorXf ti(5); ti <<
                    0.1f, 0.2f, 0.3f, 0.4f, 0.5f;
                Eigen::VectorXf out = cuben::rand::eulerMaruyama(fTest, gTest, ti, 0.0f);
                std::cout << out << std::endl;
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
