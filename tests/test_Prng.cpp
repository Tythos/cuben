/**
 * tests/test_Prng.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Prng {
            TEST(TestPrng, EmptyTest) {
                cuben::Prng prng;
                ASSERT_EQ(prng.getState(), 1);
            }

            TEST(TestPrng, StateTest) {
                cuben::Prng prngWithState(5);
                ASSERT_EQ(prngWithState.getState(), 5);
            }

            TEST(TestPrng, RollTest) {
                cuben::Prng prng;
                float result = prng.roll();
                ASSERT_GE(result, 0.0f);
                ASSERT_LT(result, 1.0f);
            }

            TEST(TestPrng, IncTest) {
                cuben::Prng prng;
                prng.roll();
                ASSERT_EQ(prng.getRollCount(), 1);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
