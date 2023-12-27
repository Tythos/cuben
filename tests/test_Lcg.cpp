/**
 * tests/test_Lcg.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Lcg {
            TEST(TestLcg, StateTest) {
                cuben::Lcg lcg;
                ASSERT_EQ(lcg.getState(), 1);
                ASSERT_EQ(lcg.getMultiplier(), 13);
                ASSERT_EQ(lcg.getOffset(), 0);
                ASSERT_EQ(lcg.getModulus(), 31);
            }

            TEST(TestLcg, CustomTest) {
                cuben::Lcg customLcg(5, 7, 3, 20);
                ASSERT_EQ(customLcg.getState(), 5);
                ASSERT_EQ(customLcg.getMultiplier(), 7);
                ASSERT_EQ(customLcg.getOffset(), 3);
                ASSERT_EQ(customLcg.getModulus(), 20);
            }

            TEST(TestLcg, RollTest) {
                cuben::Lcg lcg;
                float result = lcg.roll();
                ASSERT_GE(result, 0.0f);
                ASSERT_LT(result, 1.0f);
            }

            TEST(TestLcg, CounterTest) {
                cuben::Lcg lcg;
                lcg.roll();
                ASSERT_EQ(lcg.getRollCount(), 1);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
