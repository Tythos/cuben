/**
 * tests/test_Alfg.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Alfg {
            TEST(TestAlfg, EmptyTest) {
                cuben::Alfg prng; // using default paramaters
                float firstRoll = prng.roll();
                float secondRoll = prng.roll();
                float thirdRoll = prng.roll();
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(firstRoll, 2.02019e-2, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(secondRoll, 3.23232e-1, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(thirdRoll, 1.31313e-1, 1e-3, true));
            }

            TEST(TestAlfg, CustomTest) {
                cuben::Alfg customAlfg(7, 13);
                float firstRoll = customAlfg.roll();
                float secondRoll = customAlfg.roll();
                float thirdRoll = customAlfg.roll();
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(firstRoll, 9.39394e-1, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(secondRoll, 1.71717e-1, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(thirdRoll, 7.37374e-1, 1e-3, true));
            }

            TEST(TestAlfg, RollTest) {
                cuben::Alfg alfg;
                float result = alfg.roll();
                ASSERT_GE(result, 0.0f);
                ASSERT_LT(result, 1.0f);
            }

            TEST(TestAlfg, CountTest) {
                cuben::Alfg alfg;
                alfg.roll();
                ASSERT_EQ(alfg.getRollCount(), 1);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
