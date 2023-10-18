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
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
