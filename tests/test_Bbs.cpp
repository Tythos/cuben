/**
 * tests/test_Bbp.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Bbs {
            TEST(TestBbs, EmptyTest) {
                cuben::Bbs prng;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(prng.roll(), 7.46207e-1, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(prng.roll(), 2.96331e-1, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(prng.roll(), 6.36535e-1, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
