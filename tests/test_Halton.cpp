/**
 * tests/test_Halton.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Halton {
            TEST(TestHalton, EmptyTest) {
                ASSERT_TRUE(true);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
