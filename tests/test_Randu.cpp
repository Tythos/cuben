/**
 * tests/test_Randu.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Randu {
            TEST(TestRandu, EmptyTest) {
                ASSERT_TRUE(true);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
