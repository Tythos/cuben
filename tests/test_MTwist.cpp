/**
 * tests/test_MTwist.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_MTwist {
            TEST(TestMtwist, EmptyTest) {
                ASSERT_TRUE(true);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
