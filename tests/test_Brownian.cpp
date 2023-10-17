/**
 * tests/test_Brownian.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Brownian {
            TEST(TestBrownian, EmptyTest) {
                ASSERT_TRUE(true);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
