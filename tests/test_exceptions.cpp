/**
 * tests/test_contants.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_constants {
            TEST(TestExceptions, BasicAssertions) {
                EXPECT_STREQ(cuben::exceptions::xBisectionSign().what(), "Signs of f(a) and f(b) must be opposed and non-zero");
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
