/**
 * tests/test_MinStd.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_MinStd {
            TEST(TestMinStd, DefaultTest) {
                cuben::MinStd minStd;
                ASSERT_EQ(minStd.getState(), 1);
                ASSERT_EQ(minStd.getMultiplier(), static_cast<unsigned int>(std::pow(7, 5)));
                ASSERT_EQ(minStd.getOffset(), 0);
                ASSERT_EQ(minStd.getModulus(), static_cast<unsigned int>(std::pow(2, 31)) - 1);
            }

            TEST(TestMinStd, CustomTest) {
                cuben::MinStd customMinStd(5);
                ASSERT_EQ(customMinStd.getState(), 5);
                ASSERT_EQ(customMinStd.getMultiplier(), static_cast<unsigned int>(std::pow(7, 5)));
                ASSERT_EQ(customMinStd.getOffset(), 0);
                ASSERT_EQ(customMinStd.getModulus(), static_cast<unsigned int>(std::pow(2, 31)) - 1);
            }

            TEST(TestMinStd, RollTest) {
                cuben::MinStd minStd;
                float result = minStd.roll();
                ASSERT_GE(result, 0.0f);
                ASSERT_LT(result, 1.0f);
            }

            TEST(TestMinStd, CountTest) {
                cuben::MinStd minStd;
                minStd.roll();
                ASSERT_EQ(minStd.getRollCount(), 1);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
