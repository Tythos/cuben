/**
 * tests/test_Randu.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Randu {
            TEST(TestRandu, DefaultTest) {
                cuben::Randu randu;
                ASSERT_EQ(randu.getMultiplier(), 65539);
                ASSERT_EQ(randu.getOffset(), 0);
                ASSERT_EQ(randu.getModulus(), static_cast<unsigned int>(std::pow(2, 31)));
            }

            TEST(TestRandu, CustomTest) {
                cuben::Randu customRandu(5);
                ASSERT_EQ(customRandu.getState(), 5);
                ASSERT_EQ(customRandu.getMultiplier(), 65539);
                ASSERT_EQ(customRandu.getOffset(), 0);
                ASSERT_EQ(customRandu.getModulus(), static_cast<unsigned int>(std::pow(2, 31)));
            }

            TEST(TestRandu, RollTest) {
                cuben::Randu randu;
                float result = randu.roll();
                ASSERT_GE(result, 0.0f);
                ASSERT_LT(result, 1.0f);
            }

            TEST(TestRandu, CountTest) {
                cuben::Randu randu;
                randu.roll();
                ASSERT_EQ(randu.getRollCount(), 1);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
