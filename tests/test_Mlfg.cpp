/**
 * tests/test_Mlfg.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_Mlfg {
            TEST(TestMlfg, DefaultTest) {
                cuben::Mlfg mlfg;
                ASSERT_EQ(mlfg.getPrimary(), 418);
                ASSERT_EQ(mlfg.getSecondary(), 1279);
            }

            TEST(TestMlfg, CustomTest) {
                cuben::Mlfg customMlfg(7, 13);                
                ASSERT_EQ(customMlfg.getPrimary(), 7);
                ASSERT_EQ(customMlfg.getSecondary(), 13);
            }

            TEST(TestMlfg, InitTest) {
                cuben::Mlfg mlfg;
                Eigen::VectorXf si = mlfg.initialize(mlfg.getPrimary(), mlfg.getSecondary());
                Eigen::VectorXf actual(3); actual <<
                    si[0], si[1], si[2];
                Eigen::VectorXf expected(3); expected <<
                    0.171717, 0.414141, 0.181818;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestMlfg, RollTest) {
                cuben::Mlfg mlfg;
                float result = mlfg.roll();
                ASSERT_GE(result, 0.0f);
                ASSERT_LT(result, 1.0f);
            }

            TEST(TestMlfg, CountTest) {
                cuben::Mlfg mlfg;
                mlfg.roll();
                ASSERT_EQ(mlfg.getRollCount(), 1);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
