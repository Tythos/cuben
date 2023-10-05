/**
 * tests/test_CubicSpline.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_CubicSpline {
            TEST(TestCubicSpline, ConstructionTest) {
                cuben::CubicSpline cs;
                ASSERT_TRUE(true);
            }

            TEST(TestCubicSpline, InsertionTest) {
                cuben::CubicSpline cs;
                cs.push(1.0, 2.0);
                ASSERT_EQ(cs.getNumPoints(), 1);
                cs.push(3.0, 4.0);
                ASSERT_EQ(cs.getNumPoints(), 2);
                cs.push(5.0, 6.0);
                ASSERT_EQ(cs.getNumPoints(), 3);
                cs.push(7.0, 8.0);
                ASSERT_EQ(cs.getNumPoints(), 4);
            }

            TEST(TestCubicSpline, EvalTest) {
                cuben::CubicSpline cs;
                cs.push(1.0, 2.0);
                cs.push(3.0, 4.0);
                cs.push(5.0, 6.0);
                cs.push(7.0, 8.0);
                std::cout << "eval(4.0) = " << cs.eval(4.0) << std::endl;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(cs.eval(4.0), 5.0));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
