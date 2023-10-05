/**
 * tests/test_BezierSpline.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_BezierSpline {
            TEST(TestBezierSpline, ConstructionTest) {
                cuben::BezierSpline bs;
                ASSERT_TRUE(true);
            }

            TEST(TestBezierSpline, InsertionTest) {
                cuben::BezierSpline bs;
                bs.push(0.0, 0.0, 1.0);
                ASSERT_EQ(bs.getNumPoints(), 1);
                bs.push(1.0, 2.0, 2.0);
                ASSERT_EQ(bs.getNumPoints(), 2);
                bs.push(3.0, 3.0, 0.5);
                ASSERT_EQ(bs.getNumPoints(), 3);
                bs.push(4.0, 2.5, -1.0);
                ASSERT_EQ(bs.getNumPoints(), 4);
                bs.push(5.0, 0.0, -2.0);
                ASSERT_EQ(bs.getNumPoints(), 5);
            }

            TEST(TestBezierSpline, EvalTestExact) {
                cuben::BezierSpline bs;
                bs.push(0.0, 0.0, 1.0);
                bs.push(1.0, 2.0, 2.0);
                bs.push(3.0, 3.0, 0.5);
                bs.push(4.0, 2.5, -1.0);
                bs.push(5.0, 0.0, -2.0);
                Eigen::Vector2f xy = bs.eval(1.0);
                std::cout << "eval(1.0) = "; cuben::fundamentals::printVecTrans(xy); std::cout << std::endl;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xy(0), 1.0));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xy(1), 2.0));
            }

            TEST(TestBezierSpline, EvalTestIntermediate) {
                cuben::BezierSpline bs;
                bs.push(0.0, 0.0, 1.0);
                bs.push(1.0, 2.0, 2.0);
                bs.push(3.0, 3.0, 0.5);
                bs.push(4.0, 2.5, -1.0);
                bs.push(5.0, 0.0, -2.0);
                Eigen::Vector2f xy = bs.eval(3.5);
                std::cout << "eval(3.5) = "; cuben::fundamentals::printVecTrans(xy); std::cout << std::endl;
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xy(0), 4.625, 1e-3));
                ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(xy(1), 0.875, 1e-3));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
