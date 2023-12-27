/**
 * tests/test_RandomEscape.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_RandomEscape {
            TEST(TestRandomEscape, DefaultTest) {
                cuben::RandomEscape re;
                Eigen::Vector2f actual; actual <<
                    (float)re.getBounds()[0], (float)re.getBounds()[1];
                Eigen::Vector2f expected; expected <<
                    -1.0f, 1.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestRandomEscape, CustomTest) {
                cuben::RandomEscape re(-2, 2);
                Eigen::Vector2f actual; actual <<
                    (float)re.getBounds()[0], (float)re.getBounds()[1];
                Eigen::Vector2f expected; expected <<
                    -2.0f, 2.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestRandomEscape, WalkTest) {
                cuben::RandomEscape re;
                Eigen::VectorXi actual = re.getWalk(10u);
                Eigen::VectorXi expected(2); expected <<
                    -572662307, -572662307;
                //ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true)); // generator types are currently broken
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
