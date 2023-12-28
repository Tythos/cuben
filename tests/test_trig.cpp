/**
 * tests/test_trig.cpp
 */

#include <iostream>
#include "gtest/gtest.h"
#include "cuben.hpp"

using namespace std::literals::complex_literals;

namespace cuben {
    namespace tests {
        namespace test_trig {
            TEST(TestTrig, SftTest) {
                Eigen::VectorXf inputVector(4); inputVector <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                Eigen::VectorXcf actual = cuben::trig::sft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +5.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -1.0f+1.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], -1.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[3], -1.0f-1.0if, 1e-3, true));
            }

            TEST(TestTrig, IsftTest) {
                Eigen::VectorXcf inputVector(4); inputVector <<
                    1.0f+2.0if, 3.0f+4.0if, 5.0f+6.0if, 7.0f+8.0if;
                Eigen::VectorXf result = cuben::trig::isft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(result[0], +8.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(result[1], +2.38419e-07f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(result[2], -2.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(result[3], -4.0f+0.0if, 1e-3, true));
            }

            TEST(TestTrig, FftTest1) {
                Eigen::VectorXf inputVector(2); inputVector <<
                    1.0f, 2.0f;
                Eigen::VectorXcf actual = cuben::trig::fft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +2.1232f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -0.707107f-1.73191e-16if, 1e-3, true));
            }

            TEST(TestTrig, FftTest2) {
                Eigen::VectorXf inputVector(4); inputVector <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                Eigen::VectorXcf actual = cuben::trig::fft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +5.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -1.0f+1.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], -1.0f+2.62268e-7if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[3], -1.0f-1.0if, 1e-3, true));
            }
            
            TEST(TestTrig, FftTest3) {
                Eigen::VectorXf inputVector(3); inputVector <<
                    1.0f, 2.0f, 3.0f;
                Eigen::VectorXcf actual = cuben::trig::fft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +3.4641f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -0.866025f+0.5if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], -0.866025f-0.5if, 1e-3, true));
            }

            TEST(TestTrig, FftRecTest1) {
                Eigen::VectorXf inputVector(2); inputVector <<
                    1.0f, 2.0f;
                Eigen::VectorXcf actual = cuben::trig::fftRec(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +3.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -1.0f-2.44929e-16if, 1e-3, true));
            }

            TEST(TestTrig, FftRecTest2) {
                Eigen::VectorXf inputVector(4); inputVector <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                Eigen::VectorXcf actual = cuben::trig::fftRec(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +10.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -2.0f+2.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], -2.0f+5.24537e-7if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[3], -2.0f-2.0if, 1e-3, true));
            }

            TEST(TestTrig, FftRecTest3) {
                Eigen::VectorXf inputVector(3); inputVector <<
                    1.0f, 2.0f, 3.0f;
                Eigen::VectorXcf actual = cuben::trig::fftRec(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +6.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -1.5f+0.866025if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], -1.5f-0.866025if, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
