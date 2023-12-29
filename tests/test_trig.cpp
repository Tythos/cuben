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
            
            TEST(TestTrig, IfftTest) {
                Eigen::VectorXcf inputVector(4); inputVector <<
                    1.0f+2.0if, 3.0f+4.0if, 5.0f+6.0if, 7.0f+8.0if;
                Eigen::VectorXf actual = cuben::trig::ifft(inputVector);
                Eigen::VectorXf expected(4); expected <<
                    8.0f, 0.0f, -2.0f, -4.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestTrig, FftRecTest4) {
                Eigen::VectorXcf inputVector(4); inputVector <<
                    1.0f+2.0if, 3.0f+4.0if, 5.0f+6.0if, 7.0f+8.0if;
                Eigen::VectorXcf actual = cuben::trig::ifft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], +8.0f+0.0if, 1e-3, true));
                //ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], +0.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], -2.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[3], -4.0f+0.0if, 1e-3, true));
            }

            TEST(TestTrig, DftTest) {
                Eigen::VectorXf inputVector(4); inputVector <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                Eigen::VectorXcf actual = cuben::trig::dft(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[0], -1.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[1], -1.0f-1.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[2], +5.0f+0.0if, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isComplexWithinReltol(actual[3], -1.0f+1.0if, 1e-3, true));
            }

            TEST(TestTrig, IdftTest) {
                Eigen::VectorXcf inputVector(4); inputVector <<
                    -1.0f+0.0if, -1.0f-1.0if, +5.0f+0.0if, -1.0f+1.0if;
                Eigen::VectorXf actual = cuben::trig::idft(inputVector);
                Eigen::VectorXf expected(4); expected <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestTrig, GenFreqVecTest) {
                Eigen::VectorXf actual = cuben::trig::genFreqVec(5, 10.0f);
                Eigen::VectorXf expected(5); expected <<
                    -4.0f, -2.0f, 0.0f, 2.0f, 4.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestTrig, GenTimeVecTest) {
                Eigen::VectorXf actual = cuben::trig::genTimeVec(5, 10.0f);
                Eigen::VectorXf expected(5); expected <<
                    0.0f, 0.1f, 0.2f, 0.3f, 0.4f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }

            TEST(TestTrig, DftInterpTest) { // dftInterp() function seems to stall
                Eigen::VectorXf tmi_s(4); tmi_s <<
                    0.0f, 1.0f, 2.0f, 3.0f;
                Eigen::VectorXf xmi(4); xmi <<
                    1.0f, 0.0f, -1.0f, 0.0f;
                unsigned int n = 8;
                Eigen::VectorXf tni_s;
                Eigen::VectorXf xni;
                cuben::trig::dftInterp(tmi_s, xmi, n, tni_s, xni);
                Eigen::VectorXf expectedTni_s(n); expectedTni_s <<
                    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f;
                Eigen::VectorXf expectedXni(n); expectedXni <<
                    1.0f, 0.707107f, 0.0f, -0.707107f, -1.0f, -0.707107f, 0.0f, 0.707107f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(tni_s, expectedTni_s, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xni, expectedXni, 1e-3, true));
            }

            TEST(TestTrig, DftFitTest) {
                Eigen::VectorXf tni_s(8); tni_s <<
                    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f;
                Eigen::VectorXf xni(8); xni <<
                    1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f;
                unsigned int m = 4;
                Eigen::VectorXf tmi_s;
                Eigen::VectorXf xmi;
                cuben::trig::dftFit(tni_s, xni, m, tmi_s, xmi);
                Eigen::VectorXf expectedTmi_s(m); expectedTmi_s <<
                    0.0f, 2.0f, 4.0f, 6.0f;
                Eigen::VectorXf expectedXmi(m); expectedXmi <<
                    0.5f, -0.5f, 0.5f, -0.5f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(tmi_s, expectedTmi_s, 1e-3, true));
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(xmi, expectedXmi, 1e-3, true));
            }

            TEST(TestTrig, WienerFilterTest) {
                Eigen::VectorXf xni(8); xni <<
                    1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f;
                float p = 0.5f;
                Eigen::VectorXf actual = cuben::trig::wienerFilter(xni, p);
                Eigen::VectorXf expected(8); expected <<
                    2.25f, 0.0f, 2.25f, 0.0f, 2.25f, 0.0f, 2.25f, 0.0f;
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actual, expected, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
