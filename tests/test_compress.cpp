/**
 * tests/test_compress.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

Eigen::MatrixXf loadTestMatrix() {
    Eigen::MatrixXf Xij = Eigen::MatrixXf(8,8);
    Xij.row(0) << -304.0f,210.0f,104.0f,-69.0f,10.0f,20.0f,-12.0f,7.0f;
    Xij.row(1) << -327.0f,-260.0f,67.0f,70.0f,-10.0f,-15.0f,21.0f,8.0f;
    Xij.row(2) << 93.0f,-84.0f,-66.0f,16.0f,24.0f,-2.0f,-5.0f,9.0f;
    Xij.row(3) << 89.0f,33.0f,-19.0f,-20.0f,-26.0f,21.0f,-3.0f,0.0f;
    Xij.row(4) << -9.0f,42.0f,18.0f,27.0f,-7.0f,-17.0f,29.0f,-7.0f;
    Xij.row(5) << -5.0f,15.0f,-10.0f,17.0f,32.0f,-15.0f,-4.0f,7.0f;
    Xij.row(6) << 10.0f,3.0f,-12.0f,-1.0f,2.0f,3.0f,-2.0f,-3.0f;
    Xij.row(7) << 12.0f,30.0f,0.0f,-3.0f,-3.0f,-6.0f,12.0f,-1.0f;
    return Xij;
}

bool test() {
    // Note: JpgEncoder not yet ready for prime time
/*			JpgEncoder je = JpgEncoder();
    unsigned int nBits;
    std::vector<char> bits;
    
    // Test with initial DC component -38
    je.dpcmEnc(-38, nBits, bits);
    std::cout << "-38 => " << nBits << " of 0x" << std::hex << bits << std::endl;
    
    // Test zero-run pair (0,4)
    je.rleEnc(0, 4, nBits, bits);
    std::cout << "(0,4) => " << nBits << " of 0x" << std::hex << bits << std::endl;
    
    // Test integer identifier for 13
    je.iitEnc(13, nBits, bits);
    std::cout << "13 => " << nBits << " of 0x" << std::hex << bits << std::endl;*/
    
    Eigen::MatrixXf Xij = loadTestMatrix();
    Eigen::MatrixXf Qij = cuben::compress::linearQuant(8, 1.0f);
    Eigen::MatrixXi Yij = cuben::compress::applyQuant(Xij, Qij);
    std::cout << "Quantization:" << std::endl << Qij << std::endl << std::endl;
    std::cout << "Compressed:" << std::endl << Yij << std::endl << std::endl;
    std::cout << "Decompressed:" << std::endl << cuben::compress::reverseQuant(Yij, Qij) << std::endl << std::endl;
    
    return true;
}

namespace cuben {
    namespace tests {
        namespace test_compress {
            TEST(TestCompress, BasicLinearQuantTest) {
                Eigen::MatrixXf Xij = loadTestMatrix();
                Eigen::MatrixXf Qij = cuben::compress::linearQuant(8, 1.0f);
                Eigen::VectorXf rowSums(8); rowSums << 
                    288.0f, 352.0f, 416.0f, 480.0f, 544.0f, 608.0f, 672.0f, 736.0f;
    			//std::cout << "Quantization:" << std::endl << Qij << std::endl << std::endl;
                for (int i = 0; i < rowSums.size(); i += 1) {
                    ASSERT_TRUE(abs(Qij.row(i).sum() - rowSums[i]) / rowSums[i] < 1e-3);
                }
            }

            TEST(TestCompress, BasicApplyQuantTest) {
                Eigen::MatrixXf Xij = loadTestMatrix();
                Eigen::MatrixXf Qij = cuben::compress::linearQuant(8, 1.0f);
                Eigen::MatrixXi Yij = cuben::compress::applyQuant(Xij, Qij);
                Eigen::VectorXf rowSums(8); rowSums <<
                    -23.0f, -27.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f;
                //std::cout << "Compressed:" << std::endl << Yij << std::endl << std::endl;
                for (int i = 0; i < rowSums.size(); i += 1) {
                    ASSERT_TRUE(abs(Yij.row(i).sum() - rowSums[i]) < 1e-3);
                }
            }

            TEST(TestCompress, BasicReverseQuantTest) {
                Eigen::MatrixXf Xij = loadTestMatrix();
                Eigen::MatrixXf Qij = cuben::compress::linearQuant(8, 1.0f);
                Eigen::MatrixXi Yij = cuben::compress::applyQuant(Xij, Qij);
                Eigen::MatrixXf Iij = cuben::compress::reverseQuant(Yij, Qij);
                Eigen::VectorXf rowSums(8); rowSums <<
                    -64.0f, -456.0f, -32.0f, 64.0f, 0.0f, 0.0f, 0.0f, 0.0f;
                //std::cout << "Decompressed:" << std::endl << Iij << std::endl << std::endl;
                for (int i = 0; i < rowSums.size(); i += 1) {
                    ASSERT_TRUE(abs(Iij.row(i).sum() - rowSums[i]) < 1e-3);
                }
            }

            TEST(TestCompress, DctTest) {
                Eigen::VectorXf inputVector(4); inputVector <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                Eigen::VectorXf expectedOutput(4); expectedOutput <<
                    5.0f, -2.23044f, 0.0f, -0.158513;
                Eigen::VectorXf actualOutput = cuben::compress::dct(inputVector);
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actualOutput, expectedOutput, 1e-3, true));
            }

            TEST(TestCompress, DctInterpTest) {
                float t = 0.5f;
                Eigen::VectorXf inputVector(4); inputVector <<
                    1.0f, 2.0f, 3.0f, 4.0f;
                float expectedOutput = 2.07107e-1;
                float actualOutput = cuben::compress::dctInterp(inputVector, t);
                ASSERT_TRUE(abs(expectedOutput - actualOutput) / expectedOutput < 1e-3);
            }

            TEST(TestCompress, DctFitTest) {
                Eigen::VectorXi inputVectorXi(4); inputVectorXi <<
                    1, 2, 3, 4;
                unsigned int n = 2;
                Eigen::VectorXf expectedOutput(n); expectedOutput <<
                    5.0f, -2.23044;
                Eigen::VectorXf actualOutput = cuben::compress::dctFit(inputVectorXi, n);
                ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(actualOutput, expectedOutput, 1e-3, true));
            }

            TEST(TestCompress, Dct2dTest) {
                Eigen::MatrixXf inputMatrix(4, 4); inputMatrix <<
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f;
                unsigned int n = 4;
                Eigen::MatrixXf expectedOutput(n, n); expectedOutput <<
                    34.0f, -4.46088f, 0.0f, -0.317026,
                    -17.8435, 9.53674e-7, 4.76837e-7, 2.38419e-7,
                    0.0f, 0.0f, 0.0f, 0.0f,
                    -1.2681f, 0.0f, 0.0f, 0.0f;
                Eigen::MatrixXf actualOutput = cuben::compress::dct2d(inputMatrix);
                ASSERT_EQ(actualOutput.rows(), n);
                ASSERT_EQ(actualOutput.cols(), n);
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(actualOutput, expectedOutput, 1e-3, true));
            }

            TEST(TestCompress, Idct2dTest) {
                Eigen::MatrixXf inputMatrix(4, 4); inputMatrix <<
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f;
                unsigned int n = 4;
                Eigen::MatrixXf expectedOutput(n, n); expectedOutput <<
                    27.4139, -9.6834, 5.83564, 0.00226831,
                    -22.3747, 5.2921, -4.52673, -0.66998,
                    6.98369, -2.23063, 1.46526, 0.0610162,
                    -3.24491, 0.47807, -0.630311, -0.171214;
                Eigen::MatrixXf actualOutput = cuben::compress::idct2d(inputMatrix);
                ASSERT_EQ(actualOutput.rows(), n);
                ASSERT_EQ(actualOutput.cols(), n);
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(actualOutput, expectedOutput, 1e-3, true));
            }

            TEST(TestCompress, ApplyQuantTest) {
                Eigen::MatrixXf inputMatrixXij(4, 4); inputMatrixXij <<
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f;
                Eigen::MatrixXf inputMatrixQij(4, 4); inputMatrixQij <<
                    1.0f, 2.0f, 1.0f, 2.0f,
                    2.0f, 1.0f, 2.0f, 1.0f,
                    1.0f, 2.0f, 1.0f, 2.0f,
                    2.0f, 1.0f, 2.0f, 1.0f;
                const int n = 4;
                Eigen::MatrixXi expectedOutput(n, n); expectedOutput <<
                    1, 1, 3, 2,
                    2, 6, 3, 8,
                    9, 5, 11, 6,
                    6, 14, 7, 16;
                Eigen::MatrixXi actualOutput = cuben::compress::applyQuant(inputMatrixXij, inputMatrixQij);
                ASSERT_EQ(actualOutput.rows(), n);
                ASSERT_EQ(actualOutput.cols(), n);
                ASSERT_TRUE(actualOutput.isApprox(expectedOutput));
            }

            TEST(TestCompress, ReverseQuantTest) {
                Eigen::MatrixXi inputMatrixYij(4, 4); inputMatrixYij <<
                    1, 1, 3, 2,
                    2, 6, 3, 8,
                    9, 5, 11, 6,
                    6, 14, 7, 16;
                Eigen::MatrixXf inputMatrixQij(4, 4); inputMatrixQij <<
                    1.0f, 2.0f, 1.0f, 2.0f,
                    2.0f, 1.0f, 2.0f, 1.0f,
                    1.0f, 2.0f, 1.0f, 2.0f,
                    2.0f, 1.0f, 2.0f, 1.0f;
                unsigned int n = 4;
                Eigen::MatrixXf expectedOutput(n, n); expectedOutput <<
                    1.0f, 2.0f, 3.0f, 4.0f,
                    4.0f, 6.0f, 6.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    12.0f, 14.0f, 14.0f, 16.0f;
                Eigen::MatrixXf actualOutput = cuben::compress::reverseQuant(inputMatrixYij, inputMatrixQij);
                ASSERT_EQ(actualOutput.rows(), n);
                ASSERT_EQ(actualOutput.cols(), n);
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(actualOutput, expectedOutput, 1e-3, true));
            }

            TEST(TestCompress, LinearQuantTest) {
                const int n = 4;
                float p = 0.5f;
                Eigen::MatrixXf expectedOutput(n, n); expectedOutput <<
                    8.0f * p, 16.0f * p, 24.0f * p, 32.0f * p,
                    16.0f * p, 24.0f * p, 32.0f * p, 40.0f * p,
                    24.0f * p, 32.0f * p, 40.0f * p, 48.0f * p,
                    32.0f * p, 40.0f * p, 48.0f * p, 56.0f * p;
                Eigen::MatrixXf actualOutput = cuben::compress::linearQuant(n, p);
                ASSERT_EQ(actualOutput.rows(), n);
                ASSERT_EQ(actualOutput.cols(), n);
                ASSERT_TRUE(cuben::fundamentals::isMatrixWithinReltol(actualOutput, expectedOutput, 1e-3, true));
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
