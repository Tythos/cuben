/**
 * tests/test_fundamentals.cpp
 */

#include "fundamentals.cpp"
#include "gtest/gtest.h"

namespace cuben {
	namespace tests {
		namespace test_fundamentals {
			TEST(TestFundamentals, AssertionTests) {
				float f1 = 1.0;
				float f2 = 1.1;
				float f3 = 2.0;
				ASSERT_EQ(cuben::fundamentals::isScalarWithinReltol(f2, f1, 5e-1), true);
				ASSERT_EQ(cuben::fundamentals::isScalarWithinReltol(f3, f1, 5e-1), false);
				Eigen::Vector3f v1(1.0, 1.0, 1.0);
				Eigen::Vector3f v2(1.1, 1.1, 1.1);
				Eigen::Vector3f v3(2.0, 2.0, 2.0);
				ASSERT_EQ(cuben::fundamentals::isVectorWithinReltol(v2, v1, 1e0), true);
				ASSERT_EQ(cuben::fundamentals::isVectorWithinReltol(v3, v1, 1e0), false);
			}

			TEST(TestFundamentals, ComputationalMetricsTest) {
				float inf = std::numeric_limits<float>::infinity();
				float nan = std::nan("");
				float fin = 3.14;
				std::cout << inf << " is infinity? " << (cuben::fundamentals::isInf(inf) == true ? "true" : "false") << std::endl;
				ASSERT_EQ(cuben::fundamentals::isInf(inf), true);
				std::cout << nan << " is nan? " << (cuben::fundamentals::isNan(nan) == true ? "true" : "false") << std::endl;
				ASSERT_EQ(cuben::fundamentals::isNan(nan), true);
				std::cout << fin << " is fin? " << (cuben::fundamentals::isFin(fin) == true ? "true" : "false") << std::endl;
				ASSERT_EQ(cuben::fundamentals::isFin(fin), true);
				std::cout << "Machine epsilon: " << cuben::fundamentals::machEps() << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(cuben::fundamentals::machEps(), 5.96046e-8, 1e-3));
				std::cout << "Relative epsilon (to 10.): " << cuben::fundamentals::relEps(10.) << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(cuben::fundamentals::relEps(10.), 2.98023e-7, 1e-3));
			}

			TEST(TestFundamentals, CustomVectorManipulations) {
				Eigen::VectorXf v(3);
				v << 1., 2., 4.;
				cuben::fundamentals::printVecTrans(v);
				std::cout << std::endl << "sigma(v) = " << cuben::fundamentals::stdDev(v) << std::endl;
				ASSERT_TRUE(cuben::fundamentals::isScalarWithinReltol(cuben::fundamentals::stdDev(v), 1.52753, 1e-3));
				float val = 2.0f;
				std::cout << "Value " << val << " is in index " << cuben::fundamentals::findValue(v, val) << std::endl;
				ASSERT_EQ(cuben::fundamentals::findValue(v, val), 1);
				Eigen::Vector2i sub; sub << 2, 1;
				Eigen::Vector2i dims; dims << 3, 3;
				int ndx = cuben::fundamentals::sub2ind(dims, sub);
				std::cout << "Subindex [" << sub(0) << "," << sub(1) << "], in a [" << dims(0) << " x " << dims(1) << "] matrix, is index " << ndx << std::endl;
				ASSERT_EQ(cuben::fundamentals::sub2ind(dims, sub), 7);
				ndx = 5; Eigen::Vector2i si = cuben::fundamentals::ind2sub(dims, ndx);
				std::cout << "Index " << ndx << ", in a [" << dims(0) << " x " << dims(1) << "] matrix, is subindex [" << si(0) << "," << si(1) << "]" << std::endl;
				ASSERT_EQ(si[0], 1);
				ASSERT_EQ(si[1], 2);
				Eigen::VectorXf rv = cuben::fundamentals::initRangeVec(0.1, 0.2, 0.9);
				std::cout << "Range vector: "; cuben::fundamentals::printVecTrans(rv); std::cout << std::endl;
				Eigen::VectorXf expected(5); expected << 0.1, 0.3, 0.5, 0.7, 0.9;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(rv, expected));
			}

			TEST(TestFundamentals, ResizingExpansions) {
				Eigen::VectorXf rv = cuben::fundamentals::initRangeVec(0.1, 0.2, 0.9);
				Eigen::VectorXf sr = cuben::fundamentals::safeResize(rv, 3);
				std::cout << "Range vector resized: fd"; cuben::fundamentals::printVecTrans(sr); std::cout << std::endl;
				Eigen::VectorXf expected(3); expected << 0.1, 0.3, 0.5;
				ASSERT_TRUE(cuben::fundamentals::isVectorWithinReltol(sr, expected));
				Eigen::MatrixXf rm(3,3); rm << 1.2, 3.4, 5.6, 7.8, 9.0, 0.9, 8.7, 6.5, 4.3;
				Eigen::Vector2i newSize; newSize << 3,2;
				Eigen::MatrixXf sm = cuben::fundamentals::safeResize(rm, newSize(0), newSize(1));
				std::cout << "Random matrix:" << std::endl << rm << std::endl;
				std::cout << "Resized to [" << newSize(0) << "," << newSize(1) << "], is:" << sm << std::endl;
				Eigen::MatrixXf vdm = cuben::fundamentals::vanDerMonde(sr);
				std::cout << "van der Monde of resized range vector:" << std::endl << vdm << std::endl;
			}
		}
	}
}

int main(int nArgs, char** vArgs) {
	::testing::InitGoogleTest(&nArgs, vArgs);
	return RUN_ALL_TESTS();
}
