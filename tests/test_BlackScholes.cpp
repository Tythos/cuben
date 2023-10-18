/**
 * tests/test_BlackScholes.cpp
*/

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_BlackScholes {
            TEST(TestBlackScholes, OldTest) {
                cuben::BlackScholes bs = cuben::BlackScholes();
                Eigen::VectorXf xi = bs.getWalk(0.5f, 0.01f);
                float cv = bs.computeCallValue(12.0f, 0.5f);
                std::cout << "Simulated stock price over time:" << std::endl << xi << std::endl;
                std::cout << "Computes call values: " << cv << std::endl;
                ASSERT_TRUE(true);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
