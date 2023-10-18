/**
 * tests/test_RandomWalk.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_RandomWalk {
            TEST(TestRandomWalk, EmptyTest) {
                cuben::RandomWalk rw;
                const unsigned int nSteps = 10;
                Eigen::VectorXi walk = rw.getWalk(nSteps);
                std::cout << walk << std::endl;
                ASSERT_EQ(walk.size(), nSteps);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
