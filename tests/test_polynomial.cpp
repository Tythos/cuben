/**
 * tests/test_polynomial.cpp
 */

#include <iostream>
#include "polynomial.cpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_constants {
            TEST(TestPolynomial, PushPrintEval) {
                // report and assert push construction
                cuben::polynomial::Polynomial p;
                p.push(0, 2);
                p.push(0, 3);
                p.push(0, -3);
                p.push(0, 5);
                p.push(0, -1);
                std::cout << "Number of points: " << p.size() << std::endl;
                ASSERT_EQ(p.size(), 5);

                // report (and assert?) representation
            	p.print();

                // report and assert evaluation
                float x = 0.5;
                std::cout << "P(" << x << ") = " << p.eval(x) << std::endl;
                ASSERT_EQ(p.eval(x), 1.25);
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
