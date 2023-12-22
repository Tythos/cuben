/**
 * tests/test_JpegEncoder.cpp
 */

#include "cuben.hpp"
#include "gtest/gtest.h"

namespace cuben {
    namespace tests {
        namespace test_JpgEncoder {
            TEST(JpgEncoderTests, TestIitLen) {
                cuben::JpgEncoder je;
                ASSERT_TRUE(je.iitLen(4) == 3);
                ASSERT_TRUE(je.iitLen(-3) == 2);
                ASSERT_TRUE(je.iitLen(0) == 1);
            }

            TEST(JpgEncoderTests, TestDpcmEnc) {
                cuben::JpgEncoder jpgEncoder;
                unsigned int nBits;
                std::vector<char> bits;
                jpgEncoder.dpcmEnc(12, nBits, bits);
                EXPECT_EQ(nBits, 9);
                EXPECT_EQ(bits.size(), 2);
            }

            TEST(JpgEncoderTests, TestIitEnc) {
                cuben::JpgEncoder jpgEncoder;
                unsigned int nBits;
                std::vector<char> bits;
                jpgEncoder.iitEnc(-300, nBits, bits);
                ASSERT_EQ(nBits, 9);
                ASSERT_EQ(bits.size(), 2);
                ASSERT_EQ(bits[0], '\xff');
                ASSERT_EQ(bits[1], '\xd3');
            }

            TEST(JpgEncoderTests, TestRleEnc) {
                cuben::JpgEncoder jpgEncoder;
                unsigned int nBits;
                std::vector<char> bits;
                jpgEncoder.rleEnc(3, 5, nBits, bits);
                ASSERT_EQ(nBits, 0);
                ASSERT_EQ(bits.size(), 2);
                ASSERT_EQ(bits[0], '\0');
                ASSERT_EQ(bits[1], '\0');
            }
        }
    }
}

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
