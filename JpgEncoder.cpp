/**
 * JpegEncoder.cpp
 */

#include "cuben.hpp"

cuben::JpgEncoder::JpgEncoder() {
    dpcmLengthTable = Eigen::VectorXi(13);
    dpcmHexTable = std::vector<char>(26, 0x00);
    rleLengthTable = Eigen::MatrixXi::Zero(10,8);
    rleHexTable = std::vector<char>(2 * 11 * 8, 0x00);
    
    // Populate lookup table of DPCM lengths
    dpcmLengthTable << 1,2,2,3,3,3,4,5,6,7,8,9,9;
    
    // Populate lookup table of DPCM values (9 bits length at most, expressed here as
    // 3 hexideicmal values, or 2 chars for each entry). Note that, since all vector
    // components were initialized to 0x00, we only need to specify the non-0x00 values.
    // Value n corresponds to entries in 2n and 2n+1.
    dpcmHexTable[3] = 0x02; // second half of value 1
    dpcmHexTable[5] = 0x03; // second half of value 2
    dpcmHexTable[7] = 0x04; // second half of value 3
    dpcmHexTable[9] = 0x05; // second half of value 4
    dpcmHexTable[11] = 0x06; // second half of value 5
    dpcmHexTable[13] = 0x0e; // second half of value 6
    dpcmHexTable[15] = 0x1e; // second half of value 7
    dpcmHexTable[17] = 0x3e; // second half of value 8
    dpcmHexTable[19] = 0x7e; // second half of value 9
    dpcmHexTable[21] = 0xfe; // second half of value 10
    dpcmHexTable[22] = 0x01; // first half of value 11
    dpcmHexTable[23] = 0xfe; // second half of value 11
    dpcmHexTable[24] = 0x01; // first half of value 12
    dpcmHexTable[25] = 0xff; // second half of value 12
    
    // Populate lookup table of RLE lengths
    rleLengthTable <<
        4,2,2,3,4,5,7,8,
        0,4,5,7,8,0,0,0,
        0,5,8,0,0,0,0,0,
        0,6,9,0,0,0,0,0,
        0,6,0,0,0,0,0,0,
        0,7,0,0,0,0,0,0,
        0,8,0,0,0,0,0,0,
        0,9,0,0,0,0,0,0,
        0,9,0,0,0,0,0,0,
        0,9,0,0,0,0,0,0;
    
    // Populate lookup table ot RLE values. This only includes a subset of the spanned
    // table area; less-frequently queried values are not defined here. In a 'real' JPG
    // compression, the Huffman tree would be compiled dynamically, customized to the
    // frequency of the actual run-length values within that particular file.
    
    // Firsr row (no zeros)
    rleHexTable[1] = 0x0a;
    rleHexTable[5] = 0x02;
    rleHexTable[7] = 0x04;
    rleHexTable[9] = 0x0b;
    rleHexTable[11] = 0x1a;
    rleHexTable[13] = 0x78;
    rleHexTable[15] = 0xf8;
    
    // Second row (1 zero)
    rleHexTable[19] = 0x0c;
    rleHexTable[21] = 0x1b;
    rleHexTable[23] = 0x79;
    rleHexTable[24] = 0x01; rleHexTable[25] = 0xf6;
    
    // Third and fourth rows (2 and 3 zeros)
    rleHexTable[35] = 0x1c;
    rleHexTable[37] = 0xf9;
    rleHexTable[51] = 0x3a;
    rleHexTable[52] = 0x01;
    rleHexTable[53] = 0xf7;
    
    // Fifth through eleventh rows (4 through 11 zeros)
    rleHexTable[67] = 0x3b;
    rleHexTable[83] = 0xa7;
    rleHexTable[99] = 0xa8;
    rleHexTable[115] = 0xfa;
    rleHexTable[130] = 0x01; rleHexTable[131] = 0xf8;
 
    rleHexTable[146] = 0x01; rleHexTable[147] = 0xf9;
    rleHexTable[162] = 0x01; rleHexTable[163] = 0xfa;
}

unsigned int cuben::JpgEncoder::iitLen(int value ) {
    return (unsigned int)std::floor(log2(std::abs(value))) + 1;
}

void cuben::JpgEncoder::dpcmEnc(int value, unsigned int &nBits, std::vector<char> &bits) {
    nBits = dpcmLengthTable[value];
    bits = std::vector<char>(2);
    bits[0] = dpcmHexTable[2 * value];
    bits[1] = dpcmHexTable[2 * value + 1];
}

void cuben::JpgEncoder::iitEnc(int value, unsigned int &nBits, std::vector<char> &bits) {
    nBits = iitLen(value);
    bits = std::vector<char>(2);
    bits[0] = 0xff00 & std::abs(value) >> 8;
    bits[1] = 0x00ff & std::abs(value);
    if (value < 0) {
        bits[0] = ~bits[0];
        bits[1] = ~bits[1];
    }
}

void cuben::JpgEncoder::rleEnc(unsigned int nZeros, unsigned int nLength, unsigned int &nBits, std::vector<char> &bits) {
    nBits = rleLengthTable(nZeros,nLength);
    bits = std::vector<char>(2);
    bits[0] = rleHexTable[2 * (8 * nZeros + nLength)];
    bits[1] = rleHexTable[2 * (8 * nZeros + nLength) + 1];
}
