/*	Cuben::Compress static object
	Compression and decompression routines
	Derived from Chapter 11 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "../inc/Cuben.h"

namespace Cuben {
	namespace Compress {
		Eigen::VectorXf dct(Eigen::VectorXf xi) {
			unsigned int n = xi.rows();
			float sqrt2 = std::sqrt(2.0f);
			float c = sqrt2 / std::sqrt((float)n);
			Eigen::MatrixXf C = Eigen::MatrixXf(n,n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i == 0) {
						C(i,j) = c * 0.5f * sqrt2;
					} else {
						C(i,j) = c * std::cos(((float)i * ((float)j + 0.5f) * M_PI) / (float)n);
					}
				}
			}
			return C * xi;
		}
		
		float dctInterp(Eigen::VectorXf Xi, float t) {
			unsigned int n = Xi.rows();
			float invSqrtN = 1.0f / std::sqrt((float)n);
			float c = std::sqrt(2) * invSqrtN;
			float y = invSqrtN * Xi(0);
			for (int i = 0; i < n; i++) {
				y += c * Xi(i) * std::cos(0.5f * (float)i * (2.0f * t + 1.0f) * M_PI / (float)n);
			}
			return y;
		}
		
		Eigen::VectorXf dctFit(Eigen::VectorXf xi, unsigned int n) {
			Eigen::VectorXf Xi = dct(xi);
			return Xi.segment(0,n);
		}
		
		Eigen::MatrixXf dct2d(Eigen::MatrixXf xij) {
			unsigned int n = xij.rows();
			float sqrtInvN = std::sqrt(1.0f / (float)n);
			float sqrt2 = std::sqrt(2.0f);
			if (n != xij.cols()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::MatrixXf C = Eigen::MatrixXf(n,n);
			
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i == 0) {
						C(i,j) = sqrtInvN;
					} else {
						C(i,j) = sqrt2 * sqrtInvN * std::cos(i * (j + 0.5f) * M_PI / (float)n);
					}
				}
			}
			return C * xij * C.transpose();
		}
		
		Eigen::MatrixXf idct2d(Eigen::MatrixXf Xij) {
			unsigned int n = Xij.rows();
			float sqrtInvN = std::sqrt(1.0f / (float)n);
			float sqrt2 = std::sqrt(2.0f);
			if (n != Xij.cols()) {
				throw Cuben::xMismatchedDims();
			}
			Eigen::MatrixXf C = Eigen::MatrixXf(n,n);
			
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i == 0) {
						C(i,j) = sqrtInvN * std::cos(0.5f * i * (2.0f * j + 1.0f) * M_PI / (float)n);
					} else {
						C(i,j) = sqrt2 * sqrtInvN * std::cos(0.5f * i * (2.0f * j + 1.0f) * M_PI / (float)n);
					}
				}
			}
			return C.transpose() * Xij * C;
		}
		
		Eigen::MatrixXi applyQuant(Eigen::MatrixXf Xij, Eigen::MatrixXf Qij) {
			// Compresses the given matrix by applying quantization defined by Qij
			unsigned int n = Xij.rows();
			Eigen::MatrixXi Yij = Eigen::MatrixXi(n,n);
			if (n != Xij.cols() || n != Qij.rows() || n != Qij.cols()) {
				throw Cuben::xMismatchedDims();
			}
			
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					Yij(i,j) = (int)(Xij(i,j) / Qij(i,j));
				}
			}
			return Yij;
		}
		
		Eigen::MatrixXf reverseQuant(Eigen::MatrixXi Yij, Eigen::MatrixXf Qij) {
			// Decompress the given matrix by reversing quantization defined by Qij
			unsigned int n = Yij.rows();
			Eigen::MatrixXf Xij = Eigen::MatrixXf(n,n);
			if (n != Yij.cols() || n != Qij.rows() || n != Qij.cols()) {
				throw Cuben::xMismatchedDims();
			}
			
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					Xij(i,j) = (float)Yij(i,j) * Qij(i,j);
				}
			}
			return Xij;
		}
		
		Eigen::MatrixXf linearQuant(unsigned int n, float p) {
			// Constructs a linear quantization matrix of size n x n; compression
			// is controlled by the parameter p (greater p = greater compression)
			Eigen::MatrixXf Qij = Eigen::MatrixXf(n,n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					Qij(i,j) = 8.0f * p * ((float)i + (float)j + 1.0f);
				}
			}
			return Qij;
		}
		
		Eigen::MatrixXf jpegQuant(float p) {
			Eigen::MatrixXf Qij = Eigen::MatrixXf(8,8);
			Qij.row(0) << 16.0f,11.0f,10.0f,16.0f,24.0f,40.0f,51.0f,61.0f;
			Qij.row(1) << 12.0f,12.0f,14.0f,19.0f,26.0f,58.0f,60.0f,55.0f;
			Qij.row(2) << 14.0f,13.0f,16.0f,24.0f,40.0f,57.0f,69.0f,56.0f;
			Qij.row(3) << 14.0f,17.0f,22.0f,29.0f,51.0f,87.0f,80.0f,62.0f;
			Qij.row(4) << 18.0f,22.0f,37.0f,56.0f,68.0f,109.0f,103.0f,77.0f;
			Qij.row(5) << 24.0f,35.0f,55.0f,64.0f,81.0f,104.0f,113.0f,92.0f;
			Qij.row(6) << 49.0f,64.0f,78.0f,87.0f,103.0f,121.0f,120.0f,101.0f;
			Qij.row(7) << 72.0f,92.0f,95.0f,98.0f,112.0f,100.0f,103.0f,99.0f;
			return p * Qij;
		}
		
		Eigen::MatrixXf baseYuvQuant() {
			Eigen::MatrixXf Qij = Eigen::MatrixXf(8,8);
			Qij.row(0) << 17.0f,18.0f,24.0f,47.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(1) << 18.0f,21.0f,26.0f,66.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(2) << 24.0f,26.0f,56.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(3) << 47.0f,66.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(4) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(5) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(6) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
			Qij.row(7) << 99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f,99.0f;
			return Qij;
		}

		JpgEncoder::JpgEncoder() {
			dpcmLengthTable = Eigen::VectorXi(13);
			dpcmHexTable = std::vector<char>(26, 0x00);
			rleLengthTable = Eigen::MatrixXi::Zero(11,8);
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
			rleLengthTable << 4,2,2,3,4,5,7,8, 0,4,5,7,8,0,0,0, 0,5,8,0,0,0,0,0, 0,6,9,0,0,0,0,0, 0,6,0,0,0,0,0,0, 0,7,0,0,0,0,0,0, 0,8,0,0,0,0,0,0, 0,9,0,0,0,0,0,0, 0,9,0,0,0,0,0,0, 0,9,0,0,0,0,0,0;
			
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

		unsigned int JpgEncoder::iitLen(int value) {
			return (unsigned int)std::floor(log2(std::abs(value))) + 1;
		}
		
		void JpgEncoder::dpcmEnc(int value, unsigned int &nBits, std::vector<char> &bits) {
			nBits = dpcmLengthTable[value];
			bits = std::vector<char>(2);
			bits[0] = dpcmHexTable[2 * value];
			bits[1] = dpcmHexTable[2 * value + 1];
		}
		
		void JpgEncoder::iitEnc(int value, unsigned int &nBits, std::vector<char> &bits) {
			nBits = iitLen(value);
			bits = std::vector<char>(2);
			bits[0] = 0xff00 & std::abs(value) >> 8;
			bits[1] = 0x00ff & std::abs(value);
			if (value < 0) {
				bits[0] = ~bits[0];
				bits[1] = ~bits[1];
			}
		}
		
		void JpgEncoder::rleEnc(unsigned int nZeros, unsigned int nLength, unsigned int &nBits, std::vector<char> &bits) {
			nBits = rleLengthTable(nZeros,nLength);
			bits = std::vector<char>(2);
			bits[0] = rleHexTable[2 * (8 * nZeros + nLength)];
			bits[1] = rleHexTable[2 * (8 * nZeros + nLength) + 1];
		}
		
/*		std::vector<char> encode(Eigen::MatrixXi block) {
			// Assumes block has already been transformed and quantized
		}
		
		Eigen::MatrixXi decode(std::vector<char> bits) {
		}*/

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
			Eigen::MatrixXf Qij = linearQuant(8, 1.0f);
			Eigen::MatrixXi Yij = applyQuant(Xij, Qij);
			std::cout << "Quantization:" << std::endl << Qij << std::endl << std::endl;
			std::cout << "Compressed:" << std::endl << Yij << std::endl << std::endl;
			std::cout << "Decompressed:" << std::endl << reverseQuant(Yij, Qij) << std::endl << std::endl;
			
			return true;
		}
	}
}
