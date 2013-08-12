/*	Cuben::Trig static object
	Trigonometric interpolation and Fourier transforms
	Derived from Chapter 10 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "../inc/Cuben.h"

namespace Cuben {
	namespace Trig {
		Eigen::VectorXc sft(Eigen::VectorXf xi) {
			int n = xi.rows();
			float d = 1.0f / std::sqrt((float)n);
			std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
			Eigen::MatrixXc F = Eigen::MatrixXc(n,n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					F(i,j) = std::pow(rou, i * j);
				}
			}
			return d * F * xi;
		}
		
		Eigen::VectorXf isft(Eigen::VectorXc yi) {
			int n = yi.rows();
			float d = 1.0f / std::sqrt((float)n);
			std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), std::sin(2 * M_PI / (float)n));
			Eigen::MatrixXc iF = Eigen::MatrixXc(n,n);
			Eigen::VectorXf xi = Eigen::VectorXf(n);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					iF(i,j) = std::pow(rou, i * j);
				}
			}
			for (int i = 0; i < n; i++) {
				std::cmplx p = iF.row(i).dot(yi);
				xi(i) = (d * p).real();
			}
			return xi;
		}
		
		Eigen::VectorXc fft(Eigen::VectorXf xi) {
			// Fall back to slow transform if dimensions are not a multiple of 2
			// (this will at least allow fft for decomposition of dimensions until
			// they are odd--e.g., for a 12x12, decomposition occurs for two steps
			// until a sft is performed for a single 3x3)
			int n = xi.rows();
			float d = 1.0f / std::sqrt((float)n);
			std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
			Eigen::VectorXc yi = Eigen::VectorXc(n);
			
			if (n == 2) {
				yi(0) = d * (xi(0) + xi(1));
				yi(1) = d * (xi(0) + rou * xi(1));
			} else if (n % 2 == 0) {
				// Recursively compute fft for even and odd components
				Eigen::VectorXf evenEls(n / 2);
				Eigen::VectorXf oddEls(n / 2);
				for (int i = 0; i < n / 2; i++) {
					evenEls(i) = xi(2 * i);
					oddEls(i) = xi(2 * i + 1);
				}
				
				// In constructing larger transform, correct for scale performed in recursive step
				float correction = std::sqrt((float)(n / 2));
				Eigen::VectorXc evenTrans = correction * fft(evenEls);
				Eigen::VectorXc oddTrans = correction * fft(oddEls);
				for (int i = 0; i < n / 2; i++) {
					yi(i) = d * (evenTrans(i) + std::pow(rou, (float)i) * oddTrans(i));
					yi(i + n / 2) = d * (evenTrans(i) + std::pow(rou, (float)(i + n / 2)) * oddTrans(i));
				}
			} else {
				yi = sft(xi);
			}
			return yi;
		}
		
		Eigen::VectorXc ifft(Eigen::VectorXf xi) {
		}

		bool test() {
			Eigen::VectorXf xEven = Eigen::VectorXf(4); xEven << 2.0f,5.0f,8.0f,2.0f;
			Eigen::VectorXf xOdd = Eigen::VectorXf(4); xOdd << 3.0f,7.0f,4.0f,1.0f;
			Eigen::VectorXf x = Eigen::VectorXf(8); x << 2.0f,3.0f,5.0f,7.0f,8.0f,4.0f,2.0f,1.0f;
			std::cout << fft(xEven) << std::endl << std::endl;
			std::cout << fft(xOdd) << std::endl << std::endl;
			Eigen::VectorXc c = fft(x);
			std::cout << c << std::endl << std::endl;
			std::cout << isft(c) << std::endl;
			return true;
		}
	}
}
