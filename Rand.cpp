/*	Cuben::Rand static object
	Pseudo-random number generates and stochastic algorithms
	Derived from Chapter 9 of Timothy Sauer's 'Numerical Amalysis'
*/

#include "cuben.h"

namespace Cuben {
	namespace Rand {
		Prng::Prng() {
			state = 1;
			nRolls = 0;
		}
		
		Prng::Prng(unsigned int s) {
			state = s;
		}
		
		float Prng::roll() {
			std::srand(state);
			nRolls++;
			return (float)std::rand() / RAND_MAX;
		}
		
		unsigned int Prng::getState() {
			return state;
		}

		Lcg::Lcg() {
			state = 1;
			multiplier = 13;
			offset = 0;
			modulus = 31;
		}
		
		Lcg::Lcg(unsigned int s, unsigned int m, unsigned int o, unsigned int mod) {
			state = s;
			multiplier = m;
			offset = o;
			modulus = mod;
		}
		
		float Lcg::roll() {
			nRolls++;
			state = (multiplier * state + offset) % modulus;
			return std::abs((float)state / modulus);
		}
		
		MinStd::MinStd() {
			state = 1;
			multiplier = (unsigned int)std::pow(7, 5);
			offset = 0;
			modulus = (unsigned int)std::pow(2, 31) - 1;
		}
		
		MinStd::MinStd(unsigned int s) {
			state = s;
			multiplier = (unsigned int)std::pow(7, 5);
			offset = 0;
			modulus = (unsigned int)std::pow(2, 31) - 1;
		}
		
		Randu::Randu() {
			multiplier = 65539;
			offset = 0;
			modulus = std::pow(2, 31);
		}
		
		Randu::Randu(unsigned int s) {
			state = s;
			multiplier = 65539;
			offset = 0;
			modulus = std::pow(2, 31);
		}
		
		Alfg::Alfg() {
			j = 418;
			k = 1279;
			stateVector = initialize(j, k);
		}
		
		Alfg::Alfg(unsigned int nj, unsigned int nk) {
			j = nj;
			k = nk;
			stateVector = initialize(j, k);
		}
		
		Eigen::VectorXf Alfg::initialize(unsigned int j, unsigned int k) {
			Eigen::VectorXi piDigits(100); piDigits << 1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9;
			Eigen::VectorXi eDigits(100); eDigits << 7,1,8,2,8,1,8,2,8,4,5,9,0,4,5,2,3,5,3,6,0,2,8,7,4,7,1,3,5,2,6,6,2,4,9,7,7,5,7,2,4,7,0,9,3,6,9,9,9,5,9,5,7,4,9,6,6,9,6,7,6,2,7,7,2,4,0,7,6,6,3,0,3,5,3,5,4,7,5,9,4,5,7,1,3,8,2,1,7,8,5,2,5,1,6,6,4,2,7,4;
			Eigen::VectorXf secretSauce(100);
			Eigen::VectorXf si(k);
			for (int i = 0; i < 100; i++) {
				secretSauce(i) = (float)(10 * piDigits(i) + eDigits(i)) / 99.0f;
			}
			if (k <= 100) {
				si = secretSauce.segment(0,k);
			} else {
				int jReduced = (int)((float)j / 2.0f);
				int kReduced = (int)((float)k / 2.0f);
				si.segment(0,kReduced) = initialize(jReduced, kReduced);
				for (int i = kReduced; i < k; i++) {
					si(i) = si(i - jReduced) + si(i - kReduced);
					while (si(i) > 1.0f) { si(i) = si(i) - 1.0f; }
				}
			}
			return si;
		}
		
		float Alfg::roll() {
			nRolls++;
			unsigned int svNdx = nRolls % k;
			unsigned int jNdx = (nRolls - j) % k;
			stateVector(svNdx) = stateVector(jNdx) + stateVector(svNdx);
			while (stateVector(svNdx) > 1.0f) { stateVector(svNdx) = stateVector(svNdx) - 1.0f; }
			return stateVector(svNdx);
		}
		
		Mlfg::Mlfg() {
			j = 418;
			k = 1279;
			stateVector = initialize(j, k);
		}
		
		Mlfg::Mlfg(unsigned int nj, unsigned int nk) {
			j = nj;
			k = nk;
			stateVector = initialize(j, k);
		}
		
		Eigen::VectorXf Mlfg::initialize(unsigned int j, unsigned int k) {
			Eigen::VectorXi piDigits(100); piDigits << 1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9;
			Eigen::VectorXi eDigits(100); eDigits << 7,1,8,2,8,1,8,2,8,4,5,9,0,4,5,2,3,5,3,6,0,2,8,7,4,7,1,3,5,2,6,6,2,4,9,7,7,5,7,2,4,7,0,9,3,6,9,9,9,5,9,5,7,4,9,6,6,9,6,7,6,2,7,7,2,4,0,7,6,6,3,0,3,5,3,5,4,7,5,9,4,5,7,1,3,8,2,1,7,8,5,2,5,1,6,6,4,2,7,4;
			Eigen::VectorXf secretSauce(100);
			Eigen::VectorXf si(k);
			for (int i = 0; i < 100; i++) {
				secretSauce(i) = (float)(10 * piDigits(i) + eDigits(i)) / 99.0f;
			}
			if (k <= 100) {
				si = secretSauce.segment(0,k);
			} else {
				int jReduced = (int)((float)j / 2.0f);
				int kReduced = (int)((float)k / 2.0f);
				si.segment(0,kReduced) = initialize(jReduced, kReduced);
				for (int i = kReduced; i < k; i++) {
					si(i) = 1024.0f * si(i - jReduced) * si(i - kReduced);
					while (si(i) > 1.0f) { si(i) = si(i) - 1.0f; }
				}
			}
			return si;
		}
		
		float Mlfg::roll() {
			nRolls++;
			unsigned int svNdx = nRolls % k;
			unsigned int jNdx = (nRolls - j) % k;
			stateVector(svNdx) = 1024.0f * stateVector(jNdx) * stateVector(svNdx);
			while (stateVector(svNdx) > 1.0f) { stateVector(svNdx) = stateVector(svNdx) - 1.0f; }
			return stateVector(svNdx);
		}

		void MTwist::generate() {
			for (int i = 0; i < n; i++) {
				unsigned int y = (stateVec[i] & pow) + (stateVec[(i + 1) % n] & (pow - 1));
				stateVec[i] = stateVec[(i + m) % n] ^ (y >> 1);
				if (y % 2 != 0) {
					stateVec[i] = stateVec[i] ^ a;
				}
			}
		}
		
		MTwist::MTwist(unsigned int seed) {
			w = 32;
			n = 624;
			m = 397;
			r = 31;
			a = 0x9908B0DF;
			u = 11;
			s = 7;
			b = 0x9D2C5680;
			t = 15;
			c = 0xEFC60000;
			l = 18;
			mask = (unsigned int)std::pow(2, w) - 1;
			pow = (unsigned int)std::pow(2, r);
			spread = 1812433253;
			stateVec = std::vector<unsigned int>(n);
			ndx = 0;
			
			// Loop through state vector and initialize from seed
			stateVec[0] = seed;
			for (int i = 1; i < n; i++) {
				stateVec[i] = (spread * (stateVec[i-1] ^ (stateVec[i-1] >> 30)) + i) & mask;
			}
		}
		
		float MTwist::roll() {
			if (ndx == 0) {
				generate();
			}
			unsigned int y = stateVec[ndx];
			y ^= y >> u;
			y ^= y << s & b;
			y ^= y << t & c;
			y ^= y >> l;
			ndx = (ndx + 1) % n;
			return (float)y / (float)mask;
		}

		Bbs::Bbs() {
			p = 167;
			q = 251;
			xPrev = 15204;
		}
		
		Bbs::Bbs(unsigned int np, unsigned int nq, unsigned int nx) {
			p = np;
			q = nq;
			xPrev = nx;
		}
		
		float Bbs::roll() {
			xPrev = (xPrev * xPrev) % (p * q);
			return (float)xPrev / (float)(p * q - 1);
		}

		float stdRoll() {
			static unsigned int seed = (unsigned int)time(NULL);
			static MTwist mt = MTwist(seed);
			return mt.roll();
		}
		
		Norm::Norm() {
			mean = 0.0f;
			variance = 1.0f;
		}
		
		float Norm::roll() {
			float u1 = stdRoll();
			float u2 = stdRoll();
			return mean + std::sqrt(variance) * std::sqrt(-2.0f * std::log(1.0f - u1)) * std::cos(2 * M_PI * u2);
		}
		
		float Norm::cdf(float x) {
			// Use 1.5e-7 approximation from Abramowitz and Stugen
			x = x / std::sqrt(2);
			bool isNeg = x < 0;
			if (isNeg) { x = -x; }
			float p = 0.3275911f;
			float a1 = 0.254829592f;
			float a2 = -0.284496736f;
			float a3 = 1.421413741f;
			float a4 = -1.453152027f;
			float a5 = 1.061405429f;
			float t = 1 / (1 + p * x);
			float N = 1 - (a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t) * std::exp(-(x * x));
			if (isNeg) { N = -N; }
			return 0.5f * (1.0f + N);
		}
		
		Halton::Halton() {
			basePrime = 2;
		}
		
		Halton::Halton(unsigned int bp) {
			basePrime = bp;
		}
		
		float Halton::roll() {
			throw Cuben::xInvalidRoll();
		}
		
		Eigen::VectorXf Halton::rollAll(unsigned int numRolls) {
			float ratio = std::log((float)numRolls) / std::log((float)basePrime);
			unsigned int n = (unsigned int)std::ceil(ratio + Cuben::Fund::relEps(ratio));
			Eigen::VectorXf b = Eigen::VectorXf::Zero(n);
			Eigen::VectorXf u = Eigen::VectorXf(numRolls);
			for (int j = 0; j < numRolls; j++) {
				int i = 0;
				b(0) = b(0) + 1;
				while (b(i) > (float)basePrime - 1.0f + Cuben::Fund::machEps()) {
					b(i) = 0;
					i++;
					b(i) = b(i) + 1;
				}
				u(j) = 0.0f;
				for (int k = 0; k < b.rows(); k++) {
					std::cout << u(j) << std::endl;
					u(j) = u(j) + b(k) * std::pow((float)basePrime, -(float)(k + 1.0f));
				}
			}
			return u;
		}
		
		RandomWalk::RandomWalk() {
		}
		
		Eigen::VectorXi RandomWalk::getWalk(unsigned int nSteps) {
			Eigen::VectorXi wi = Eigen::VectorXi(nSteps);
			wi(0) = 0;
			for (int i = 1; i < nSteps; i++) {
				float seed = stdRoll();
				while (seed == 0.5f) {
					seed = stdRoll();
				}
				wi(i) = wi(i-1) + 2 * (seed > 0.5f) - 1;
			}
			return wi;
		}
		
		RandomEscape::RandomEscape() {
			lBound = 1;
			uBound = 1;
		}
		
		RandomEscape::RandomEscape(unsigned int lb, unsigned int ub) {
			lBound = lb;
			uBound = ub;
		}
		
		Eigen::VectorXi RandomEscape::getWalk(unsigned int nSteps) {
			nSteps = 1024;
			RandomWalk rw = RandomWalk();
			Eigen::VectorXi wi = rw.getWalk(nSteps);
			
			while (wi.minCoeff() > -(int)lBound && wi.maxCoeff() < (int)uBound) {
				nSteps = 2 * nSteps;
				wi = rw.getWalk(nSteps);
			}
			
			int lNdx = Cuben::Fund::findValue(wi, -(int)lBound);
			int uNdx = Cuben::Fund::findValue(wi, (int)uBound);
			if (lNdx < uNdx && lNdx > -1 || uNdx < 0) {
				wi = wi.segment(0, lNdx + 1);
			} else {
				wi = wi.segment(0, uNdx + 1);
			}
			return wi;
		}
		
		void RandomEscape::setBounds(unsigned lb, unsigned int ub) {
			lBound = lb;
			uBound = ub;
		}
		
		Brownian::Brownian() {
			normPrng = Norm();
		}
		
		Eigen::VectorXf Brownian::sampleWalk(Eigen::VectorXf xi) {
			Eigen::VectorXf yi = Eigen::VectorXf(xi.rows());
			yi(0) = 0.0f;
			for (int i = 1; i < xi.rows(); i++) {
				yi(i) = yi(i-1) + std::sqrt(xi(i) - xi(i-1)) * normPrng.roll();
			}
			return yi;
		}

		BrownianBridge::BrownianBridge() {
			t0 = 1.0f;
			x0 = 1.0f;
			tf = 3.0f;
			xf = 2.0f;
		}
		
		BrownianBridge::BrownianBridge(float nt0, float nx0, float ntf, float nxf) {
			t0 = nt0;
			x0 = nx0;
			tf = ntf;
			xf = nxf;
		}
		
		Eigen::VectorXf BrownianBridge::getWalk(float dt) {
			Eigen::VectorXf ti = Cuben::Fund::initRangeVec(t0, dt, tf);
			Eigen::VectorXf xi = Eigen::VectorXf(ti.rows());
			xi(0) = x0;
			Norm normPrng = Norm();
			for (int i = 1; i < ti.rows(); i++) {
				float f = (xf - xi(i-1)) / (tf - ti(i-1));
				xi(i) = xi(i-1) + f * (ti(i) - ti(i-1)) + normPrng.roll() * std::sqrt(ti(i) - ti(i-1));
			}
			return xi;
		}

		float BrownianBridge::bbf(float t, float x) {
			return (xf - x) / (tf - t);
		}
		
		float BrownianBridge::bbg(float t, float x) {
			return 1.0f;
		}

		Eigen::VectorXf eulerMaruyama(float(*f)(float,float), float(*g)(float,float), Eigen::VectorXf ti, float x0) {
			Eigen::VectorXf xi = Eigen::VectorXf(ti.rows());
			xi(0) = x0;
			Norm normPrng = Norm();
			for (int i = 1; i < ti.rows(); i++) {
				xi(i) = xi(i-1) + f(ti(i-1), xi(i-1)) * (ti(i) - ti(i-1)) + g(ti(i-1), xi(i-1)) * normPrng.roll() * std::sqrt(ti(i) - ti(i-1));
			}
			return xi;
		}
		
		Eigen::VectorXf milstein(float(*f)(float,float), float(*g)(float,float), float(*dgdx)(float,float), Eigen::VectorXf ti, float x0) {
			Eigen::VectorXf xi = Eigen::VectorXf(ti.rows());
			xi(0) = x0;
			Norm normPrng = Norm();
			for (int i = 1; i < ti.rows(); i++) {
				float zi = normPrng.roll();
				xi(i) = xi(i-1) + f(ti(i-1), xi(i-1)) * (ti(i) - ti(i-1)) + g(ti(i-1), xi(i-1)) * zi * std::sqrt(ti(i) - ti(i-1)) + 0.5f * g(ti(i-1), xi(i-1)) * dgdx(ti(i-1), xi(i-1)) * (ti(i) - ti(i-1)) * (zi * zi - 1.0f);
			}
			return xi;
		}
		
		BlackScholes::BlackScholes() {
			initPrice = 12.0f;
			strikePrice = 15.0f;
			interestRate = 0.05;
			volatility = 0.25;
		}
		
		BlackScholes::BlackScholes(float nip, float ncp, float nir, float nv) {
			initPrice = nip;
			strikePrice = ncp;
			interestRate = nir;
			volatility = nv;
		}
		
		Eigen::VectorXf BlackScholes::getWalk(float tf, float dt) {
			int nSteps = (int)std::ceil(tf / dt);
			float sqrtDt = std::sqrt(dt);
			Eigen::VectorXf xi = Eigen::VectorXf(nSteps + 1);
			Norm normPrng = Norm();
			xi(0) = initPrice;
			
			for (int i = 1; i <= nSteps; i++) {
				xi(i) = xi(i-1) + interestRate * xi(i-1) * dt + volatility * xi(i-1) * normPrng.roll() * sqrtDt;
			}
			return xi;
		}
		
		float BlackScholes::computeCallValue(float price, float tf) {
			float d1 = (std::log(price / strikePrice) + (interestRate + 0.5f * volatility * volatility) * tf) / (volatility * std::sqrt(tf));
			float d2 = (std::log(price / strikePrice) + (interestRate - 0.5f * volatility * volatility) * tf) / (volatility * std::sqrt(tf));
			Norm n = Norm();
			return price * n.cdf(d1) - strikePrice * std::exp(-interestRate * tf) * n.cdf(d2);
		}

		float fTest(float t, float x) {
			return 0.1f * x;
		}
		
		float gTest(float t, float x) {
			return 0.3f * x;
		}

		float dgdxTest(float t, float x) {
			return 0.3f;
		}
		
		bool test() {
			BlackScholes bs = BlackScholes();
			Eigen::VectorXf xi = bs.getWalk(0.5f, 0.01f);
			float cv = bs.computeCallValue(12.0f, 0.5f);
			std::cout << "Simulated stock price over time:" << std::endl << xi << std::endl;
			std::cout << "Computes call value: " << cv << std::endl;
			return true;
		}
	}
}
