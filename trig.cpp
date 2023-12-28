/**
 * trig.cpp
 */

#include "cuben.hpp"

Eigen::VectorXcf cuben::trig::sft(Eigen::VectorXf xi) {
    int n = xi.rows();
    float d = 1.0f / std::sqrt((float)n);
    std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
    Eigen::MatrixXcf F = Eigen::MatrixXcf(n,n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            F(i,j) = std::pow(rou, i * j);
        }
    }
    return d * F * xi;
}

Eigen::VectorXf cuben::trig::isft(Eigen::VectorXcf yi) {
    int n = yi.rows();
    float d = 1.0f / std::sqrt((float)n);
    std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), std::sin(2 * M_PI / (float)n));
    Eigen::MatrixXcf iF = Eigen::MatrixXcf(n,n);
    Eigen::VectorXf xi = Eigen::VectorXf(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            iF(i,j) = std::pow(rou, -(float)(i * j));
        }
    }
    for (int i = 0; i < n; i++) {
        std::cmplx p = iF.row(i).dot(yi);
        xi(i) = (d * p).real();
    }
    return xi;
}

Eigen::VectorXcf cuben::trig::fft(Eigen::VectorXf xi) {
    int n = xi.rows();
    float d = 1.0f / std::sqrt((float)n);
    std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
    Eigen::VectorXcf yi = Eigen::VectorXcf(n);
    
    // Course of action depends on input size
    if (n == 2) {
        // Compute unit (2x2) transformation; normalize, as this is not a recursive call.
        yi(0) = d * (xi(0) + xi(1));
        yi(1) = d * (xi(0) + rou * xi(1));
    } else if (n % 2 == 0) {
        // Invoke recursive FFT (fftRec()) on seperate halves. Since this is the initialization
        // (and conclusion) of a recusrsive call, normalize by d.
        Eigen::VectorXf evenEls(n / 2);
        Eigen::VectorXf oddEls(n / 2);
        for (int i = 0; i < n / 2; i++) {
            evenEls(i) = xi(2 * i);
            oddEls(i) = xi(2 * i + 1);
        }
        Eigen::VectorXcf evenTrans = fftRec(evenEls);
        Eigen::VectorXcf oddTrans = fftRec(oddEls);
        for (int i = 0; i < n / 2; i++) {
            yi(i) = d * (evenTrans(i) + std::pow(rou, (float)i) * oddTrans(i));
            yi(i + n / 2) = d * (evenTrans(i) + std::pow(rou, (float)(i + n / 2)) * oddTrans(i));
        }
    } else {
        // Fall back to slow (explicit) FFT method. The result is already normalized.
        yi = sft(xi);
    }
    return yi;
}

Eigen::MatrixXcf cuben::trig::fftRec(Eigen::VectorXf xi) {
    int n = xi.rows();
    std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
    Eigen::VectorXcf yi = Eigen::VectorXcf(n);
    
    // Course of action depends on input size
    if (n == 2) {
        // Compute unit (2x2) transformation; do not normalize, as this is a recursive call.
        yi(0) = xi(0) + xi(1);
        yi(1) = xi(0) + rou * xi(1);
    } else if (n % 2 == 0) {
        // Repeat recursion for smaller subset, then construct and return un-normalized transformation
        Eigen::VectorXf evenEls(n / 2);
        Eigen::VectorXf oddEls(n / 2);
        for (int i = 0; i < n / 2; i++) {
            evenEls(i) = xi(2 * i);
            oddEls(i) = xi(2 * i + 1);
        }
        Eigen::VectorXcf evenTrans = fftRec(evenEls);
        Eigen::VectorXcf oddTrans = fftRec(oddEls);
        for (int i = 0; i < n / 2; i++) {
            yi(i) = evenTrans(i) + std::pow(rou, (float)i) * oddTrans(i);
            yi(i + n / 2) = evenTrans(i) + std::pow(rou, (float)(i + n / 2)) * oddTrans(i);
        }
    } else {
        // Fall back to slow (explicit) FFT method. Since this is normalized, the normalization
        // must be corrected.
        yi = std::sqrt((float)n) * sft(xi);
    }
    return yi;
}

Eigen::VectorXf cuben::trig::ifft(Eigen::VectorXcf yi) {
    int n = yi.rows();
    float d = 1.0f / std::sqrt((float)n);
    std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
    Eigen::VectorXcf xi = Eigen::VectorXcf(n);
    
    // First, compute equivalent (complex) transformation of the conjugate of yi
    Eigen::VectorXcf cyi = Eigen::VectorXcf(n);
    for (int i = 0; i < n; i++) {
        cyi(i) = std::conj(yi(i));
    }
    
    // Specific step in constructing the transformation of the conjugate depends on
    // the size of the input
    if (n == 2) {
        // Compute unit (2x2) transformation; can be normalized, since this is the top-level call
        xi(0) = d * (cyi(0) + cyi(1));
        xi(1) = d * (cyi(0) + rou * cyi(1));
    } else if (n % 2 == 0) {
        // Invoke recursion for smaller subset, normalizing at conclusion
        Eigen::VectorXcf evenEls(n / 2);
        Eigen::VectorXcf oddEls(n / 2);
        for (int i = 0; i < n / 2; i++) {
            evenEls(i) = cyi(2 * i);
            oddEls(i) = cyi(2 * i + 1);
        }
        Eigen::VectorXcf evenTrans = fftRec(evenEls);
        Eigen::VectorXcf oddTrans = fftRec(oddEls);
        for (int i = 0; i < n / 2; i++) {
            xi(i) = d * (evenTrans(i) + std::pow(rou, (float)i) * oddTrans(i));
            xi(i + n / 2) = d * (evenTrans(i) + std::pow(rou, (float)(i + n / 2)) * oddTrans(i));
        }
    } else {
        // Fall back to a complex (normalized) variation of the explicit FFT method
        Eigen::MatrixXcf F = Eigen::MatrixXcf(n,n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                F(i,j) = std::pow(rou, i * j);
            }
        }
        xi = d * F * cyi;
    }
    
    // The actual inverse is the real component of the conjugate of the transformation of the conjugate
    Eigen::VectorXf rxi = Eigen::VectorXf(n);
    for (int i = 0; i < n; i++) {
        // This should be redundant, as real(c) = real(c*)--but truncation can produce small
        // imaginary offsets for time domain components.
        rxi(i) = std::conj(xi(i)).real();
    }
    return rxi;
}

Eigen::MatrixXcf cuben::trig::fftRec(Eigen::VectorXcf xi) {
    // Note: this method is duplicated here with complex arguments to faciliate the inverse transform
    int n = xi.rows();
    std::cmplx rou = std::cmplx(std::cos(2 * M_PI / (float)n), -std::sin(2 * M_PI / (float)n));
    Eigen::VectorXcf yi = Eigen::VectorXcf(n);
    
    // Course of action depends on input size
    if (n == 2) {
        // Compute unit (2x2) transformation; do not normalize, as this is a recursive call.
        yi(0) = xi(0) + xi(1);
        yi(1) = xi(0) + rou * xi(1);
    } else if (n % 2 == 0) {
        // Repeat recursion for smaller subset, then construct and return un-normalized transformation
        Eigen::VectorXcf evenEls(n / 2);
        Eigen::VectorXcf oddEls(n / 2);
        for (int i = 0; i < n / 2; i++) {
            evenEls(i) = xi(2 * i);
            oddEls(i) = xi(2 * i + 1);
        }
        Eigen::VectorXcf evenTrans = fftRec(evenEls);
        Eigen::VectorXcf oddTrans = fftRec(oddEls);
        for (int i = 0; i < n / 2; i++) {
            yi(i) = evenTrans(i) + std::pow(rou, (float)i) * oddTrans(i);
            yi(i + n / 2) = evenTrans(i) + std::pow(rou, (float)(i + n / 2)) * oddTrans(i);
        }
    } else {
        // Fall back to slow (explicit) FFT method. Since this is recreated here for the case
        // of complex elements, we just don't include the normalization.
        Eigen::MatrixXcf F = Eigen::MatrixXcf(n,n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                F(i,j) = std::pow(rou, i * j);
            }
        }
        yi = F * xi;
    }
    return yi;
}

Eigen::VectorXcf cuben::trig::dft(Eigen::VectorXf xi) {
    // Transforms the given discrete time-domain signal into the frequency domain
    unsigned int n = xi.rows();
    unsigned int midNdx = n / 2;
    Eigen::VectorXcf XiCirc = fft(xi);
    Eigen::VectorXcf Xi = Eigen::VectorXcf(n);
    
    if (n % 2 == 0) {
        // For signals with an even number of samples, the resulting transform vector
        // is asymmetric, as w = -1 is not duplicated to retain size.
        Xi.segment(0,midNdx) = XiCirc.segment(midNdx,midNdx);
        Xi.segment(midNdx,midNdx) = XiCirc.segment(0,midNdx);
    } else {
        // For signals with an odd number of samples, the resulting transform vector
        // is symmetric, with w = 1 in the middle, but does not include w = -1.
        midNdx = (n - 1) / 2;
        Xi.segment(0,midNdx) = XiCirc.segment(midNdx+1,midNdx);
        Xi.segment(midNdx,midNdx+1) = XiCirc.segment(0,midNdx+1);
    }
    return Xi;
}

Eigen::VectorXf cuben::trig::idft(Eigen::VectorXcf Xi) {
    // Transforms the given discrete time-domain signal into the frequency domain
    unsigned int n = Xi.rows();
    unsigned int midNdx = n / 2;
    Eigen::VectorXcf XiCirc = Eigen::VectorXcf(n);
    
    if (n % 2 == 0) {
        // For signals with an even number of samples, the transform vector
        // is asymmetric, with w = -1 is not duplicated to retain size.
        XiCirc.segment(0,midNdx) = Xi.segment(midNdx,midNdx);
        XiCirc.segment(midNdx,midNdx) = Xi.segment(0,midNdx);
    } else {
        // For signals with an odd number of samples, the resulting transform vector
        // is symmetric, with w = 1 in the middle, but does not include w = -1.
        midNdx = (n - 1) / 2;
        XiCirc.segment(0,midNdx+1) = Xi.segment(midNdx,midNdx+1);
        XiCirc.segment(midNdx+1,midNdx) = Xi.segment(0,midNdx);
    }
    return ifft(XiCirc);
}

Eigen::VectorXf cuben::trig::genFreqVec(unsigned int n, float fSamp_hz) {
    Eigen::VectorXf fi_hz = Eigen::VectorXf(n);
    if (n % 2 == 0) {
        float fMax_hz = 0.5f * fSamp_hz;
        float df_hz = fSamp_hz / (float)n;
        fi_hz.segment(0,n) = cuben::fundamentals::initRangeVec(-fMax_hz, df_hz, fMax_hz - df_hz);
    } else {
        float fMax_hz = 0.5f * fSamp_hz * ((float)n - 1.0f) / ((float)n);
        float df_hz = fSamp_hz / (float)n;
        fi_hz.segment(0,n) = cuben::fundamentals::initRangeVec(-fMax_hz, df_hz, fMax_hz);
    }
    return fi_hz;
}

Eigen::VectorXf cuben::trig::genTimeVec(unsigned int n, float fSamp_hz) {
    float dt_s = 1.0f / fSamp_hz;
    float tf_s = dt_s * ((float)n - 1.0f);
    return cuben::fundamentals::initRangeVec(0.0f, dt_s, tf_s);
}

void cuben::trig::dftInterp(Eigen::VectorXf tmi_s, Eigen::VectorXf xmi, unsigned int n, Eigen::VectorXf &tni_s, Eigen::VectorXf &xni) {
    Eigen::VectorXcf Xmi = dft(xmi);
    unsigned int m = Xmi.rows();
    unsigned int nMargin = (unsigned int)std::ceil(0.5f * ((float)n - (float)m));
    Eigen::VectorXcf Xni = Eigen::VectorXcf(n);
    for (int i = 0; i < nMargin; i++) {
        Xni(i) = std::cmplx(0.0f,0.0f);
        Xni(n-i-1) = std::cmplx(0.0f,0.0f);
    }
    Xni.segment(nMargin,m) = Xmi;
    xni = idft(Xni) * std::sqrt((float)n / (float)m);
    tni_s = genTimeVec(xni.rows(), (float)n / (float)m / (tmi_s(1) - tmi_s(0)));
}

void cuben::trig::dftFit(Eigen::VectorXf tni_s, Eigen::VectorXf xni, unsigned int m, Eigen::VectorXf &tmi_s, Eigen::VectorXf &xmi) {
    Eigen::VectorXcf Xni = dft(xni);
    unsigned int n = Xni.rows();
    unsigned int nMargin = (unsigned int)std::floor(0.5f * ((float)n - (float)m));
    Eigen::VectorXcf Xmi = Xni.segment(nMargin,m);
    std::cout << "Xmi:" << std::endl << Xmi << std::endl << std::endl;
    xmi = idft(Xmi) * std::sqrt((float)m / (float)n);
    tmi_s = genTimeVec(xmi.rows(), (float)m / (float)n / (tni_s(1) - tni_s(0)));
}

Eigen::VectorXf cuben::trig::wienerFilter(Eigen::VectorXf xni, float p) {
    unsigned int n = xni.rows();
    float noise = 0.5f * cuben::fundamentals::stdDev(xni);
    Eigen::VectorXcf Xi = dft(xni);

    std::cout << "std dev: " << 2.0f * noise << std::endl;
    noise = (float)n * (p * noise) * (p * noise);
    for (int i = 0; i < n; i++) {
        float x2 = (std::conj(Xi(i)) * Xi(i)).real();
        if (i < 16) { std::cout << Xi(i); }
        Xi(i) = std::max(x2 - noise, 0.0f) / x2 * Xi(i);
        if (i < 16) { std::cout << Xi(i) << std::endl; }
    }
    return idft(Xi);
}
