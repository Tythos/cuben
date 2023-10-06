/**
 * leastsq.cpp
 */

#include "cuben.hpp"

Eigen::VectorXf cuben::leastsq::invertNormal(Eigen::MatrixXf A, Eigen::VectorXf y) {
    if (A.rows() != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    return (A.transpose() * A).inverse() * A.transpose() * y;
}

Eigen::VectorXf cuben::leastsq::invertSvd(Eigen::MatrixXf A, Eigen::VectorXf y) {
    if (A.rows() != y.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf pinvA = svd.matrixV() * (svd.singularValues().array().inverse().matrix().asDiagonal()) * svd.matrixU().transpose();
    return pinvA * y;
}

Eigen::VectorXf cuben::leastsq::fitPolynomial(Eigen::VectorXf xi, Eigen::VectorXf yi, int degree) {
    int n = xi.rows();
    Eigen::MatrixXf A(n, degree + 1);
    if (n != yi.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    for (int i = 0; i < n; i++) {
        A(i,0) = 1;
        for (int j = 1; j <= degree; j++) {
            A(i,j) = xi(i) * A(i,j-1);
        }
    }
    return cuben::leastsq::invertSvd(A, yi);
}

Eigen::VectorXf cuben::leastsq::fitPeriodic(Eigen::VectorXf xi, Eigen::VectorXf yi, int degree) {
    if (xi.size() != yi.size()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    int m = xi.size();
    int n = 2 * degree + 1;
    Eigen::MatrixXf X(m, n);
    for (int i = 0; i < m; ++i) {
        X(i, 0) = 1.0;
        for (int j = 1; j <= degree; ++j) {
            X(i, 2 * j - 1) = std::cos(2 * M_PI * j * xi(i));
            X(i, 2 * j) = std::sin(2 * M_PI * j * xi(i));
        }
    }
    Eigen::VectorXf p = (X.transpose() * X).ldlt().solve(X.transpose() * yi);
    return p;
}

Eigen::VectorXf cuben::leastsq::fitExponential(Eigen::VectorXf xi, Eigen::VectorXf yi) {
    int n = xi.rows();
    Eigen::MatrixXf A(n, 2);
    Eigen::VectorXf b(n);
    Eigen::VectorXf c(2);
    if (n != yi.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    for (int i = 0; i < n; i++) {
        A(i,0) = 1;
        A(i,1) = xi(i);
        b(i) = std::log(yi(i));
    }
    c = cuben::leastsq::invertSvd(A, b);
    c(0) = std::exp(c(0));
    return c;
}

Eigen::VectorXf cuben::leastsq::fitPower(Eigen::VectorXf xi, Eigen::VectorXf yi) {
    int n = xi.rows();
    Eigen::MatrixXf A(n, 2);
    Eigen::VectorXf b(n);
    Eigen::VectorXf c(2);
    if (n != yi.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    for (int i = 0; i < n; i++) {
        A(i,0) = 1;
        A(i,1) = std::log(xi(i));
        b(i) = std::log(yi(i));
    }
    c = cuben::leastsq::invertSvd(A, b);
    c(0) = std::exp(c(0));
    return c;
}

Eigen::VectorXf cuben::leastsq::fitGamma(Eigen::VectorXf xi, Eigen::VectorXf yi) {
    int n = xi.rows();
    Eigen::MatrixXf A(n, 2);
    Eigen::VectorXf b(n);
    Eigen::VectorXf c(2);
    if (n != yi.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    for (int i = 0; i < n; i++) {
        A(i,0) = 1;
        A(i,1) = std::log(xi(i));
        b(i) = std::log(yi(i));
    }
    c = cuben::leastsq::invertSvd(A, b);
    c(0) = std::exp(c(0));
    return c;
}

Eigen::VectorXf cuben::leastsq::fitCompoundExpo(Eigen::VectorXf xi, Eigen::VectorXf yi) {
    int n = xi.rows();
    Eigen::MatrixXf A(n, 3);
    Eigen::VectorXf b(n);
    Eigen::VectorXf c(3);
    if (n != yi.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    for (int i = 0; i < n; i++) {
        A(i,0) = 1;
        A(i,1) = std::log(xi(i));
        A(i,2) = xi(i);
        b(i) = std::log(yi(i));
    }
    c = cuben::leastsq::invertNormal(A, b);
    c(0) = std::exp(c(0));
    return c;
}

Eigen::MatrixXf cuben::leastsq::computeGramSchmidt(Eigen::MatrixXf A) {
    Eigen::MatrixXf Q(A.rows(), A.cols());
    Q.col(0) = A.col(0) / A.col(0).norm();
    for (int i = 1; i < A.cols(); i++) {
        Q.col(i) = Q.col(0) * (Q.col(0).transpose() * A.col(i));
        for (int j = 1; j < i; j++) {
            Q.col(i) += Q.col(j) * (Q.col(j).transpose() * A.col(i));
        }
        Q.col(i) = A.col(i) - Q.col(i);
        Q.col(i) = Q.col(i) / Q.col(i).norm();
    }
    return Q;
}

void cuben::leastsq::qrFactor(Eigen::MatrixXf A, Eigen::MatrixXf &Q, Eigen::MatrixXf &R) {
    Q = Eigen::MatrixXf(A.rows(), A.cols());
    R = Eigen::MatrixXf::Zero(A.cols(), A.cols());
    R(0,0) = A.col(0).norm();
    Q.col(0) = A.col(0) / R(0,0);
    for (int i = 1; i < A.cols(); i++) {
        Q.col(i) = Q.col(0) * (Q.col(0).transpose() * A.col(i));
        R(0,i) = Q.col(0).transpose() * A.col(i);
        for (int j = 1; j < i; j++) {
            R(j,i) = Q.col(j).transpose() * A.col(i);
            Q.col(i) += Q.col(j) * R(j,i);
        }
        Q.col(i) = A.col(i) - Q.col(i);
        R(i,i) = Q.col(i).norm();
        Q.col(i) = Q.col(i) / R(i,i);
    }
}

Eigen::VectorXf cuben::leastsq::qrLeastSq(Eigen::MatrixXf A, Eigen::VectorXf b) {
    Eigen::MatrixXf Q(A.rows(), A.cols());
    Eigen::MatrixXf R(A.cols(), A.cols());
    cuben::leastsq::qrFactor(A, Q, R);
    return R.inverse() * (Q.transpose() * b);
}

Eigen::MatrixXf cuben::leastsq::householderReflector(Eigen::VectorXf a, Eigen::VectorXf b) {
    int n = a.rows();
    if (n != b.rows()) {
        throw cuben::exceptions::xMismatchedDims();
    }
    Eigen::VectorXf v = a - b;
    Eigen::MatrixXf P = v * v.transpose() / (v.transpose() * v);
    return Eigen::MatrixXf::Identity(n,n) - 2 * P;
}

void cuben::leastsq::hhQrFactor(Eigen::MatrixXf A, Eigen::MatrixXf &Q, Eigen::MatrixXf &R) {
    int nIterations = std::min(A.rows() - 1, A.cols());
    Eigen::MatrixXf H(0,0);
    Eigen::VectorXf x;
    Eigen::VectorXf w;
    Eigen::MatrixXf subMat(0,0);
    Eigen::MatrixXf Hfull(0,0);

    Q = Eigen::MatrixXf::Identity(A.rows(), A.rows());
    R = A;
    for (int i = 0; i < nIterations; i++) {
        // Compute householder reflector for reduction of this column of A
        subMat = R.block(i, i, R.rows() - i, R.cols() - i);
        x = subMat.col(0);
        w = Eigen::VectorXf::Zero(x.rows());
        //w(0) = x(0) < 0 ? x.norm() : -x.norm();
        w(0) = x.norm();
        H = cuben::leastsq::householderReflector(x, w);
        
        // Update Q and R computations
        Hfull = Eigen::MatrixXf::Identity(A.rows(), A.rows());
        Hfull.block(i, i, A.rows() - i, A.rows() - i) = H;
        Q = Q * Hfull;
        R = Hfull * R;
        
        // Correct R for drift
        for (int j = 0; j <= i; j++) {
            for (int k = j + 1; k < R.rows(); k++) {
                R(k,j) = 0.0f;
            }
        }
        
        // Debug report
        std::cout << "i = " << i << std::endl;
        std::cout << "subMat:" << std::endl << subMat << std::endl;
        std::cout << "H:" << std::endl << H << std::endl;
        std::cout << "Q:" << std::endl << Q << std::endl;
        std::cout << "R:" << std::endl << R << std::endl << std::endl;
    }
}

Eigen::VectorXf cuben::leastsq::nonLinearGaussNewton(Eigen::VectorXf(*f)(Eigen::VectorXf), Eigen::MatrixXf(*dfdx)(Eigen::VectorXf), Eigen::VectorXf x0, int nIter) {
    Eigen::VectorXf f0 = f(x0);
    Eigen::MatrixXf dfdx0 = dfdx(x0);
    Eigen::MatrixXf A;
    Eigen::VectorXf b;
    Eigen::VectorXf dx(x0.rows()); dx(0) = cuben::constants::iterTol + 1.0f;
    int nf = f0.rows();
    int nx = x0.rows();
    if (nf != dfdx0.rows() || dfdx0.cols() != nx) {
        throw cuben::exceptions::xMismatchedDims();
    }
    
    int n = 0;
    while (dx.sum() > cuben::constants::iterTol && n < cuben::constants::iterLimit) {
//			for (int i = 0; i < nIter; i++) {
        n++;
        A = dfdx0.transpose() * dfdx0;
        b = -dfdx0.transpose() * f0;
        dx = A.inverse() * b;
        x0 = x0 + dx;
        f0 = f(x0);
        dfdx0 = dfdx(x0);
    }
    return x0;
}
