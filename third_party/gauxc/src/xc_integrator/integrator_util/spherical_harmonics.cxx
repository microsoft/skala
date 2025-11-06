#include "spherical_harmonics.hpp"
// Computes the normalization constants N(l,m) for spherical harmonics up to degree lmax
// N(l,m) = sqrt((2l + 1) / (4π) * ( (l - m)! / (l + m)! ) )
// for m = 0, N(l,0) = sqrt(4π / (2l + 1))
// for m > 0, N(l,m) = -N(l,m-1) / sqrt((l - m + 1) * (l + m))
std::vector<double> sph_nlm(const int lmax) {
    std::vector<double> nlm((lmax + 1) * (lmax + 1), 0.0);
    for (int l = 0; l <= lmax; ++l) {
        // For m = 0
        int ind = l*l+l;
        double tmp = std::sqrt( 4.0 * M_PI / (2 * l + 1) );
        nlm[ind] = 1 / tmp;
        // For m != 0
        tmp = nlm[ind] * std::sqrt(2.0);
        for (int m = 1; m <= l; ++m) {
            tmp = -tmp / std::sqrt(static_cast<double>((l - m + 1) * (l + m)));
            nlm[ind + m ] = tmp;
        }
    }
    return nlm;
}

// Computes associated Legendre polynomials P_l^m(cos(theta)) up to degree lmax
// // Input:
// // - cos_theta: cos(theta), where -1 <= cos_theta <= 1
// // - sin_theta: sin(theta), where 0 <= sin_theta <= 1
// // - lmax: maximum degree of the polynomials to compute, lmax >= 0
// // Output:
// // - Returns a vector with values of associated Legendre polynomials, flattened to 1D with size (lmax+1)*(lmax+1)
std::vector<double> sph_plm (const double cos_theta, const double sin_theta, const int lmax) {
    std::vector<double> plms((lmax + 1) * (lmax + 1), 0.0);
    
    // Base cases
    plms[0] = 1.0;  // P_0^0 = 1
    if (lmax == 0) return plms;

    plms[2] = cos_theta;   // P_1^0 = cos(theta)
    plms[3] = -sin_theta;  // P_1^1 = -sin(theta)
    if (lmax == 1) return plms;

    double cos_theta2 = cos_theta * cos_theta;
    plms[6] = 1.5 * cos_theta2 - 0.5; // P_2^0 (cos(theta)) = 1.5 * cos^2(theta) - 0.5, idx = 2*2 + 2 + 0 = 6
    plms[7] = -3 * sin_theta * cos_theta; // P_2^1 (cos(theta)) = -3 * sin(theta) * cos(theta)
    plms[8] = 3 * sin_theta * sin_theta; // P_2^2 (cos(theta)) = -3 * sin^2(theta)
    if (lmax == 2) return plms;
    
    plms[12] = 2.5 * cos_theta2 * cos_theta - 1.5 * cos_theta; // P_3^0 (cos(theta)) = 2.5 * cos^3(theta) - 1.5 * cos(theta)
    plms[13] = -7.5 * cos_theta2 * sin_theta + 1.5 * sin_theta ; // P_3^1 (cos(theta)) = -7.5 * cos^2(theta) * sin(theta) + 1.5 * sin(theta)
    plms[14] = -5.0 * sin_theta * plms[7]; // P_3^2 (cos(theta)) = -5.0 * sin(theta) * P_2^1 (cos(theta))
    plms[15] = -5.0 * sin_theta * plms[8]; // P_3^3 (cos(theta)) = -5.0 * sin(theta) * P_2^2 (cos(theta))
    if (lmax == 3) return plms;
    // Recurrence calculation for larger p
    for (int l = 4; l <= lmax; ++l) {
        double work = (2.0 * l - 1) * cos_theta;
        for (int m = 0; m < l; ++m) {
            int ind = l * l + l + m;
            int pl1m_ind = (l - 1) * (l - 1) + l - 1 + m;
            int pl2m_ind = (l - 2) * (l - 2) + l - 2 + m;
            plms[ind] = (work * plms[pl1m_ind] - (l + m - 1) * plms[pl2m_ind]) / (l - m);
        }
        // Special case for m = l, P_m^m = -sin_theta * (2*m+1) * P_{m-1}^{m-1}
        plms[(l+1)*(l+1) - 1] = -sin_theta * (2 * (l - 1) + 1) * plms[l*l-1];
    }
    return plms;
}

// Computes spherical harmonics Y_l^m(theta, phi) = N(l,m) P_l^m(cos(theta)) e^(imphi)
// up to degree lmax at point x, with scaling factors nlm
// - Returns a vector with size (lmax+1)*(lmax+1)
void sph_legendre(const int lmax, const std::array<double, 3> x, const std::vector<double>& nlm, double* ylms) {
    assert(x.size() == 3);
    double rho = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    if (rho == 0.0) {
        return;
    }
    double sin_theta = sqrt(x[0] * x[0] + x[1] * x[1]) / rho; // sin(theta) = r_xy/rho
    if (sin_theta != 0.0) {
        double cos_theta = x[2] / rho;
        std::vector<double> plm = sph_plm(cos_theta, sin_theta, lmax);
        for (int l = 0; l <= lmax; l++) {
            int ind = l * l + l;
            ylms[ind] = plm[ind] * nlm[ind]; // m = 0 implicitly uses `vcos(1) = 1`
            for (int m = 1; m <= l; ++m) {
                ylms[ind + m] = plm[ind + m] * nlm[ind + m];
                ylms[ind - m] = ylms[ind + m];
            }
        }
    } else {
        // x = 0, y = 0, z != 0
        double cos_theta = (x[2] > 0.0) ? 1.0 : -1.0;
        for (int l = 0; l <= lmax; l ++) {
            int ind = l * l + l;
            ylms[ind] = nlm[ind];
            if (l % 2 != 0) {
                ylms[ind] *= cos_theta;
            }
        }
    }
}

// compute scaled spherical harmonics, with precomputed normalization factors
//    4π     |x - a|^l
//  ------  ----------- Y_l^m(|x - a|)
//  2l + 1       r^l
void scaled_ylm_new(const int lmax, const std::array<double, 3> x, const std::array<double, 3> a, const double r, const std::vector<double>& nlm, double* ylm) {
    std::array<double, 3> delta = {x[0] - a[0], x[1] - a[1], x[2] - a[2]};
    double dnorm = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    assert(dnorm != 0.0);   
    std::array<double, 3> delta_norm = {delta[0] / dnorm, delta[1] / dnorm, delta[2] / dnorm};
    double phi = atan2(delta_norm[1], delta_norm[0]);
    sph_legendre(lmax, delta_norm, nlm, ylm);
    for (int l = 0; l <= lmax; l++) {
        double ratio = pow(dnorm / r, l) * 4.0 * M_PI / (2 * l + 1);
        for (int m = -l; m <= l; m++) {
            int ind = l * l + l + m;
            if (m == 0) {
              ylm[ind] *= ratio;
            } else if (m < 0) {
              ylm[ind] *= - ratio * sin(m * phi);
            } else {
              ylm[ind] *= ratio * cos(m * phi);
            }
        }
    }
}

// compute scaled spherical harmonics, with standard library functions
// DISABLED ON MACOS: std::sph_legendre not available in Apple's libc++
// This function is unused. Added in commit 66784a4 (Feb 2025), never called.
// Production uses scaled_ylm_new() above.
#ifndef __APPLE__
std::vector<double> scaled_ylm_std(int lmax, std::array<double, 3> x, std::array<double, 3> a, double r) {

    std::vector<double> delta = {x[0] - a[0], x[1] - a[1], x[2] - a[2]};
    double dnorm = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    assert(dnorm != 0.0);
    std::vector<double> delta_norm = {delta[0] / dnorm, delta[1] / dnorm, delta[2] / dnorm};

    double rho = sqrt(delta_norm[0] * delta_norm[0] + delta_norm[1] * delta_norm[1] + delta_norm[2] * delta_norm[2]);
    double theta = acos(delta_norm[2] / rho);
    double phi = atan2(delta_norm[1], delta_norm[0]);

    std::vector<double> ylm((lmax + 1) * (lmax + 1), 0.0);
    for (int l = 0; l <= lmax; l++) {
        double ratio = pow(dnorm / r, l) * 4.0 * M_PI / (2 * l + 1);
        for (int m = 0; m <= l; m++) {
            double sph = std::sph_legendre(l, m, theta) * ratio;
            if (m == 0) {
              ylm[l * l + l] = sph;
            } else {
              if (m % 2 != 0) {
                  sph *= -1;
              }
              sph *= sqrt(2.0);
              ylm[l * l + l - m ] = sph * sin(m * phi);
              ylm[l * l + l + m ] = sph * cos(m * phi);
            }
        }
    }
    return ylm;
}
#endif // __APPLE__

void scaled_ylm_matrix(const int lmax, const double* points, const int32_t  npts, const std::array<double, 3> center, const double radius, double* ylm_matrix) {
  int nharmonics = (lmax + 1) * (lmax + 1);
  auto nlm = sph_nlm(lmax);
  for (int i = 0; i < npts; ++i) {
    const std::array<double, 3> x = {points[3 * i], points[3 * i + 1], points[3 * i + 2]};
    scaled_ylm_new(lmax, x, center, radius, nlm, ylm_matrix + i * nharmonics);
  }
}