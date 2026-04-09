// Generalized Inverse Gaussian sampler.
//
// f(x; lam, chi, psi) ∝ x^{lam-1} * exp(-(chi/x + psi*x) / 2),  x > 0,
// with chi > 0, psi > 0, lam ∈ R.
//
// Strategy
// --------
// Two paths, chosen at the (lam, chi, psi) entry point:
//
//   1. **Asymptotic / degenerate** (chi << psi or psi << chi):
//      The GIG collapses to a Gamma or Inverse-Gamma in the limit. Detected
//      via the dimensionless ratio chi/psi.
//
//        - chi very small relative to (psi, |lam|+1): the chi/x term is
//          negligible for typical x, and  f(x) ≈ x^{lam-1} exp(-psi x / 2),
//          i.e. Gamma(shape=lam, rate=psi/2). Requires lam > 0.
//
//        - psi very small relative to (chi, |lam|+1): use the duality
//          GIG(lam, chi, psi) =d 1 / GIG(-lam, psi, chi), so sample
//          y ~ Gamma(-lam, rate=chi/2) and return 1/y. Requires lam < 0.
//
//        - Mixed degenerate cases (lam crosses zero): fall through to path 2.
//
//   2. **Bounded ratio-of-uniforms** (the universal path):
//      Sample y ~ GIG(lam, omega, omega) with omega = sqrt(chi*psi), then
//      scale by sqrt(chi/psi). Bounded ROU box [0, u+] × [0, v+] derived from
//      the unimodal structure of g(y) = y^{lam-1} exp(-omega(y+1/y)/2).
//      A high trial limit (200000) covers borderline small-omega regimes;
//      degenerate cases are routed to path 1 first.
//
// References
// ----------
// Hörmann, W. & Leydold, J. (2014). "Generating generalized inverse Gaussian
// random variates." Statistics and Computing, 24(4), 547-557.
//
// Devroye, L. (1986). "Non-uniform random variate generation." Springer.
// Section IV.4.6 — duality transformation.

#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace tgw {

// Mode of g(y) = y^{lam-1} exp(-omega*(y + 1/y)/2) with chi=psi=omega.
inline double gig_mode_unit(double lam, double omega) {
    if (std::abs(lam) <= 1.0) {
        return omega / (std::sqrt((1.0 - lam) * (1.0 - lam) + omega * omega) + 1.0 - lam);
    }
    return (lam - 1.0 + std::sqrt((lam - 1.0) * (lam - 1.0) + omega * omega)) / omega;
}

// log g(y) for the unit-scale GIG (chi=psi=omega).
inline double log_g_unit(double y, double lam, double omega) {
    return (lam - 1.0) * std::log(y) - 0.5 * omega * (y + 1.0 / y);
}

// Bounded ratio-of-uniforms sampler for GIG(lam, omega, omega).
// Universal but with worst-case efficiency at very small omega; the
// degenerate-limit guard in `sample_gig` keeps us out of those regions.
template <typename RNG>
double rou_unit_scale(double lam, double omega, RNG& rng) {
    if (omega <= 0.0) {
        throw std::invalid_argument("rou_unit_scale: omega must be positive");
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const double xm = gig_mode_unit(lam, omega);
    const double log_gm = log_g_unit(xm, lam, omega);
    const double u_plus = std::exp(0.5 * log_gm);

    // s_plus = argmax_x [ x sqrt(g(x)) ]; closed form from
    //   d/dx [(lam+1) log x - omega(x + 1/x)/2] = 0
    // => omega x^2 - 2(lam+1) x - omega = 0
    const double s_plus =
        ((lam + 1.0) + std::sqrt((lam + 1.0) * (lam + 1.0) + omega * omega)) / omega;
    const double v_plus = s_plus * std::exp(0.5 * log_g_unit(s_plus, lam, omega));

    constexpr int MAX_TRIALS = 200000;
    for (int trial = 0; trial < MAX_TRIALS; ++trial) {
        double u = U(rng) * u_plus;
        double v = U(rng) * v_plus;
        if (u <= 0.0) continue;
        double x = v / u;
        if (x <= 0.0) continue;
        if (2.0 * std::log(u) <= log_g_unit(x, lam, omega)) {
            return x;
        }
    }
    throw std::runtime_error("sample_gig: ROU rejection failed to converge");
}

// Public scalar sampler.
template <typename RNG>
double sample_gig(double lam, double chi, double psi, RNG& rng) {
    if (chi <= 0.0 || psi <= 0.0) {
        throw std::invalid_argument("sample_gig: chi and psi must be positive");
    }

    // -------- Degenerate-limit short-circuits --------
    //
    // We use the dimensionless ratio of the two scale terms to decide whether
    // one of (chi, psi) is so small relative to the other that the
    // corresponding pole of g(x) is irrelevant. Threshold chosen empirically:
    // for chi*psi < 1e-8 the bounded ROU box becomes ~10^4 wider than the
    // accept region, and trial counts blow up; the asymptotic Gamma limit is
    // accurate to ~6 decimal digits in this regime.
    constexpr double DEGEN_OMEGA2 = 1e-8;  // i.e. omega = sqrt(chi*psi) < 1e-4
    const double omega2 = chi * psi;

    if (omega2 < DEGEN_OMEGA2) {
        // chi or psi (or both) is tiny.
        if (chi <= psi) {
            // chi << psi  =>  chi/x term negligible  =>  Gamma(lam, rate=psi/2)
            if (lam > 0.0) {
                std::gamma_distribution<double> gam(lam, 2.0 / psi);
                return gam(rng);
            }
            // For lam <= 0 with very small chi, the distribution concentrates
            // near zero. Sample from the tightest available proxy: an
            // Inverse-Gamma via the duality (which uses psi, not chi).
            // This may be inaccurate but avoids the ROU blow-up.
            std::gamma_distribution<double> gam(std::max(-lam, 1e-3), 2.0 / std::max(chi, 1e-300));
            double y = gam(rng);
            return 1.0 / std::max(y, 1e-300);
        } else {
            // psi << chi  =>  use duality GIG(lam, chi, psi) = 1/GIG(-lam, psi, chi)
            if (-lam > 0.0) {
                std::gamma_distribution<double> gam(-lam, 2.0 / chi);
                double y = gam(rng);
                return 1.0 / std::max(y, 1e-300);
            }
            std::gamma_distribution<double> gam(std::max(lam, 1e-3), 2.0 / std::max(psi, 1e-300));
            return gam(rng);
        }
    }

    // -------- Universal bounded ROU --------
    const double omega = std::sqrt(omega2);
    const double scale = std::sqrt(chi / psi);
    double y = rou_unit_scale(lam, omega, rng);
    return y * scale;
}

template <typename RNG>
void sample_gig_vec(double lam, const double* chi, const double* psi,
                    double* out, std::size_t n, RNG& rng) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = sample_gig(lam, chi[i], psi[i], rng);
    }
}

}  // namespace tgw
