#ifndef TWO_BODY_CHARGE_HH
#define TWO_BODY_CHARGE_HH

#include "Operator.hh"

#include <armadillo>

#include <array>
#include <complex>
#include <functional>
#include <string>
#include <vector>

namespace imsrg_util
{
namespace chiral_charge
{

/// All momenta and masses passed to this module are in MeV (hbar=c=1).
/// The momentum-space charge-density kernel returned below has units MeV^-3.
using Vec3 = arma::Col<double>::fixed<3>;

/// Basis ordering: spin_1 x spin_2 x isospin_1 x isospin_2, with every
/// two-state factor ordered (+1/2,-1/2).
using SpinIsospinMatrix = arma::cx_mat;

/// Spin-only basis ordering spin_1 x spin_2.  For a state of good total
/// isospin, the isoscalar kernel can be contracted in this smaller space;
/// tau_1.tau_2 is replaced by 2*T*(T+1)-3.
using TwoSpinMatrix = arma::cx_mat::fixed<4, 4>;

/// Ultraviolet prescription applied to the pointwise charge kernel.
///
/// SMS reproduces Filin et al. Eqs. (34)-(41).  Nonlocal multiplies the
/// otherwise unregulated kernel by f(p')f(p), with
/// f(p)=exp[-(p/Lambda)^(2n)].  The latter is a useful GO/EM-style envelope,
/// but is not an SRG evolution of the electromagnetic operator.
enum class RegulatorScheme
{
  SMS,
  Nonlocal,
  Unregulated
};

const char* RegulatorSchemeName(RegulatorScheme scheme);
RegulatorScheme RegulatorSchemeFromString(const std::string& name);

/// Contact couplings A, B, and C in the convention of Filin et al.,
/// Phys. Rev. C 103, 024313 (2021), Eqs. (39)-(41).  Values stored here
/// have units MeV^-5.
struct ContactLECs
{
  double A = 0.0;
  double B = 0.0;
  double C = 0.0;

  /// Convert the dimensionless coefficients M_i quoted in units
  /// F_pi^-2 Lambda_b^-3 to A,B,C in MeV^-5.  The convention is
  /// M1=A+B+C/3, M2=C, M3=A-3B-C after removing the common unit.
  static ContactLECs FromDimensionlessM(
      double M1, double M2, double M3,
      double f_pi_mev = 92.4, double breakdown_scale_mev = 650.0);

  /// Inverse of FromDimensionlessM().
  std::array<double, 3> ToDimensionlessM(
      double f_pi_mev = 92.4, double breakdown_scale_mev = 650.0) const;
};

struct ChargeDensityParameters
{
  double g_a = 1.29;
  double f_pi_mev = 92.4;
  double nucleon_mass_mev = 938.918;
  double pion_mass_mev = 138.03;
  double cutoff_mev = 500.0;
  RegulatorScheme regulator_scheme = RegulatorScheme::SMS;
  int regulator_exponent = 4;
  double beta_8 = 0.25;
  double beta_9 = -0.25;
  double electric_charge = 1.0;
  ContactLECs contact;

  void Validate() const;
};

Vec3 MakeVec3(double x, double y, double z);

/// Breit-frame relative-momentum convention:
///   p1  = -k/4 + p,   p2  = -k/4 - p,
///   p1' =  k/4 + p',  p2' =  k/4 - p'.
/// Thus q_i=p_i'-p_i and q1+q2=k, as in Filin et al. Eq. (25).
Vec3 MomentumTransfer1(const Vec3& p, const Vec3& pprime, const Vec3& k);
Vec3 MomentumTransfer2(const Vec3& p, const Vec3& pprime, const Vec3& k);

/// SMS substitutions in Filin et al. Eqs. (34) and (35).
double RegulatedPionPropagator(double momentum_squared_mev2,
                              const ChargeDensityParameters& parameters);
double RegulatedSquaredPionPropagator(
    double momentum_squared_mev2,
    const ChargeDensityParameters& parameters);

/// External relative-momentum envelope.  This is f(p')f(p) for the
/// Nonlocal scheme and one for SMS or Unregulated.
double RegulatorEnvelope(
    const Vec3& p, const Vec3& pprime,
    const ChargeDensityParameters& parameters);

/// Isoscalar OPE charge density in transfer variables, including the
/// explicitly exchanged (1<->2) contribution.  For Nonlocal this low-level
/// function contains bare pion propagators but cannot apply f(p')f(p),
/// because p and p' are not arguments.  Use TwoBodyChargeDensity() for the
/// complete nonlocal-regulated kernel.
SpinIsospinMatrix OnePionExchange(
    const Vec3& q1, const Vec3& q2, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters);

/// Complete OPE kernel in external relative momenta.  Unlike the transfer-
/// variable overload above, this applies the Nonlocal f(p')f(p) envelope.
SpinIsospinMatrix OnePionExchangeFromMomenta(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters);

/// Regulated isoscalar contact charge density, Filin et al. Eqs. (39)-(41).
SpinIsospinMatrix Contact(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters);

SpinIsospinMatrix TwoBodyChargeDensity(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    bool include_ope = true, bool include_contact = true);

/// Spin-space form of TwoBodyChargeDensity() for a channel of good total
/// isospin T=0 or 1.  This is algebraically identical to projecting the 16x16
/// kernel on the coupled isospin state, but avoids expensive full-matrix
/// allocations in the numerical partial-wave projection.
TwoSpinMatrix TwoBodyChargeSpinMatrix(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    int total_isospin, double ge_isoscalar,
    const ChargeDensityParameters& parameters,
    bool include_ope = true, bool include_contact = true);

/// Relative Frobenius-norm residual for
/// rho(p',p;k)^dagger = rho(p,p';-k).
double HermiticityResidual(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    bool include_ope = true, bool include_contact = true);

// Numerical partial-wave, relative-HO, and pair-CM projection.
/// Relative partial-wave state |p (l S) J M; T MT>.
struct PartialWaveChannel
{
  int l = 0;
  int S = 0;
  int J = 0;
  int M = 0;
  int T = 0;
  int MT = 0;

  void Validate() const;
};

struct ProjectionParameters
{
  int polar_order = 8;
  int azimuthal_order = 16;

  void Validate() const;
};

struct RadialProjectionParameters
{
  int radial_order = 32;
  double max_momentum_mev = 1200.0;

  void Validate() const;
};

struct CenterOfMassProjectionParameters
{
  int radial_order = 48;
  double max_radius_fm = 15.0;

  void Validate() const;
};

/// A pointwise kernel K(p,p',k) in the same momentum and basis convention as
/// TwoBodyChargeDensity().
using KernelFunction = std::function<SpinIsospinMatrix(
    const Vec3&, const Vec3&, const Vec3&)>;

/// Numerically evaluate
/// <p' (l'S')J'M';T'MT' | K(k zhat) | p(lS)JM;TMT>.
/// Spherical harmonics have unit angular norm.  Consequently a constant
/// central kernel projects to 4*pi in the l=l'=0 channel, matching the
/// conventional 2*pi integral over cos(theta) used by PWD.
std::complex<double> ProjectKernel(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    double pprime_mev, double p_mev, double k_mev,
    const ProjectionParameters& quadrature,
    const KernelFunction& kernel);

/// Batched form of ProjectKernel.  Results are row-major in the order
/// pprime_mev[ibra] * p_mev.size() + iket.  Angular basis states are prepared
/// once for the complete radial grid.
std::vector<std::complex<double>> ProjectKernelGrid(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    const std::vector<double>& pprime_mev,
    const std::vector<double>& p_mev, double k_mev,
    const ProjectionParameters& quadrature,
    const KernelFunction& kernel);

/// Optimized good-isospin projection of the chiral two-body charge kernel.
/// This is algebraically identical to ProjectKernelGrid() with the full
/// 16x16 TwoBodyChargeDensity(), but contracts only the 4x4 spin kernel and
/// uses axial covariance about k to remove one redundant azimuthal integral.
std::vector<std::complex<double>> ProjectChargeKernelGrid(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    const std::vector<double>& pprime_mev,
    const std::vector<double>& p_mev, double k_mev,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    const ProjectionParameters& quadrature,
    bool include_ope = true, bool include_contact = true);

std::complex<double> PartialWaveME(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    double pprime_mev, double p_mev, double k_mev,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    const ProjectionParameters& quadrature,
    bool include_ope = true, bool include_contact = true);

/// Normalized relative-coordinate HO radial wave function in momentum space,
/// with integral_0^infinity dp p^2 |R_nl(p)|^2 = 1.  The momentum p is the
/// physical two-nucleon relative momentum (p1-p2)/2, so the oscillator length
/// is b_rel=sqrt(2/(m_N hbar_omega)) in natural units.  The returned value has
/// units MeV^(-3/2).
double RelativeHOMomentumRadial(
    int n, int l, double hbar_omega_mev, double momentum_mev,
    double nucleon_mass_mev);

/// Project a pointwise momentum-space kernel into relative HO states.  No
/// (2*pi)^3 factors are inserted: this follows the partial-wave convention in
/// which a constant central kernel has an s-wave matrix element 4*pi.
std::complex<double> RelativeHOMatrixElement(
    int nbra, const PartialWaveChannel& bra,
    int nket, const PartialWaveChannel& ket,
    double hbar_omega_mev, double k_mev,
    double nucleon_mass_mev,
    const ProjectionParameters& angular_quadrature,
    const RadialProjectionParameters& radial_quadrature,
    const KernelFunction& kernel);

std::complex<double> RelativeHOChargeME(
    int nbra, const PartialWaveChannel& bra,
    int nket, const PartialWaveChannel& ket,
    double hbar_omega_mev, double k_mev, double ge_isoscalar,
    const ChargeDensityParameters& parameters,
    const ProjectionParameters& angular_quadrature,
    const RadialProjectionParameters& radial_quadrature,
    bool include_ope = true, bool include_contact = true);

/// Normalized radial wave function for the physical pair center coordinate
/// R=(r1+r2)/2.  With the repository's equal-mass Moshinsky convention its
/// oscillator length is b_cm=hbar*c/sqrt(2 m_N hbar_omega).
double CenterOfMassHORadial(
    int N, int L, double hbar_omega_mev, double radius_fm,
    double nucleon_mass_mev);

/// Matrix element <N'L'm'|exp(i k R_z)|NLm> for the physical pair center.
/// k is in MeV and R in fm.  This factor supplies the pair-CM dependence that
/// is absent from translationally invariant two-body-potential constructors.
std::complex<double> CenterOfMassPlaneWaveME(
    int Nbra, int Lbra, int Mbra,
    int Nket, int Lket, int Mket,
    double hbar_omega_mev, double k_mev, double nucleon_mass_mev,
    const CenterOfMassProjectionParameters& quadrature);

// Antisymmetrized J-coupled IMSRG operator construction.
struct TwoBodyChargeOperatorParameters
{
  ChargeDensityParameters density;
  ProjectionParameters angular_quadrature;
  RadialProjectionParameters relative_radial_quadrature;
  CenterOfMassProjectionParameters center_of_mass_quadrature;
  bool include_ope = true;
  bool include_contact = true;
  double imaginary_tolerance = 1.0e-8;

  void Validate() const;
};

/// Construct the rank-zero, parity-even, charge-conserving two-body charge
/// multipole directly in the normalized antisymmetrized J-coupled convention
/// used by Operator::TwoBody.  momentum_transfer_mev and G_E^S(Q^2) are
/// explicit so the caller can use the same q mesh and nucleon form factor as
/// the existing one-body response operators.
Operator TwoBodyChargeOperator(
    ModelSpace& modelspace, double momentum_transfer_mev,
    double ge_isoscalar,
    const TwoBodyChargeOperatorParameters& parameters);

} // namespace chiral_charge
} // namespace imsrg_util

#endif
