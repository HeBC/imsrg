#include "TwoBodyChargeCurrent.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>

namespace imsrg_util
{
namespace chiral_charge
{
namespace
{

using Complex = std::complex<double>;

arma::cx_mat PauliX()
{
  arma::cx_mat out(2, 2, arma::fill::zeros);
  out(0, 1) = 1.0;
  out(1, 0) = 1.0;
  return out;
}

arma::cx_mat PauliY()
{
  arma::cx_mat out(2, 2, arma::fill::zeros);
  out(0, 1) = Complex(0.0, -1.0);
  out(1, 0) = Complex(0.0, 1.0);
  return out;
}

arma::cx_mat PauliZ()
{
  arma::cx_mat out(2, 2, arma::fill::zeros);
  out(0, 0) = 1.0;
  out(1, 1) = -1.0;
  return out;
}

const arma::cx_mat& Identity2()
{
  static const arma::cx_mat identity = arma::eye<arma::cx_mat>(2, 2);
  return identity;
}

const arma::cx_mat& Identity16()
{
  static const arma::cx_mat identity = arma::eye<arma::cx_mat>(16, 16);
  return identity;
}

const TwoSpinMatrix& Identity4()
{
  static const TwoSpinMatrix identity = []() {
    TwoSpinMatrix value;
    value.eye();
    return value;
  }();
  return identity;
}

arma::cx_mat Embed(const arma::cx_mat& op, int factor)
{
  arma::cx_mat result(1, 1, arma::fill::ones);
  for (int i = 0; i < 4; ++i)
  {
    result = arma::kron(result, i == factor ? op : Identity2());
  }
  return result;
}

TwoSpinMatrix EmbedSpin(const arma::cx_mat& op, int factor)
{
  TwoSpinMatrix result;
  result = factor == 0 ? arma::kron(op, Identity2())
                       : arma::kron(Identity2(), op);
  return result;
}

const std::array<arma::cx_mat, 3>& Sigma1()
{
  static const std::array<arma::cx_mat, 3> ops = {
      Embed(PauliX(), 0), Embed(PauliY(), 0), Embed(PauliZ(), 0)};
  return ops;
}

const std::array<arma::cx_mat, 3>& Sigma2()
{
  static const std::array<arma::cx_mat, 3> ops = {
      Embed(PauliX(), 1), Embed(PauliY(), 1), Embed(PauliZ(), 1)};
  return ops;
}

const std::array<TwoSpinMatrix, 3>& SpinSigma1()
{
  static const std::array<TwoSpinMatrix, 3> ops = {
      EmbedSpin(PauliX(), 0), EmbedSpin(PauliY(), 0),
      EmbedSpin(PauliZ(), 0)};
  return ops;
}

const std::array<TwoSpinMatrix, 3>& SpinSigma2()
{
  static const std::array<TwoSpinMatrix, 3> ops = {
      EmbedSpin(PauliX(), 1), EmbedSpin(PauliY(), 1),
      EmbedSpin(PauliZ(), 1)};
  return ops;
}

const std::array<arma::cx_mat, 3>& Tau1()
{
  static const std::array<arma::cx_mat, 3> ops = {
      Embed(PauliX(), 2), Embed(PauliY(), 2), Embed(PauliZ(), 2)};
  return ops;
}

const std::array<arma::cx_mat, 3>& Tau2()
{
  static const std::array<arma::cx_mat, 3> ops = {
      Embed(PauliX(), 3), Embed(PauliY(), 3), Embed(PauliZ(), 3)};
  return ops;
}

arma::cx_mat VectorDot(const std::array<arma::cx_mat, 3>& operators,
                       const Vec3& vector)
{
  arma::cx_mat out(16, 16, arma::fill::zeros);
  for (int i = 0; i < 3; ++i)
  {
    out += vector(i) * operators[i];
  }
  return out;
}

TwoSpinMatrix SpinVectorDot(
    const std::array<TwoSpinMatrix, 3>& operators, const Vec3& vector)
{
  TwoSpinMatrix out;
  out.zeros();
  for (int i = 0; i < 3; ++i)
  {
    out += vector(i) * operators[i];
  }
  return out;
}

const arma::cx_mat& SigmaDotSigma()
{
  static const arma::cx_mat value =
      Sigma1()[0] * Sigma2()[0]
      + Sigma1()[1] * Sigma2()[1]
      + Sigma1()[2] * Sigma2()[2];
  return value;
}

const TwoSpinMatrix& SpinSigmaDotSigma()
{
  static const TwoSpinMatrix value =
      SpinSigma1()[0] * SpinSigma2()[0]
      + SpinSigma1()[1] * SpinSigma2()[1]
      + SpinSigma1()[2] * SpinSigma2()[2];
  return value;
}

const arma::cx_mat& TauDotTau()
{
  static const arma::cx_mat value =
      Tau1()[0] * Tau2()[0]
      + Tau1()[1] * Tau2()[1]
      + Tau1()[2] * Tau2()[2];
  return value;
}

double SquaredNorm(const Vec3& vector)
{
  return arma::dot(vector, vector);
}

bool IsValidRegulatorScheme(RegulatorScheme scheme)
{
  return scheme == RegulatorScheme::SMS
      || scheme == RegulatorScheme::Nonlocal
      || scheme == RegulatorScheme::Unregulated;
}

double RegulatorEnvelopeUnchecked(
    const Vec3& p, const Vec3& pprime,
    const ChargeDensityParameters& parameters)
{
  if (parameters.regulator_scheme != RegulatorScheme::Nonlocal)
  {
    return 1.0;
  }
  const double cutoff2 = parameters.cutoff_mev * parameters.cutoff_mev;
  const double p_power = std::pow(SquaredNorm(p) / cutoff2,
                                  parameters.regulator_exponent);
  const double pprime_power = std::pow(SquaredNorm(pprime) / cutoff2,
                                       parameters.regulator_exponent);
  return std::exp(-(p_power + pprime_power));
}

double ContactInternalRegulator(
    const Vec3& x, const Vec3& y,
    const ChargeDensityParameters& parameters)
{
  if (parameters.regulator_scheme != RegulatorScheme::SMS)
  {
    return 1.0;
  }
  return std::exp(-(SquaredNorm(x) + SquaredNorm(y))
                  / (parameters.cutoff_mev * parameters.cutoff_mev));
}

double RegulatedPionPropagatorUnchecked(
    double momentum_squared_mev2,
    const ChargeDensityParameters& parameters)
{
  const double denominator = momentum_squared_mev2
                           + parameters.pion_mass_mev
                             * parameters.pion_mass_mev;
  if (parameters.regulator_scheme != RegulatorScheme::SMS)
  {
    return 1.0 / denominator;
  }
  const double cutoff2 = parameters.cutoff_mev * parameters.cutoff_mev;
  return std::exp(-denominator / cutoff2) / denominator;
}

double RegulatedSquaredPionPropagatorUnchecked(
    double momentum_squared_mev2,
    const ChargeDensityParameters& parameters)
{
  const double denominator = momentum_squared_mev2
                           + parameters.pion_mass_mev
                             * parameters.pion_mass_mev;
  if (parameters.regulator_scheme != RegulatorScheme::SMS)
  {
    return 1.0 / (denominator * denominator);
  }
  const double cutoff2 = parameters.cutoff_mev * parameters.cutoff_mev;
  return (1.0 / (denominator * denominator)
          + 1.0 / (cutoff2 * denominator))
         * std::exp(-denominator / cutoff2);
}

void ValidateVector(const Vec3& vector, const char* name)
{
  if (!vector.is_finite())
  {
    std::ostringstream message;
    message << name << " contains a non-finite component";
    throw std::invalid_argument(message.str());
  }
}

double ContactE1(const Vec3& x, const Vec3& y,
                 const ChargeDensityParameters& parameters)
{
  const double x2 = SquaredNorm(x);
  const double y2 = SquaredNorm(y);
  const double regulator = ContactInternalRegulator(x, y, parameters);
  return (x2 - y2) * regulator;
}

SpinIsospinMatrix ContactE2(const Vec3& x, const Vec3& y,
                            const ChargeDensityParameters& parameters)
{
  const double x2 = SquaredNorm(x);
  const double y2 = SquaredNorm(y);
  const double regulator = ContactInternalRegulator(x, y, parameters);
  return (VectorDot(Sigma1(), x) * VectorDot(Sigma2(), x)
          - VectorDot(Sigma1(), y) * VectorDot(Sigma2(), y))
         * regulator;
}

TwoSpinMatrix ContactE2Spin(const Vec3& x, const Vec3& y,
                            const ChargeDensityParameters& parameters)
{
  const double x2 = SquaredNorm(x);
  const double y2 = SquaredNorm(y);
  const double regulator = ContactInternalRegulator(x, y, parameters);
  return (SpinVectorDot(SpinSigma1(), x)
              * SpinVectorDot(SpinSigma2(), x)
          - SpinVectorDot(SpinSigma1(), y)
              * SpinVectorDot(SpinSigma2(), y))
         * regulator;
}

double ContactF1(const Vec3& p, const Vec3& pprime, const Vec3& k,
                 const ChargeDensityParameters& parameters)
{
  return ContactE1(p - 0.5 * k, pprime, parameters)
       + ContactE1(p + 0.5 * k, pprime, parameters)
       + ContactE1(pprime - 0.5 * k, p, parameters)
       + ContactE1(pprime + 0.5 * k, p, parameters);
}

SpinIsospinMatrix ContactF2(const Vec3& p, const Vec3& pprime,
                            const Vec3& k,
                            const ChargeDensityParameters& parameters)
{
  return ContactE2(p - 0.5 * k, pprime, parameters)
       + ContactE2(p + 0.5 * k, pprime, parameters)
       + ContactE2(pprime - 0.5 * k, p, parameters)
       + ContactE2(pprime + 0.5 * k, p, parameters);
}

TwoSpinMatrix ContactF2Spin(const Vec3& p, const Vec3& pprime,
                            const Vec3& k,
                            const ChargeDensityParameters& parameters)
{
  return ContactE2Spin(p - 0.5 * k, pprime, parameters)
       + ContactE2Spin(p + 0.5 * k, pprime, parameters)
       + ContactE2Spin(pprime - 0.5 * k, p, parameters)
       + ContactE2Spin(pprime + 0.5 * k, p, parameters);
}

TwoSpinMatrix OnePionExchangeSpin(
    const Vec3& q1, const Vec3& q2, const Vec3& k,
    int total_isospin, double ge_isoscalar,
    const ChargeDensityParameters& parameters)
{
  const double coefficient = parameters.electric_charge * ge_isoscalar
      * parameters.g_a * parameters.g_a
      / (16.0 * parameters.f_pi_mev * parameters.f_pi_mev
         * parameters.nucleon_mass_mev);
  const double tau_eigenvalue = 2.0 * total_isospin
                              * (total_isospin + 1.0) - 3.0;
  const double coefficient_beta9 = 1.0 - 2.0 * parameters.beta_9;
  const double coefficient_beta8 = 2.0 * parameters.beta_8 - 1.0;

  const TwoSpinMatrix direct =
      coefficient_beta9
          * SpinVectorDot(SpinSigma1(), k)
          * SpinVectorDot(SpinSigma2(), q2)
          * RegulatedPionPropagatorUnchecked(SquaredNorm(q2), parameters)
      + coefficient_beta8
          * SpinVectorDot(SpinSigma1(), q2)
          * SpinVectorDot(SpinSigma2(), q2)
          * arma::dot(q2, k)
          * RegulatedSquaredPionPropagatorUnchecked(
              SquaredNorm(q2), parameters);
  const TwoSpinMatrix exchanged =
      coefficient_beta9
          * SpinVectorDot(SpinSigma2(), k)
          * SpinVectorDot(SpinSigma1(), q1)
          * RegulatedPionPropagatorUnchecked(SquaredNorm(q1), parameters)
      + coefficient_beta8
          * SpinVectorDot(SpinSigma2(), q1)
          * SpinVectorDot(SpinSigma1(), q1)
          * arma::dot(q1, k)
          * RegulatedSquaredPionPropagatorUnchecked(
              SquaredNorm(q1), parameters);
  return coefficient * tau_eigenvalue * (direct + exchanged);
}

TwoSpinMatrix ContactSpin(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters)
{
  const double f1 = ContactF1(p, pprime, k, parameters);
  const TwoSpinMatrix f2 = ContactF2Spin(p, pprime, k, parameters);
  const TwoSpinMatrix ab = parameters.contact.A * Identity4()
                         + parameters.contact.B * SpinSigmaDotSigma();
  return 2.0 * parameters.electric_charge * ge_isoscalar
       * (f1 * ab + parameters.contact.C * f2);
}

} // namespace

const char* RegulatorSchemeName(RegulatorScheme scheme)
{
  switch (scheme)
  {
    case RegulatorScheme::SMS:
      return "sms";
    case RegulatorScheme::Nonlocal:
      return "nonlocal";
    case RegulatorScheme::Unregulated:
      return "unregulated";
  }
  throw std::invalid_argument("Unknown two-body charge regulator scheme");
}

RegulatorScheme RegulatorSchemeFromString(const std::string& name)
{
  std::string normalized = name;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (normalized == "sms" || normalized == "filin")
  {
    return RegulatorScheme::SMS;
  }
  if (normalized == "nonlocal" || normalized == "go"
      || normalized == "em" || normalized == "em18-20")
  {
    return RegulatorScheme::Nonlocal;
  }
  if (normalized == "none" || normalized == "unregulated")
  {
    return RegulatorScheme::Unregulated;
  }
  throw std::invalid_argument(
      "Unknown two-body charge regulator '" + name
      + "'; choose sms, nonlocal, or unregulated");
}

ContactLECs ContactLECs::FromDimensionlessM(
    double M1, double M2, double M3,
    double f_pi_mev, double breakdown_scale_mev)
{
  if (!(std::isfinite(M1) && std::isfinite(M2) && std::isfinite(M3)
        && std::isfinite(f_pi_mev) && f_pi_mev > 0.0
        && std::isfinite(breakdown_scale_mev)
        && breakdown_scale_mev > 0.0))
  {
    throw std::invalid_argument("Invalid contact-LEC conversion input");
  }
  const double unit = 1.0 / (f_pi_mev * f_pi_mev
                             * std::pow(breakdown_scale_mev, 3));
  ContactLECs result;
  result.A = 0.25 * (3.0 * M1 + M3) * unit;
  result.B = (0.25 * (M1 - M3) - M2 / 3.0) * unit;
  result.C = M2 * unit;
  return result;
}

std::array<double, 3> ContactLECs::ToDimensionlessM(
    double f_pi_mev, double breakdown_scale_mev) const
{
  if (!(std::isfinite(A) && std::isfinite(B) && std::isfinite(C)
        && std::isfinite(f_pi_mev) && f_pi_mev > 0.0
        && std::isfinite(breakdown_scale_mev)
        && breakdown_scale_mev > 0.0))
  {
    throw std::invalid_argument("Invalid contact-LEC conversion input");
  }
  const double inverse_unit = f_pi_mev * f_pi_mev
                            * std::pow(breakdown_scale_mev, 3);
  return {{(A + B + C / 3.0) * inverse_unit,
           C * inverse_unit,
           (A - 3.0 * B - C) * inverse_unit}};
}

void ChargeDensityParameters::Validate() const
{
  const bool finite = std::isfinite(g_a)
                   && std::isfinite(f_pi_mev)
                   && std::isfinite(nucleon_mass_mev)
                   && std::isfinite(pion_mass_mev)
                   && std::isfinite(cutoff_mev)
                   && std::isfinite(beta_8)
                   && std::isfinite(beta_9)
                   && std::isfinite(electric_charge)
                   && std::isfinite(contact.A)
                   && std::isfinite(contact.B)
                   && std::isfinite(contact.C);
  if (!finite || !IsValidRegulatorScheme(regulator_scheme)
      || regulator_exponent <= 0
      || f_pi_mev <= 0.0 || nucleon_mass_mev <= 0.0
      || pion_mass_mev <= 0.0 || cutoff_mev <= 0.0)
  {
    throw std::invalid_argument(
        "Charge-density parameters must be finite and all scales positive");
  }
}

Vec3 MakeVec3(double x, double y, double z)
{
  Vec3 result;
  result(0) = x;
  result(1) = y;
  result(2) = z;
  return result;
}

Vec3 MomentumTransfer1(const Vec3& p, const Vec3& pprime, const Vec3& k)
{
  ValidateVector(p, "p");
  ValidateVector(pprime, "pprime");
  ValidateVector(k, "k");
  return pprime - p + 0.5 * k;
}

Vec3 MomentumTransfer2(const Vec3& p, const Vec3& pprime, const Vec3& k)
{
  ValidateVector(p, "p");
  ValidateVector(pprime, "pprime");
  ValidateVector(k, "k");
  return p - pprime + 0.5 * k;
}

double RegulatedPionPropagator(
    double momentum_squared_mev2,
    const ChargeDensityParameters& parameters)
{
  parameters.Validate();
  if (!std::isfinite(momentum_squared_mev2)
      || momentum_squared_mev2 < 0.0)
  {
    throw std::invalid_argument("Pion momentum squared must be finite and nonnegative");
  }
  return RegulatedPionPropagatorUnchecked(
      momentum_squared_mev2, parameters);
}

double RegulatedSquaredPionPropagator(
    double momentum_squared_mev2,
    const ChargeDensityParameters& parameters)
{
  parameters.Validate();
  if (!std::isfinite(momentum_squared_mev2)
      || momentum_squared_mev2 < 0.0)
  {
    throw std::invalid_argument("Pion momentum squared must be finite and nonnegative");
  }
  return RegulatedSquaredPionPropagatorUnchecked(
      momentum_squared_mev2, parameters);
}

double RegulatorEnvelope(
    const Vec3& p, const Vec3& pprime,
    const ChargeDensityParameters& parameters)
{
  parameters.Validate();
  ValidateVector(p, "p");
  ValidateVector(pprime, "pprime");
  return RegulatorEnvelopeUnchecked(p, pprime, parameters);
}

SpinIsospinMatrix OnePionExchange(
    const Vec3& q1, const Vec3& q2, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters)
{
  parameters.Validate();
  ValidateVector(q1, "q1");
  ValidateVector(q2, "q2");
  ValidateVector(k, "k");
  if (!std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument("G_E^S must be finite");
  }
  if (parameters.regulator_scheme == RegulatorScheme::Nonlocal)
  {
    throw std::invalid_argument(
        "OnePionExchange(q1,q2,k) cannot apply a nonlocal p,pprime "
        "envelope; use OnePionExchangeFromMomenta or "
        "TwoBodyChargeDensity");
  }

  const double coefficient = parameters.electric_charge * ge_isoscalar
      * parameters.g_a * parameters.g_a
      / (16.0 * parameters.f_pi_mev * parameters.f_pi_mev
         * parameters.nucleon_mass_mev);
  const double coefficient_beta9 = 1.0 - 2.0 * parameters.beta_9;
  const double coefficient_beta8 = 2.0 * parameters.beta_8 - 1.0;

  const arma::cx_mat sigma1_k = VectorDot(Sigma1(), k);
  const arma::cx_mat sigma2_k = VectorDot(Sigma2(), k);
  const arma::cx_mat sigma1_q1 = VectorDot(Sigma1(), q1);
  const arma::cx_mat sigma2_q1 = VectorDot(Sigma2(), q1);
  const arma::cx_mat sigma1_q2 = VectorDot(Sigma1(), q2);
  const arma::cx_mat sigma2_q2 = VectorDot(Sigma2(), q2);

  // The beta_8 term deliberately contains one contraction from each
  // nucleon.  Replacing either product by (sigma_i.q)^2 is incorrect.
  const arma::cx_mat direct =
      coefficient_beta9 * sigma1_k * sigma2_q2
          * RegulatedPionPropagator(SquaredNorm(q2), parameters)
      + coefficient_beta8 * sigma1_q2 * sigma2_q2
          * arma::dot(q2, k)
          * RegulatedSquaredPionPropagator(SquaredNorm(q2), parameters);

  const arma::cx_mat exchanged =
      coefficient_beta9 * sigma2_k * sigma1_q1
          * RegulatedPionPropagator(SquaredNorm(q1), parameters)
      + coefficient_beta8 * sigma2_q1 * sigma1_q1
          * arma::dot(q1, k)
          * RegulatedSquaredPionPropagator(SquaredNorm(q1), parameters);

  return coefficient * TauDotTau() * (direct + exchanged);
}

SpinIsospinMatrix OnePionExchangeFromMomenta(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters)
{
  parameters.Validate();
  ValidateVector(p, "p");
  ValidateVector(pprime, "pprime");
  ValidateVector(k, "k");
  ChargeDensityParameters propagator_parameters = parameters;
  if (parameters.regulator_scheme == RegulatorScheme::Nonlocal)
  {
    propagator_parameters.regulator_scheme = RegulatorScheme::Unregulated;
  }
  return RegulatorEnvelopeUnchecked(p, pprime, parameters)
       * OnePionExchange(MomentumTransfer1(p, pprime, k),
                         MomentumTransfer2(p, pprime, k),
                         k, ge_isoscalar, propagator_parameters);
}

SpinIsospinMatrix Contact(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters)
{
  parameters.Validate();
  ValidateVector(p, "p");
  ValidateVector(pprime, "pprime");
  ValidateVector(k, "k");
  if (!std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument("G_E^S must be finite");
  }

  const double f1 = ContactF1(p, pprime, k, parameters);
  const arma::cx_mat f2 = ContactF2(p, pprime, k, parameters);
  const arma::cx_mat ab = parameters.contact.A * Identity16()
                        + parameters.contact.B * SigmaDotSigma();
  return RegulatorEnvelopeUnchecked(p, pprime, parameters)
       * 2.0 * parameters.electric_charge * ge_isoscalar
       * (f1 * ab + parameters.contact.C * f2);
}

SpinIsospinMatrix TwoBodyChargeDensity(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    bool include_ope, bool include_contact)
{
  SpinIsospinMatrix result(16, 16, arma::fill::zeros);
  if (include_ope)
  {
    result += OnePionExchangeFromMomenta(
        p, pprime, k, ge_isoscalar, parameters);
  }
  if (include_contact)
  {
    result += Contact(p, pprime, k, ge_isoscalar, parameters);
  }
  return result;
}

TwoSpinMatrix TwoBodyChargeSpinMatrix(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    int total_isospin, double ge_isoscalar,
    const ChargeDensityParameters& parameters,
    bool include_ope, bool include_contact)
{
  parameters.Validate();
  ValidateVector(p, "p");
  ValidateVector(pprime, "pprime");
  ValidateVector(k, "k");
  if ((total_isospin != 0 && total_isospin != 1)
      || !std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument(
        "Spin-space charge kernel requires T=0 or 1 and finite G_E^S");
  }

  TwoSpinMatrix result;
  result.zeros();
  if (include_ope)
  {
    result += OnePionExchangeSpin(
        MomentumTransfer1(p, pprime, k),
        MomentumTransfer2(p, pprime, k), k,
        total_isospin, ge_isoscalar, parameters);
  }
  if (include_contact)
  {
    result += ContactSpin(
        p, pprime, k, ge_isoscalar, parameters);
  }
  return RegulatorEnvelopeUnchecked(p, pprime, parameters) * result;
}

double HermiticityResidual(
    const Vec3& p, const Vec3& pprime, const Vec3& k,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    bool include_ope, bool include_contact)
{
  const SpinIsospinMatrix lhs = TwoBodyChargeDensity(
      p, pprime, k, ge_isoscalar, parameters,
      include_ope, include_contact);
  const SpinIsospinMatrix rhs = TwoBodyChargeDensity(
      pprime, p, -k, ge_isoscalar, parameters,
      include_ope, include_contact).t();
  const double scale = std::max(1.0, std::max(arma::norm(lhs, "fro"),
                                             arma::norm(rhs, "fro")));
  return arma::norm(lhs - rhs, "fro") / scale;
}

} // namespace chiral_charge
} // namespace imsrg_util
// ============================================================================
// Numerical partial-wave, relative-HO, and pair-center-of-mass projection.
// ============================================================================

#include "AngMom.hh"
#include "PhysicalConstants.hh"

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_legendre.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace imsrg_util
{
namespace chiral_charge
{
namespace
{

const double kPi = std::acos(-1.0);

struct AngularPoint
{
  double cos_theta;
  double phi;
  double weight;
};

int Phase(int exponent)
{
  return (std::abs(exponent) % 2 == 0) ? 1 : -1;
}

std::complex<double> SphericalHarmonic(
    int l, int m, double cos_theta, double phi)
{
  const int abs_m = std::abs(m);
  const double normalized_legendre =
      gsl_sf_legendre_sphPlm(l, abs_m, cos_theta);
  const std::complex<double> positive = normalized_legendre
      * std::exp(std::complex<double>(0.0, abs_m * phi));
  if (m >= 0)
  {
    return positive;
  }
  return static_cast<double>(Phase(abs_m)) * std::conj(positive);
}

std::vector<AngularPoint> MakePolarMesh(
    const ProjectionParameters& quadrature)
{
  quadrature.Validate();
  gsl_integration_glfixed_table* polar =
      gsl_integration_glfixed_table_alloc(quadrature.polar_order);
  if (polar == nullptr)
  {
    throw std::runtime_error("Unable to allocate Gauss-Legendre angular mesh");
  }

  std::vector<AngularPoint> points;
  points.reserve(quadrature.polar_order);
  for (int ix = 0; ix < quadrature.polar_order; ++ix)
  {
    double x = 0.0;
    double wx = 0.0;
    gsl_integration_glfixed_point(-1.0, 1.0, ix, &x, &wx, polar);
    points.push_back({x, 0.0, wx});
  }
  gsl_integration_glfixed_table_free(polar);
  return points;
}

std::vector<AngularPoint> MakeAngularMesh(
    const ProjectionParameters& quadrature)
{
  const std::vector<AngularPoint> polar_mesh = MakePolarMesh(quadrature);
  std::vector<AngularPoint> points;
  points.reserve(
      quadrature.polar_order * quadrature.azimuthal_order);
  const double phi_weight = 2.0 * kPi / quadrature.azimuthal_order;
  for (const AngularPoint& polar : polar_mesh)
  {
    for (int iphi = 0; iphi < quadrature.azimuthal_order; ++iphi)
    {
      points.push_back({
          polar.cos_theta,
          2.0 * kPi * iphi / quadrature.azimuthal_order,
          polar.weight * phi_weight});
    }
  }
  return points;
}

Vec3 MomentumVector(double magnitude, const AngularPoint& point)
{
  const double sin_theta = std::sqrt(std::max(
      0.0, 1.0 - point.cos_theta * point.cos_theta));
  return MakeVec3(magnitude * sin_theta * std::cos(point.phi),
                  magnitude * sin_theta * std::sin(point.phi),
                  magnitude * point.cos_theta);
}

double ProjectionCG(double j1, double m1, double j2, double m2,
                    double J, double M)
{
  if (std::abs(m1 + m2 - M) > 1.0e-12)
  {
    return 0.0;
  }
  return AngMom::CG(j1, m1, j2, m2, J, M);
}

arma::cx_vec CoupledAngularState(
    const PartialWaveChannel& channel, const AngularPoint& point)
{
  arma::cx_vec state(16, arma::fill::zeros);
  for (int ml = -channel.l; ml <= channel.l; ++ml)
  {
    for (int ms = -channel.S; ms <= channel.S; ++ms)
    {
      const double orbital_spin_cg = ProjectionCG(
          channel.l, ml, channel.S, ms, channel.J, channel.M);
      if (std::abs(orbital_spin_cg) < 1.0e-15)
      {
        continue;
      }
      const std::complex<double> angular =
          SphericalHarmonic(channel.l, ml, point.cos_theta, point.phi);
      for (int spin1 = 0; spin1 < 2; ++spin1)
      {
        const double ms1 = spin1 == 0 ? 0.5 : -0.5;
        for (int spin2 = 0; spin2 < 2; ++spin2)
        {
          const double ms2 = spin2 == 0 ? 0.5 : -0.5;
          const double spin_cg = ProjectionCG(
              0.5, ms1, 0.5, ms2, channel.S, ms);
          if (std::abs(spin_cg) < 1.0e-15)
          {
            continue;
          }
          for (int iso1 = 0; iso1 < 2; ++iso1)
          {
            const double mt1 = iso1 == 0 ? 0.5 : -0.5;
            for (int iso2 = 0; iso2 < 2; ++iso2)
            {
              const double mt2 = iso2 == 0 ? 0.5 : -0.5;
              const double iso_cg = ProjectionCG(
                  0.5, mt1, 0.5, mt2, channel.T, channel.MT);
              if (std::abs(iso_cg) < 1.0e-15)
              {
                continue;
              }
              const int index = ((spin1 * 2 + spin2) * 2 + iso1) * 2
                              + iso2;
              state(index) += angular * orbital_spin_cg * spin_cg * iso_cg;
            }
          }
        }
      }
    }
  }
  return state;
}

arma::cx_vec CoupledAngularSpinState(
    const PartialWaveChannel& channel, const AngularPoint& point)
{
  arma::cx_vec state(4, arma::fill::zeros);
  for (int ml = -channel.l; ml <= channel.l; ++ml)
  {
    for (int ms = -channel.S; ms <= channel.S; ++ms)
    {
      const double orbital_spin_cg = ProjectionCG(
          channel.l, ml, channel.S, ms, channel.J, channel.M);
      if (std::abs(orbital_spin_cg) < 1.0e-15)
      {
        continue;
      }
      const std::complex<double> angular =
          SphericalHarmonic(channel.l, ml, point.cos_theta, point.phi);
      for (int spin1 = 0; spin1 < 2; ++spin1)
      {
        const double ms1 = spin1 == 0 ? 0.5 : -0.5;
        for (int spin2 = 0; spin2 < 2; ++spin2)
        {
          const double ms2 = spin2 == 0 ? 0.5 : -0.5;
          const double spin_cg = ProjectionCG(
              0.5, ms1, 0.5, ms2, channel.S, ms);
          state(2 * spin1 + spin2) +=
              angular * orbital_spin_cg * spin_cg;
        }
      }
    }
  }
  return state;
}

using AngularStateFunction = std::function<arma::cx_vec(
    const PartialWaveChannel&, const AngularPoint&)>;

void ValidateProjectionInputs(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    const std::vector<double>& pprime_mev,
    const std::vector<double>& p_mev, double k_mev,
    const ProjectionParameters& quadrature)
{
  bra.Validate();
  ket.Validate();
  quadrature.Validate();
  if (pprime_mev.empty() || p_mev.empty()
      || !std::isfinite(k_mev) || k_mev < 0.0)
  {
    throw std::invalid_argument(
        "Partial-wave momentum grids must be nonempty and k nonnegative");
  }
  for (double momentum : pprime_mev)
  {
    if (!std::isfinite(momentum) || momentum < 0.0)
    {
      throw std::invalid_argument(
          "Bra partial-wave momentum must be finite and nonnegative");
    }
  }
  for (double momentum : p_mev)
  {
    if (!std::isfinite(momentum) || momentum < 0.0)
    {
      throw std::invalid_argument(
          "Ket partial-wave momentum must be finite and nonnegative");
    }
  }
}

template<typename MatrixKernel>
std::vector<std::complex<double>> ProjectAxialKernelGrid(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    const std::vector<double>& pprime_mev,
    const std::vector<double>& p_mev, double k_mev,
    const ProjectionParameters& quadrature,
    const AngularStateFunction& state_function,
    const MatrixKernel& kernel,
    std::size_t matrix_dimension)
{
  const std::vector<AngularPoint> polar_mesh = MakePolarMesh(quadrature);
  const double phi_weight = 2.0 * kPi / quadrature.azimuthal_order;
  std::vector<AngularPoint> bra_angles;
  bra_angles.reserve(
      polar_mesh.size() * quadrature.azimuthal_order);
  for (const AngularPoint& polar : polar_mesh)
  {
    for (int iphi = 0; iphi < quadrature.azimuthal_order; ++iphi)
    {
      bra_angles.push_back({
          polar.cos_theta,
          2.0 * kPi * iphi / quadrature.azimuthal_order,
          polar.weight * phi_weight});
    }
  }

  std::vector<arma::cx_vec> bra_states;
  std::vector<arma::cx_vec> ket_states;
  bra_states.reserve(bra_angles.size());
  ket_states.reserve(polar_mesh.size());
  for (const AngularPoint& point : bra_angles)
  {
    bra_states.push_back(state_function(bra, point));
  }
  for (const AngularPoint& point : polar_mesh)
  {
    ket_states.push_back(state_function(ket, point));
  }

  const Vec3 k = MakeVec3(0.0, 0.0, k_mev);
  std::vector<std::complex<double>> results(
      pprime_mev.size() * p_mev.size(), 0.0);
  if (bra.M != ket.M)
  {
    return results;
  }

  std::vector<std::vector<Vec3>> bra_momentum_grid(pprime_mev.size());
  std::vector<std::vector<Vec3>> ket_momentum_grid(p_mev.size());
  for (std::size_t ipprime = 0; ipprime < pprime_mev.size(); ++ipprime)
  {
    bra_momentum_grid[ipprime].reserve(bra_angles.size());
    for (const AngularPoint& point : bra_angles)
    {
      bra_momentum_grid[ipprime].push_back(
          MomentumVector(pprime_mev[ipprime], point));
    }
  }
  for (std::size_t ip = 0; ip < p_mev.size(); ++ip)
  {
    ket_momentum_grid[ip].reserve(polar_mesh.size());
    for (const AngularPoint& point : polar_mesh)
    {
      ket_momentum_grid[ip].push_back(
          MomentumVector(p_mev[ip], point));
    }
  }

  const std::size_t number_bra_momenta = pprime_mev.size();
  const std::size_t number_ket_momenta = p_mev.size();
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (std::size_t ipprime = 0; ipprime < number_bra_momenta; ++ipprime)
  {
    for (std::size_t ip = 0; ip < number_ket_momenta; ++ip)
    {
      std::complex<double>& result = results[ipprime * p_mev.size() + ip];
      for (std::size_t ibra = 0; ibra < bra_angles.size(); ++ibra)
      {
        for (std::size_t iket = 0; iket < polar_mesh.size(); ++iket)
        {
          const auto value = kernel(
              ket_momentum_grid[ip][iket],
              bra_momentum_grid[ipprime][ibra], k);
          if (value.n_rows != matrix_dimension
              || value.n_cols != matrix_dimension || !value.is_finite())
          {
            throw std::runtime_error(
                "Pointwise kernel returned an invalid matrix");
          }
          result += 2.0 * kPi * bra_angles[ibra].weight
                  * polar_mesh[iket].weight
                  * arma::cdot(bra_states[ibra], value * ket_states[iket]);
        }
      }
    }
  }
  return results;
}

} // namespace

void PartialWaveChannel::Validate() const
{
  if (l < 0 || (S != 0 && S != 1) || (T != 0 && T != 1)
      || J < std::abs(l - S) || J > l + S
      || std::abs(M) > J || std::abs(MT) > T)
  {
    throw std::invalid_argument("Invalid relative partial-wave channel");
  }
}

void ProjectionParameters::Validate() const
{
  if (polar_order < 2 || azimuthal_order < 2)
  {
    throw std::invalid_argument(
        "Angular quadrature requires at least two polar and azimuthal points");
  }
}

void RadialProjectionParameters::Validate() const
{
  if (radial_order < 2 || !std::isfinite(max_momentum_mev)
      || max_momentum_mev <= 0.0)
  {
    throw std::invalid_argument(
        "Radial quadrature requires at least two points and positive p_max");
  }
}

void CenterOfMassProjectionParameters::Validate() const
{
  if (radial_order < 2 || !std::isfinite(max_radius_fm)
      || max_radius_fm <= 0.0)
  {
    throw std::invalid_argument(
        "Pair-CM quadrature requires at least two points and positive r_max");
  }
}

std::complex<double> ProjectKernel(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    double pprime_mev, double p_mev, double k_mev,
    const ProjectionParameters& quadrature,
    const KernelFunction& kernel)
{
  const std::vector<std::complex<double>> values = ProjectKernelGrid(
      bra, ket, std::vector<double>{pprime_mev},
      std::vector<double>{p_mev}, k_mev, quadrature, kernel);
  return values.front();
}

std::vector<std::complex<double>> ProjectKernelGrid(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    const std::vector<double>& pprime_mev,
    const std::vector<double>& p_mev, double k_mev,
    const ProjectionParameters& quadrature,
    const KernelFunction& kernel)
{
  ValidateProjectionInputs(
      bra, ket, pprime_mev, p_mev, k_mev, quadrature);
  if (!kernel)
  {
    throw std::invalid_argument("Partial-wave kernel callback is empty");
  }

  const std::vector<AngularPoint> mesh = MakeAngularMesh(quadrature);
  std::vector<arma::cx_vec> bra_states;
  std::vector<arma::cx_vec> ket_states;
  bra_states.reserve(mesh.size());
  ket_states.reserve(mesh.size());
  for (const AngularPoint& point : mesh)
  {
    bra_states.push_back(CoupledAngularState(bra, point));
    ket_states.push_back(CoupledAngularState(ket, point));
  }

  const Vec3 k = MakeVec3(0.0, 0.0, k_mev);
  std::vector<std::complex<double>> results(
      pprime_mev.size() * p_mev.size(), 0.0);
  std::vector<std::vector<Vec3>> bra_momentum_grid(pprime_mev.size());
  std::vector<std::vector<Vec3>> ket_momentum_grid(p_mev.size());
  for (std::size_t ipprime = 0; ipprime < pprime_mev.size(); ++ipprime)
  {
    bra_momentum_grid[ipprime].reserve(mesh.size());
    for (const AngularPoint& point : mesh)
    {
      bra_momentum_grid[ipprime].push_back(
          MomentumVector(pprime_mev[ipprime], point));
    }
  }
  for (std::size_t ip = 0; ip < p_mev.size(); ++ip)
  {
    ket_momentum_grid[ip].reserve(mesh.size());
    for (const AngularPoint& point : mesh)
    {
      ket_momentum_grid[ip].push_back(
          MomentumVector(p_mev[ip], point));
    }
  }

  const std::size_t number_bra_momenta = pprime_mev.size();
  const std::size_t number_ket_momenta = p_mev.size();
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (std::size_t ipprime = 0; ipprime < number_bra_momenta; ++ipprime)
  {
    for (std::size_t ip = 0; ip < number_ket_momenta; ++ip)
    {
      std::complex<double>& result = results[ipprime * p_mev.size() + ip];
      for (std::size_t ibra = 0; ibra < mesh.size(); ++ibra)
      {
        for (std::size_t iket = 0; iket < mesh.size(); ++iket)
        {
          const SpinIsospinMatrix value = kernel(
              ket_momentum_grid[ip][iket],
              bra_momentum_grid[ipprime][ibra], k);
          if (value.n_rows != 16 || value.n_cols != 16 || !value.is_finite())
          {
            throw std::runtime_error(
                "Pointwise kernel must return a finite 16x16 matrix");
          }
          result += mesh[ibra].weight * mesh[iket].weight
                  * arma::cdot(bra_states[ibra], value * ket_states[iket]);
        }
      }
    }
  }
  return results;
}

std::vector<std::complex<double>> ProjectChargeKernelGrid(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    const std::vector<double>& pprime_mev,
    const std::vector<double>& p_mev, double k_mev,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    const ProjectionParameters& quadrature,
    bool include_ope, bool include_contact)
{
  ValidateProjectionInputs(
      bra, ket, pprime_mev, p_mev, k_mev, quadrature);
  parameters.Validate();
  if (!std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument("G_E^S must be finite");
  }
  if (bra.T != ket.T || bra.MT != ket.MT)
  {
    return std::vector<std::complex<double>>(
        pprime_mev.size() * p_mev.size(), 0.0);
  }
  return ProjectAxialKernelGrid(
      bra, ket, pprime_mev, p_mev, k_mev, quadrature,
      CoupledAngularSpinState,
      [&](const Vec3& p, const Vec3& pprime, const Vec3& k) {
        return TwoBodyChargeSpinMatrix(
            p, pprime, k, bra.T, ge_isoscalar, parameters,
            include_ope, include_contact);
      },
      4);
}

std::complex<double> PartialWaveME(
    const PartialWaveChannel& bra, const PartialWaveChannel& ket,
    double pprime_mev, double p_mev, double k_mev,
    double ge_isoscalar, const ChargeDensityParameters& parameters,
    const ProjectionParameters& quadrature,
    bool include_ope, bool include_contact)
{
  parameters.Validate();
  if (!std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument("G_E^S must be finite");
  }
  const std::vector<std::complex<double>> values = ProjectChargeKernelGrid(
      bra, ket, std::vector<double>{pprime_mev},
      std::vector<double>{p_mev}, k_mev, ge_isoscalar,
      parameters, quadrature, include_ope, include_contact);
  return values.front();
}

double RelativeHOMomentumRadial(
    int n, int l, double hbar_omega_mev, double momentum_mev,
    double nucleon_mass_mev)
{
  if (n < 0 || l < 0 || !std::isfinite(hbar_omega_mev)
      || hbar_omega_mev <= 0.0 || !std::isfinite(momentum_mev)
      || momentum_mev < 0.0 || !std::isfinite(nucleon_mass_mev)
      || nucleon_mass_mev <= 0.0)
  {
    throw std::invalid_argument("Invalid relative HO radial-wave-function input");
  }

  const double b_mev_inverse =
      std::sqrt(2.0 / (nucleon_mass_mev * hbar_omega_mev));
  const double x = momentum_mev * b_mev_inverse;
  const double normalization = 2.0 * std::sqrt(
      gsl_sf_fact(n) * std::pow(2.0, n + l)
      / (std::sqrt(kPi) * gsl_sf_doublefact(2 * n + 2 * l + 1))
      * std::pow(b_mev_inverse, 3));
  return static_cast<double>(Phase(n)) * normalization
       * std::pow(x, l) * std::exp(-0.5 * x * x)
       * gsl_sf_laguerre_n(n, l + 0.5, x * x);
}

std::complex<double> RelativeHOMatrixElement(
    int nbra, const PartialWaveChannel& bra,
    int nket, const PartialWaveChannel& ket,
    double hbar_omega_mev, double k_mev,
    double nucleon_mass_mev,
    const ProjectionParameters& angular_quadrature,
    const RadialProjectionParameters& radial_quadrature,
    const KernelFunction& kernel)
{
  bra.Validate();
  ket.Validate();
  angular_quadrature.Validate();
  radial_quadrature.Validate();
  if (nbra < 0 || nket < 0 || !std::isfinite(hbar_omega_mev)
      || hbar_omega_mev <= 0.0 || !std::isfinite(k_mev) || k_mev < 0.0
      || !std::isfinite(nucleon_mass_mev) || nucleon_mass_mev <= 0.0)
  {
    throw std::invalid_argument("Invalid relative HO projection input");
  }
  if (!kernel)
  {
    throw std::invalid_argument("Relative HO kernel callback is empty");
  }

  gsl_integration_glfixed_table* radial =
      gsl_integration_glfixed_table_alloc(radial_quadrature.radial_order);
  if (radial == nullptr)
  {
    throw std::runtime_error("Unable to allocate radial Gauss-Legendre mesh");
  }

  struct RadialPoint
  {
    double momentum;
    double weight;
    double wave_function;
  };
  std::vector<RadialPoint> bra_mesh;
  std::vector<RadialPoint> ket_mesh;
  bra_mesh.reserve(radial_quadrature.radial_order);
  ket_mesh.reserve(radial_quadrature.radial_order);
  for (int i = 0; i < radial_quadrature.radial_order; ++i)
  {
    double momentum = 0.0;
    double weight = 0.0;
    gsl_integration_glfixed_point(
        0.0, radial_quadrature.max_momentum_mev, i,
        &momentum, &weight, radial);
    bra_mesh.push_back({
        momentum, weight,
        RelativeHOMomentumRadial(
            nbra, bra.l, hbar_omega_mev, momentum, nucleon_mass_mev)});
    ket_mesh.push_back({
        momentum, weight,
        RelativeHOMomentumRadial(
            nket, ket.l, hbar_omega_mev, momentum, nucleon_mass_mev)});
  }
  gsl_integration_glfixed_table_free(radial);

  std::complex<double> result = 0.0;
  for (const RadialPoint& bra_point : bra_mesh)
  {
    for (const RadialPoint& ket_point : ket_mesh)
    {
      result += bra_point.weight * ket_point.weight
              * bra_point.momentum * bra_point.momentum
              * ket_point.momentum * ket_point.momentum
              * bra_point.wave_function * ket_point.wave_function
              * ProjectKernel(
                    bra, ket, bra_point.momentum, ket_point.momentum,
                    k_mev, angular_quadrature, kernel);
    }
  }
  // Moshinsky brackets relate coordinate-space HO states.  Transforming only
  // the relative factor to momentum space contributes
  //   [(-i)^(2n'+l')]^* (-i)^(2n+l).
  // The (-1)^n pieces are already included in the real radial functions;
  // these are the remaining orbital Fourier phases.  They cancel for the
  // l'=l potentials handled by the legacy translationally invariant path but
  // are essential for a finite-q charge operator.
  const std::complex<double> orbital_fourier_phase =
      std::pow(std::complex<double>(0.0, 1.0), bra.l)
      * std::pow(std::complex<double>(0.0, -1.0), ket.l);
  return orbital_fourier_phase * result;
}

std::complex<double> RelativeHOChargeME(
    int nbra, const PartialWaveChannel& bra,
    int nket, const PartialWaveChannel& ket,
    double hbar_omega_mev, double k_mev, double ge_isoscalar,
    const ChargeDensityParameters& parameters,
    const ProjectionParameters& angular_quadrature,
    const RadialProjectionParameters& radial_quadrature,
    bool include_ope, bool include_contact)
{
  parameters.Validate();
  if (!std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument("G_E^S must be finite");
  }
  return RelativeHOMatrixElement(
      nbra, bra, nket, ket, hbar_omega_mev, k_mev,
      parameters.nucleon_mass_mev, angular_quadrature, radial_quadrature,
      [&](const Vec3& p, const Vec3& pprime, const Vec3& k) {
        return TwoBodyChargeDensity(
            p, pprime, k, ge_isoscalar, parameters,
            include_ope, include_contact);
      });
}

double CenterOfMassHORadial(
    int N, int L, double hbar_omega_mev, double radius_fm,
    double nucleon_mass_mev)
{
  if (N < 0 || L < 0 || !std::isfinite(hbar_omega_mev)
      || hbar_omega_mev <= 0.0 || !std::isfinite(radius_fm)
      || radius_fm < 0.0 || !std::isfinite(nucleon_mass_mev)
      || nucleon_mass_mev <= 0.0)
  {
    throw std::invalid_argument("Invalid pair-CM HO radial-wave-function input");
  }

  const double b_fm = PhysConst::HBARC
      / std::sqrt(2.0 * nucleon_mass_mev * hbar_omega_mev);
  const double x = radius_fm / b_fm;
  const double normalization = 2.0 * std::sqrt(
      gsl_sf_fact(N) * std::pow(2.0, N + L)
      / (std::sqrt(kPi) * gsl_sf_doublefact(2 * N + 2 * L + 1)
         * std::pow(b_fm, 3)));
  return normalization * std::pow(x, L) * std::exp(-0.5 * x * x)
       * gsl_sf_laguerre_n(N, L + 0.5, x * x);
}

std::complex<double> CenterOfMassPlaneWaveME(
    int Nbra, int Lbra, int Mbra,
    int Nket, int Lket, int Mket,
    double hbar_omega_mev, double k_mev, double nucleon_mass_mev,
    const CenterOfMassProjectionParameters& quadrature)
{
  quadrature.Validate();
  if (Nbra < 0 || Lbra < 0 || std::abs(Mbra) > Lbra
      || Nket < 0 || Lket < 0 || std::abs(Mket) > Lket
      || !std::isfinite(hbar_omega_mev) || hbar_omega_mev <= 0.0
      || !std::isfinite(k_mev) || k_mev < 0.0
      || !std::isfinite(nucleon_mass_mev) || nucleon_mass_mev <= 0.0)
  {
    throw std::invalid_argument("Invalid pair-CM plane-wave input");
  }
  if (Mbra != Mket)
  {
    return 0.0;
  }

  gsl_integration_glfixed_table* radial =
      gsl_integration_glfixed_table_alloc(quadrature.radial_order);
  if (radial == nullptr)
  {
    throw std::runtime_error("Unable to allocate pair-CM radial mesh");
  }

  std::complex<double> result = 0.0;
  for (int lambda = std::abs(Lbra - Lket);
       lambda <= Lbra + Lket; ++lambda)
  {
    if ((Lbra + lambda + Lket) % 2 != 0)
    {
      continue;
    }
    const double angular = static_cast<double>(Phase(Mbra))
        * (2.0 * lambda + 1.0)
        * std::sqrt((2.0 * Lbra + 1.0) * (2.0 * Lket + 1.0))
        * AngMom::ThreeJ(Lbra, lambda, Lket, 0, 0, 0)
        * AngMom::ThreeJ(Lbra, lambda, Lket, -Mbra, 0, Mket);
    if (std::abs(angular) < 1.0e-16)
    {
      continue;
    }

    double radial_integral = 0.0;
    for (int ir = 0; ir < quadrature.radial_order; ++ir)
    {
      double radius = 0.0;
      double weight = 0.0;
      gsl_integration_glfixed_point(
          0.0, quadrature.max_radius_fm, ir,
          &radius, &weight, radial);
      const double argument = k_mev * radius / PhysConst::HBARC;
      radial_integral += weight * radius * radius
          * CenterOfMassHORadial(
                Nbra, Lbra, hbar_omega_mev, radius, nucleon_mass_mev)
          * CenterOfMassHORadial(
                Nket, Lket, hbar_omega_mev, radius, nucleon_mass_mev)
          * gsl_sf_bessel_jl(lambda, argument);
    }
    result += std::pow(std::complex<double>(0.0, 1.0), lambda)
            * angular * radial_integral;
  }
  gsl_integration_glfixed_table_free(radial);
  return result;
}

} // namespace chiral_charge
} // namespace imsrg_util

// ============================================================================
// Antisymmetrized J-coupled single-particle TBME construction.
// ============================================================================

#include "AngMom.hh"

#include <gsl/gsl_integration.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace imsrg_util
{
namespace chiral_charge
{
namespace
{

int TBMEPhase(int exponent)
{
  return (std::abs(exponent) % 2 == 0) ? 1 : -1;
}

double AntisymmetryNormalization(int a, int b)
{
  return a == b ? 1.0 / std::sqrt(2.0) : 1.0;
}

double Coupling(double j1, double m1, double j2, double m2,
                double J, double M)
{
  if (std::abs(m1 + m2 - M) > 1.0e-12)
  {
    return 0.0;
  }
  return AngMom::CG(j1, m1, j2, m2, J, M);
}

/// Overlap <[Lambda (l S)j]J | [(Lambda l)L S]J>.  Evaluating the
/// magnetic-quantum-number sum directly avoids importing a six-j phase from
/// a potentially different Moshinsky convention.
double OrbitalSpinRecoupling(
    int Lambda, int l, int L, int S, int jrelative, int J)
{
  const int total_projection = J;
  double result = 0.0;
  for (int mLambda = -Lambda; mLambda <= Lambda; ++mLambda)
  {
    for (int ml = -l; ml <= l; ++ml)
    {
      const int mL = mLambda + ml;
      if (std::abs(mL) > L)
      {
        continue;
      }
      const double first_orbital = Coupling(
          Lambda, mLambda, l, ml, L, mL);
      if (std::abs(first_orbital) < 1.0e-15)
      {
        continue;
      }
      for (int ms = -S; ms <= S; ++ms)
      {
        const int mj = ml + ms;
        if (std::abs(mj) > jrelative)
        {
          continue;
        }
        result += first_orbital
            * Coupling(L, mL, S, ms, J, total_projection)
            * Coupling(l, ml, S, ms, jrelative, mj)
            * Coupling(Lambda, mLambda, jrelative, mj,
                       J, total_projection);
      }
    }
  }
  return result;
}

struct PairComponent
{
  int N;
  int Lambda;
  int mLambda;
  int n;
  int l;
  int S;
  int jrelative;
  int mrelative;
  double amplitude;
};

std::vector<PairComponent> ExpandOrderedPair(
    ModelSpace& modelspace, const Orbit& first, const Orbit& second,
    int J, int M)
{
  std::vector<PairComponent> components;
  const int energy = 2 * first.n + first.l + 2 * second.n + second.l;
  const double jfirst = 0.5 * first.j2;
  const double jsecond = 0.5 * second.j2;

  for (int S = 0; S <= 1; ++S)
  {
    const int Lminimum = std::max(
        std::abs(first.l - second.l), std::abs(J - S));
    const int Lmaximum = std::min(first.l + second.l, J + S);
    for (int L = Lminimum; L <= Lmaximum; ++L)
    {
      const double jj_to_ls = std::sqrt(
          (2.0 * L + 1.0) * (2.0 * S + 1.0)
          * (first.j2 + 1.0) * (second.j2 + 1.0))
          * AngMom::NineJ(first.l, second.l, L,
                          0.5, 0.5, S,
                          jfirst, jsecond, J);
      if (std::abs(jj_to_ls) < 1.0e-15)
      {
        continue;
      }

      for (int N = 0; 2 * N <= energy; ++N)
      {
        for (int n = 0; 2 * (N + n) <= energy; ++n)
        {
          const int remaining = energy - 2 * (N + n);
          for (int Lambda = 0; Lambda <= remaining; ++Lambda)
          {
            const int l = remaining - Lambda;
            if (std::abs(Lambda - l) > L || Lambda + l < L)
            {
              continue;
            }
            const double moshinsky = modelspace.GetMoshinsky(
                N, Lambda, n, l,
                first.n, first.l, second.n, second.l, L);
            if (std::abs(moshinsky) < 1.0e-15)
            {
              continue;
            }

            for (int jrelative = std::abs(l - S);
                 jrelative <= l + S; ++jrelative)
            {
              if (std::abs(Lambda - jrelative) > J
                  || Lambda + jrelative < J)
              {
                continue;
              }
              const double recoupling = OrbitalSpinRecoupling(
                  Lambda, l, L, S, jrelative, J);
              if (std::abs(recoupling) < 1.0e-15)
              {
                continue;
              }
              for (int mLambda = -Lambda;
                   mLambda <= Lambda; ++mLambda)
              {
                const int mrelative = M - mLambda;
                if (std::abs(mrelative) > jrelative)
                {
                  continue;
                }
                const double total_coupling = Coupling(
                    Lambda, mLambda, jrelative, mrelative, J, M);
                if (std::abs(total_coupling) < 1.0e-15)
                {
                  continue;
                }
                components.push_back({
                    N, Lambda, mLambda, n, l, S,
                    jrelative, mrelative,
                    jj_to_ls * moshinsky * recoupling * total_coupling});
              }
            }
          }
        }
      }
    }
  }
  return components;
}

using RelativeKey = std::array<int, 14>;
using RelativeGridKey = std::array<int, 12>;
using CenterOfMassKey = std::array<int, 6>;
using PairExpansionKey = std::array<int, 4>;

class MatrixElementEvaluator
{
 public:
  MatrixElementEvaluator(
      ModelSpace& modelspace, double momentum_transfer_mev,
      double ge_isoscalar,
      const TwoBodyChargeOperatorParameters& parameters)
      : modelspace_(modelspace), momentum_transfer_mev_(momentum_transfer_mev),
        ge_isoscalar_(ge_isoscalar), parameters_(parameters),
        hbar_omega_mev_(modelspace.GetHbarOmega())
  {
    const int order = parameters_.relative_radial_quadrature.radial_order;
    gsl_integration_glfixed_table* radial =
        gsl_integration_glfixed_table_alloc(order);
    if (radial == nullptr)
    {
      throw std::runtime_error("Unable to allocate operator radial mesh");
    }
    radial_momenta_.reserve(order);
    radial_weights_.reserve(order);
    for (int i = 0; i < order; ++i)
    {
      double momentum = 0.0;
      double weight = 0.0;
      gsl_integration_glfixed_point(
          0.0, parameters_.relative_radial_quadrature.max_momentum_mev,
          i, &momentum, &weight, radial);
      radial_momenta_.push_back(momentum);
      radial_weights_.push_back(weight);
    }
    gsl_integration_glfixed_table_free(radial);
  }

  std::complex<double> Direct(
      const Orbit& a, const Orbit& b,
      const Orbit& c, const Orbit& d,
      int J, int M)
  {
    const int MTbra = (a.tz2 + b.tz2) / 2;
    const int MTket = (c.tz2 + d.tz2) / 2;
    if (MTbra != MTket)
    {
      return 0.0;
    }

    const std::vector<PairComponent>& bra = Expansion(a, b, J, M);
    const std::vector<PairComponent>& ket = Expansion(c, d, J, M);
    std::complex<double> result = 0.0;
    for (int T = std::abs(MTbra); T <= 1; ++T)
    {
      const double isospin_bra = Coupling(
          0.5, 0.5 * a.tz2, 0.5, 0.5 * b.tz2, T, MTbra);
      const double isospin_ket = Coupling(
          0.5, 0.5 * c.tz2, 0.5, 0.5 * d.tz2, T, MTket);
      if (std::abs(isospin_bra * isospin_ket) < 1.0e-15)
      {
        continue;
      }

      for (const PairComponent& bra_component : bra)
      {
        for (const PairComponent& ket_component : ket)
        {
          const std::complex<double> center = CenterOfMass(
              bra_component, ket_component);
          if (std::abs(center) < 1.0e-16)
          {
            continue;
          }
          const std::complex<double> relative = Relative(
              bra_component, ket_component, T, MTbra);
          result += isospin_bra * isospin_ket
                  * bra_component.amplitude * ket_component.amplitude
                  * center * relative;
        }
      }
    }
    return result;
  }

 private:
  const std::vector<PairComponent>& Expansion(
      const Orbit& first, const Orbit& second, int J, int M)
  {
    const PairExpansionKey key = {{first.index, second.index, J, M}};
    const auto found = expansion_cache_.find(key);
    if (found != expansion_cache_.end())
    {
      return found->second;
    }
    expansion_cache_[key] = ExpandOrderedPair(
        modelspace_, first, second, J, M);
    return expansion_cache_.at(key);
  }

  std::complex<double> Relative(
      const PairComponent& bra, const PairComponent& ket, int T, int MT)
  {
    const RelativeKey key = {{
        bra.n, bra.l, bra.S, bra.jrelative, bra.mrelative,
        ket.n, ket.l, ket.S, ket.jrelative, ket.mrelative,
        T, MT, parameters_.include_ope ? 1 : 0,
        parameters_.include_contact ? 1 : 0}};
    const auto found = relative_cache_.find(key);
    if (found != relative_cache_.end())
    {
      return found->second;
    }

    const std::vector<std::complex<double>>& grid = RelativeGrid(
        bra, ket, T, MT);
    std::complex<double> value = 0.0;
    const std::size_t order = radial_momenta_.size();
    for (std::size_t ibra = 0; ibra < order; ++ibra)
    {
      const double pprime = radial_momenta_[ibra];
      const double wave_bra = RelativeHOMomentumRadial(
          bra.n, bra.l, hbar_omega_mev_, pprime,
          parameters_.density.nucleon_mass_mev);
      for (std::size_t iket = 0; iket < order; ++iket)
      {
        const double p = radial_momenta_[iket];
        const double wave_ket = RelativeHOMomentumRadial(
            ket.n, ket.l, hbar_omega_mev_, p,
            parameters_.density.nucleon_mass_mev);
        value += radial_weights_[ibra] * radial_weights_[iket]
               * pprime * pprime * p * p * wave_bra * wave_ket
               * grid[ibra * order + iket];
      }
    }
    value *= std::pow(std::complex<double>(0.0, 1.0), bra.l)
           * std::pow(std::complex<double>(0.0, -1.0), ket.l);
    relative_cache_[key] = value;
    return value;
  }

  const std::vector<std::complex<double>>& RelativeGrid(
      const PairComponent& bra, const PairComponent& ket, int T, int MT)
  {
    const RelativeGridKey key = {{
        bra.l, bra.S, bra.jrelative, bra.mrelative,
        ket.l, ket.S, ket.jrelative, ket.mrelative,
        T, MT, parameters_.include_ope ? 1 : 0,
        parameters_.include_contact ? 1 : 0}};
    const auto found = relative_grid_cache_.find(key);
    if (found != relative_grid_cache_.end())
    {
      return found->second;
    }

    PartialWaveChannel bra_channel;
    bra_channel.l = bra.l;
    bra_channel.S = bra.S;
    bra_channel.J = bra.jrelative;
    bra_channel.M = bra.mrelative;
    bra_channel.T = T;
    bra_channel.MT = MT;
    PartialWaveChannel ket_channel;
    ket_channel.l = ket.l;
    ket_channel.S = ket.S;
    ket_channel.J = ket.jrelative;
    ket_channel.M = ket.mrelative;
    ket_channel.T = T;
    ket_channel.MT = MT;
    relative_grid_cache_[key] = ProjectChargeKernelGrid(
        bra_channel, ket_channel, radial_momenta_, radial_momenta_,
        momentum_transfer_mev_, ge_isoscalar_, parameters_.density,
        parameters_.angular_quadrature,
        parameters_.include_ope, parameters_.include_contact);
    return relative_grid_cache_.at(key);
  }

  std::complex<double> CenterOfMass(
      const PairComponent& bra, const PairComponent& ket)
  {
    const CenterOfMassKey key = {{
        bra.N, bra.Lambda, bra.mLambda,
        ket.N, ket.Lambda, ket.mLambda}};
    const auto found = center_of_mass_cache_.find(key);
    if (found != center_of_mass_cache_.end())
    {
      return found->second;
    }
    const std::complex<double> value = CenterOfMassPlaneWaveME(
        bra.N, bra.Lambda, bra.mLambda,
        ket.N, ket.Lambda, ket.mLambda,
        hbar_omega_mev_, momentum_transfer_mev_,
        parameters_.density.nucleon_mass_mev,
        parameters_.center_of_mass_quadrature);
    center_of_mass_cache_[key] = value;
    return value;
  }

  ModelSpace& modelspace_;
  double momentum_transfer_mev_;
  double ge_isoscalar_;
  const TwoBodyChargeOperatorParameters& parameters_;
  double hbar_omega_mev_;
  std::vector<double> radial_momenta_;
  std::vector<double> radial_weights_;
  std::map<RelativeKey, std::complex<double>> relative_cache_;
  std::map<RelativeGridKey, std::vector<std::complex<double>>>
      relative_grid_cache_;
  std::map<CenterOfMassKey, std::complex<double>> center_of_mass_cache_;
  std::map<PairExpansionKey, std::vector<PairComponent>> expansion_cache_;
};

} // namespace

void TwoBodyChargeOperatorParameters::Validate() const
{
  density.Validate();
  angular_quadrature.Validate();
  relative_radial_quadrature.Validate();
  center_of_mass_quadrature.Validate();
  if ((!include_ope && !include_contact)
      || !std::isfinite(imaginary_tolerance)
      || imaginary_tolerance <= 0.0)
  {
    throw std::invalid_argument(
        "Two-body charge operator needs a component and positive tolerance");
  }
}

Operator TwoBodyChargeOperator(
    ModelSpace& modelspace, double momentum_transfer_mev,
    double ge_isoscalar,
    const TwoBodyChargeOperatorParameters& parameters)
{
  parameters.Validate();
  if (!std::isfinite(momentum_transfer_mev)
      || momentum_transfer_mev < 0.0 || !std::isfinite(ge_isoscalar))
  {
    throw std::invalid_argument(
        "Momentum transfer and G_E^S must be finite; momentum is nonnegative");
  }

  Operator result(modelspace, 0, 0, 0, 2);
  result.SetHermitian();
  result.is_reduced = true;
  if (momentum_transfer_mev == 0.0 || ge_isoscalar == 0.0)
  {
    return result;
  }

  modelspace.PreCalculateMoshinsky();
  MatrixElementEvaluator evaluator(
      modelspace, momentum_transfer_mev, ge_isoscalar, parameters);
  for (auto& matrix_entry : result.TwoBody.MatEl)
  {
    const int chbra = matrix_entry.first[0];
    const int chket = matrix_entry.first[1];
    if (chbra != chket)
    {
      continue;
    }
    TwoBodyChannel& channel = modelspace.GetTwoBodyChannel(chbra);
    const int J = channel.J;
    const int number_kets = channel.GetNumberKets();
    for (int ibra = 0; ibra < number_kets; ++ibra)
    {
      Ket& bra = channel.GetKet(ibra);
      const Orbit& a = modelspace.GetOrbit(bra.p);
      const Orbit& b = modelspace.GetOrbit(bra.q);
      for (int iket = ibra; iket < number_kets; ++iket)
      {
        Ket& ket = channel.GetKet(iket);
        const Orbit& c = modelspace.GetOrbit(ket.p);
        const Orbit& d = modelspace.GetOrbit(ket.q);
        std::complex<double> direct = 0.0;
        std::complex<double> exchanged = 0.0;
        for (int M = -J; M <= J; ++M)
        {
          direct += evaluator.Direct(a, b, c, d, J, M);
          exchanged += evaluator.Direct(a, b, d, c, J, M);
        }
        direct /= 2.0 * J + 1.0;
        exchanged /= 2.0 * J + 1.0;
        const int ket_exchange_phase =
            TBMEPhase((c.j2 + d.j2) / 2 - J);
        const std::complex<double> reduced = std::sqrt(2.0 * J + 1.0)
            * AntisymmetryNormalization(c.index, d.index)
            * AntisymmetryNormalization(a.index, b.index)
            * (direct - static_cast<double>(ket_exchange_phase) * exchanged);
        const double scale = std::max(1.0, std::abs(reduced.real()));
        if (std::abs(reduced.imag())
            > parameters.imaginary_tolerance * scale)
        {
          std::ostringstream message;
          message << "Rank-zero two-body charge TBME has a non-negligible "
                  << "imaginary part: channel=" << chbra
                  << " ibra=" << ibra << " iket=" << iket
                  << " value=" << reduced;
          throw std::runtime_error(message.str());
        }
        result.TwoBody.SetTBME(
            chbra, chket, ibra, iket, reduced.real());
      }
    }
  }
  return result;
}

} // namespace chiral_charge
} // namespace imsrg_util
