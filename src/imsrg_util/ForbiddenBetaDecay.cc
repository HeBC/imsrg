// First-forbidden one-body operators, translated from HeBC/kshell's
// interaction.f90, sp_matrix_element.f90, and harmonic_oscillator.f90.
// CS spherical tensors; HO radial functions positive at the origin.
// These are physical reduced matrix elements, without weak couplings,
// initial-spin factors, or the 1/sqrt(2J+1) factor of an OBTD contraction.
#include "imsrg_util.hh"
#include "AngMom.hh"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_laguerre.h>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

namespace imsrg_util {
namespace {
// <la ja || C_1 || lb jb>. j2 variables are twice the angular momentum.
double c1_rme(int la, int ja2, int lb, int jb2)
{
  if ((la+lb+1)%2 || std::abs(la-lb)>1 || la+lb<1 ||
      std::abs(ja2-jb2)>2 || ja2+jb2<2) return 0.0;
  return AngMom::phase(la+lb+1+(ja2+jb2)/2) * std::sqrt(jb2+1.0)
       * AngMom::CG(jb2/2.0,0.5,1.0,0.0,ja2/2.0,0.5);
}

// <l ja || sigma || l jb>; KSHELL s_redmat is half this value.
double sigma_rme(int l, int ja2, int jb2)
{
  return 2.0 * AngMom::phase(l+1+(ja2+1)/2)
       * std::sqrt(1.5*(ja2+1.0)*(jb2+1.0))
       * AngMom::SixJ(0.5,ja2/2.0,l,jb2/2.0,0.5,1.0);
}

// Angular coefficient for [C_1 x sigma]_J, in that order.
double spin_dipole_angular(const Orbit& a, const Orbit& b, int J)
{
  double sum=0.0;
  for (int jt2=std::max(1,2*b.l-1); jt2<=2*b.l+1; jt2+=2)
    sum += AngMom::SixJ(1,1,J,b.j2/2.0,a.j2/2.0,jt2/2.0)
         * c1_rme(a.l,a.j2,b.l,jt2) * sigma_rme(b.l,jt2,b.j2);
  return AngMom::phase(J+(a.j2+b.j2)/2)*std::sqrt(2.0*J+1.0)*sum;
}

// b <a || nabla || b>. This is a real gradient, NOT -i*hbar*nabla.
double gradient_scaled(int na, int la, int ja2, int nb, int lb, int jb2)
{
  double radial=0.0;
  if      (na==nb   && la==lb+1) radial=-std::sqrt((lb+1.0)*(nb+lb+1.5));
  else if (na==nb-1 && la==lb+1) radial=-std::sqrt((lb+1.0)*nb);
  else if (na==nb   && la==lb-1) radial=-std::sqrt(lb*(nb+lb+0.5));
  else if (na==nb+1 && la==lb-1) radial=-std::sqrt(lb*(nb+1.0));
  else return 0.0;
  return radial * AngMom::phase((3+2*la+jb2)/2)
       * std::sqrt((ja2+1.0)*(jb2+1.0))
       * AngMom::SixJ(ja2/2.0,1,jb2/2.0,lb,0.5,la);
}

struct WeightedRadial {
  const Orbit& a;
  const Orbit& b;
  int power;
  double radius_over_b;
  int electron_k;
};

// (2/3) I(k,1,1,1;r) for a uniform sphere, with t=r/R. The k=1
// normalization is the D profile used by KSHELL's primed operators.
double scaled_charge_weight(double t, int electron_k)
{
  if (electron_k==1)
    return (t<=1.0) ? 1.0-t*t/5.0 : (1.0-1.0/(5.0*t*t))/t;
  // Uniform sphere, Eq. (B1) of Stroemberg et al., arXiv:2104.02614.
  const double k=electron_k;
  if (t<=1.0) return 1.0-(2*k+1)/(3*(2*k+3))*t*t;
  const double u=1.0/t;
  double u_to_2k=u*u;
  for (int exponent=2; exponent<2*electron_k; ++exponent) u_to_2k*=u;
  return u*((2*k+1)/(3*k)-u_to_2k/(k*(2*k+3)));
}

double weighted_integrand(double x, void* raw)
{
  const auto& p=*static_cast<WeightedRadial*>(raw);
  if (x==0.0) return 0.0;
  const double log_norm=std::log(2.0)+0.5*(std::lgamma(p.a.n+1.0)+std::lgamma(p.b.n+1.0)
                           -std::lgamma(p.a.n+p.a.l+1.5)-std::lgamma(p.b.n+p.b.l+1.5));
  const double t=x/p.radius_over_b;
  const double weight=scaled_charge_weight(t,p.electron_k);
  return std::exp(log_norm-x*x+(2.0+p.a.l+p.b.l+p.power)*std::log(x))
       * gsl_sf_laguerre_n(p.a.n,p.a.l+0.5,x*x)
       * gsl_sf_laguerre_n(p.b.n,p.b.l+0.5,x*x)*weight;
}

// Integrate separately on either side of the charge-sphere surface. A fixed
// Gauss rule avoids changing GSL's process-wide error handler. Comparing two
// resolutions detects a failure to converge. The HO tail beyond xmax is
// exponentially negligible (at least 12 oscillator lengths past its scale).
double weighted_radial(const Orbit& a, const Orbit& b, int power, double radius_over_b,
                       int electron_k=1)
{
  WeightedRadial data{a,b,power,radius_over_b,electron_k};
  gsl_function f; f.function=&weighted_integrand; f.params=&data;
  const double xmax=std::sqrt(2.0*(a.n+b.n)+a.l+b.l+power+3.0)+12.0;
  const double split=std::min(radius_over_b,xmax);
  auto integrate=[&](size_t points) {
    std::unique_ptr<gsl_integration_glfixed_table,decltype(&gsl_integration_glfixed_table_free)>
      table(gsl_integration_glfixed_table_alloc(points),gsl_integration_glfixed_table_free);
    if (!table) throw std::bad_alloc();
    double value=gsl_integration_glfixed(&f,0.0,split,table.get());
    if (split<xmax) value+=gsl_integration_glfixed(&f,split,xmax,table.get());
    return value;
  };
  const double coarse=integrate(128), fine=integrate(256);
  if (!std::isfinite(coarse) || !std::isfinite(fine) ||
      std::abs(fine-coarse)>1.e-10*(1.0+std::abs(fine)))
    throw std::runtime_error("Forbidden beta decay: weighted radial integral did not converge");
  return fine;
}
// Suhonen Eq. (7.189), with the radial factor evaluated separately.
static double unique_angular(const Orbit& oa, const Orbit& ob, int K)
{
  const int la = oa.l;
  const int lb = ob.l;
  const int ja = oa.j2;
  const int jb = ob.j2;

  // Selection factor: (1 + (-1)^{la+lb+K})
  const int sel = (1 + AngMom::phase(la + lb + K));
  if (sel == 0) return 0.0;

  // Overall phase: (-1)^{la + ja + jb + K + 1}
  const int exp1 = la + (ja + jb) / 2 + K + 1;
  const double ph1 = AngMom::phase(exp1);

  const double pref = 0.5 / std::sqrt((K + 1.0) * (2.0 * K + 3.0));
  const double jhat_a = std::sqrt(ja + 1.0);
  const double jhat_b = std::sqrt(jb + 1.0);

  // 3j: (ja jb K+1; 1/2 -1/2 0)
  const double threej = AngMom::ThreeJ(ja * 0.5, jb * 0.5, K + 1.0, 0.5, -0.5, 0.0);

  // bracket terms in Eq. (7.189)
  const double termA = AngMom::phase(K) * ( ja + 1.0 ); // (-1)^K * jhat_a^2

  const int exp_jsum = (ja + jb) / 2 + 1;
  const double termB = AngMom::phase(exp_jsum) * ( jb + 1.0 ); // (-1)^{ja+jb+1} * jhat_b^2

  // (-1)^{la+ja+K-1/2} is integer exponent (half + half + integer)
  const int expC = la + K + ( ja  - 1 ) /2;
  const double termC_phase = AngMom::phase(expC);

  const double termC = 2.0 * (K + 1.0) * termC_phase;

  const double bracket = termA + termB + termC;

  return pref * ph1 * sel / 2.0 * jhat_a * jhat_b * threej * bracket;
}

double oscillator_length(ModelSpace& ms)
{
  const double hw=ms.GetHbarOmega();
  if (!(hw>0.0) || !std::isfinite(hw))
    throw std::invalid_argument("Forbidden beta decay requires positive finite hbar omega");
  return std::sqrt(PhysConst::HBARC*PhysConst::HBARC/PhysConst::M_NUCLEON/hw);
}

double charge_radius(ModelSpace& ms, double radius_fm)
{
  if (!std::isfinite(radius_fm) || radius_fm<0.0)
    throw std::invalid_argument("Charge radius must be finite and nonnegative");
  if (radius_fm>0.0) return radius_fm;
  const double A=ms.GetTargetMass();
  if (!(A>0.0))
    throw std::invalid_argument("Supply a charge radius or a positive target mass");
  return (1.15+1.8*std::pow(A,-2.0/3.0)-1.2*std::pow(A,-4.0/3.0))*std::cbrt(A);
}

Operator unique_moment(ModelSpace& ms, int K, int N, double radius_over_b, int electron_k=1)
{
  if (K<1 || K==std::numeric_limits<int>::max() ||
      N<0 || N>(std::numeric_limits<int>::max()-K)/2)
    throw std::invalid_argument("UniqueForbidden requires K>=1 and N>=0 with representable K+2N");
  const int power=K+2*N;
  const bool weighted=radius_over_b>0.0;
  Operator op(ms,K+1,1,K%2,2);
  op.SetHermitian(); // Both charge directions, as in the original unique constructor.
  std::map<std::array<int,4>,double> radial_cache;
  for (auto i : ms.all_orbits) {
    const Orbit& a=ms.GetOrbit(i);
    for (auto j : op.OneBodyChannels.at({a.l,a.j2,a.tz2})) {
      const Orbit& b=ms.GetOrbit(j);
      const double angular=unique_angular(a,b,K);
      if (angular==0.0) continue;
      const std::array<int,4> key={a.n,a.l,b.n,b.l};
      auto found=radial_cache.find(key);
      if (found==radial_cache.end()) {
        // Convert the shared (2/3)*I convention to BB's full I.
        const double integral=weighted ? 1.5*weighted_radial(a,b,power,radius_over_b,electron_k)
                                       : RadialIntegral(a.n,a.l,b.n,b.l,power);
        found=radial_cache.emplace(key,integral).first;
      }
      const double value=angular*found->second;
      if (!std::isfinite(value)) throw std::overflow_error("Nonfinite unique-forbidden matrix element");
      if (!weighted && N==0 && std::abs(value)<1.e-5) continue; // Historical cutoff.
      op.OneBody(i,j)=value;
    }
  }
  return op;
}
} // namespace

Operator UniqueForbidden_ChargeExchange_CS(ModelSpace& ms, int K, int N)
{
  return unique_moment(ms,K,N,0.0);
}

Operator UniqueForbidden_I1111_ChargeExchange_CS(ModelSpace& ms, int K, int N, double radius_fm)
{
  return UniqueForbidden_Coulomb_ChargeExchange_CS(ms,K,N,1,radius_fm);
}

Operator UniqueForbidden_Coulomb_ChargeExchange_CS(ModelSpace& ms, int K, int N,
                                                   int electron_k, double radius_fm)
{
  if (electron_k<1 || electron_k>4)
    throw std::invalid_argument("Implemented BB charge weights have electron k=1 through 4");
  const double radius_over_b=charge_radius(ms,radius_fm)/oscillator_length(ms);
  if (!(radius_over_b>0.0) || !std::isfinite(radius_over_b))
    throw std::invalid_argument("BB charge radius is outside the oscillator scale");
  return unique_moment(ms,K,N,radius_over_b,electron_k);
}

Operator FirstForbiddenBetaDecay_Op(ModelSpace& ms, const std::string& name,
                                   int N, const std::string& beta_type, double radius_fm)
{
  static const std::map<std::string,int> ranks={{"0rs",0},{"0sp",0},{"0rsd",0},
    {"1r",1},{"1rs",1},{"1p",1},{"1rd",1},{"1rsd",1},{"2rs",2}};
  auto found=ranks.find(name);
  if (found==ranks.end()) throw std::invalid_argument("Unknown first-forbidden operator: "+name);
  if (N<0 || N>(std::numeric_limits<int>::max()-1)/2)
    throw std::invalid_argument("FirstForbiddenBetaDecay requires a nonnegative radial order N");
  const bool gradient=(name=="0sp" || name=="1p");
  if (gradient && N!=0)
    throw std::invalid_argument("FirstForbiddenBetaDecay gradient operators currently require N=0");
  if (beta_type!="beta-" && beta_type!="beta+" && beta_type!="both")
    throw std::invalid_argument("FirstForbiddenBetaDecay beta type must be beta-, beta+, or both");
  if (!std::isfinite(radius_fm) || radius_fm<0.0)
    throw std::invalid_argument("FirstForbiddenBetaDecay radius must be finite and nonnegative");
  const bool weighted=(name.back()=='d');
  if (weighted) radius_fm=charge_radius(ms,radius_fm);
  const int J=found->second, power=1+2*N;
  const double b=oscillator_length(ms);
  Operator op(ms,J,1,1,2);
  // A single charged component is not Hermitian. 'both' also includes
  // anti-Hermitian spatial tensors, so do not impose an incorrect transpose.
  op.SetNonHermitian();
  std::map<std::array<int,4>,double> radial_cache;
  for (auto i: ms.all_orbits) {
    const Orbit& a=ms.GetOrbit(i);
    if ((beta_type=="beta-" && a.tz2!=-1) || (beta_type=="beta+" && a.tz2!=1)) continue;
    for (auto j: op.OneBodyChannels.at({a.l,a.j2,a.tz2})) {
      const Orbit& c=ms.GetOrbit(j);
      if (a.tz2==c.tz2) continue;
      double value=0.0;
      if (name=="1p") value=gradient_scaled(a.n,a.l,a.j2,c.n,c.l,c.j2)/b;
      else if (name=="0sp") {
        for (int jt2=std::max(1,2*a.l-1); jt2<=2*a.l+1; jt2+=2)
          value+=AngMom::SixJ(1,1,0,c.j2/2.0,a.j2/2.0,jt2/2.0)
               * sigma_rme(a.l,a.j2,jt2)*gradient_scaled(a.n,a.l,jt2,c.n,c.l,c.j2)/b;
        value*=AngMom::phase((a.j2+c.j2)/2);
      } else {
        const double angular=(name=="1r" || name=="1rd") ? c1_rme(a.l,a.j2,c.l,c.j2)
                                                                    : spin_dipole_angular(a,c,J);
        if (angular==0.0) continue;
        const std::array<int,4> key={a.n,a.l,c.n,c.l};
        auto radial=radial_cache.find(key);
        if (radial==radial_cache.end()) {
          const double integral=weighted ? weighted_radial(a,c,power,radius_fm/b)
                                         : RadialIntegral(a.n,a.l,c.n,c.l,power);
          radial=radial_cache.emplace(key,std::pow(b,power)*integral).first;
        }
        value=angular*radial->second;
      }
      if (!std::isfinite(value)) throw std::overflow_error("FirstForbiddenBetaDecay matrix element overflow");
      op.OneBody(i,j)=value;
    }
  }
  return op;
}
} // namespace imsrg_util
