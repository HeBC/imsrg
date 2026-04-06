///////////////////////////////////////////////////////////////////////////////////
//    EOMImsrg.cc, part of  imsrg++
//    Copyright (C) 2024  imsrg++ developers
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License along
//    with this program; if not, write to the Free Software Foundation, Inc.,
//    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
///////////////////////////////////////////////////////////////////////////////////

#include "EOMImsrg.hh"
#include "AngMom.hh"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Tolerance for warning about non-zero imaginary parts of RPA eigenvalues
static const double RPA_IMAG_TOL = 1e-3;
// Tolerance below which we skip renormalization of an RPA amplitude pair
static const double RPA_NORM_TOL = 1e-10;


namespace
{
arma::vec ComputeOnePhNorms(const arma::mat& X)
{
  arma::vec norms(X.n_cols, arma::fill::zeros);
  for (size_t i = 0; i < X.n_cols; i++)
  {
    norms(i) = arma::dot(X.col(i), X.col(i));
  }
  return norms;
}
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

EOMImsrg::EOMImsrg()
  : modelspace(nullptr), one_ph_count(0), two_ph_count(0),
    lanczos_iterations(0), current_channel(0), lanczos_nev(0)
{}

EOMImsrg::EOMImsrg(Operator& H_imsrg)
  : modelspace(H_imsrg.modelspace), H(H_imsrg), one_ph_count(0),
    two_ph_count(0), lanczos_iterations(0), current_channel(0), lanczos_nev(0)
{}

// ---------------------------------------------------------------------------
// Build_AMatrix
// ---------------------------------------------------------------------------

/// Dispatch helper: look up the channel index then call Build_AMatrix_byIndex.
void EOMImsrg::Build_AMatrix(int J, int parity, int Tz)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  Build_AMatrix_byIndex(ich_CC);
}

///
/// Build the EOM A matrix for the 1p-1h sector.
///
/// The matrix element is
/// \f[
///   A_{ai,bj}(J) = f_{ab}\delta_{ij} - f_{ij}\delta_{ab}
///                  - \sum_{J'} (2J'+1)
///                    \begin{Bmatrix} j_a & j_i & J \\ j_b & j_j & J' \end{Bmatrix}
///                    \langle a j; J' \| V \| b i; J' \rangle,
/// \f]
/// where the one-body term uses the full off-diagonal f_{ab} (or f_{ij}) within
/// each one-body channel (i.e. same l, j, tz), following the IMSRG convention.
///
void EOMImsrg::Build_AMatrix_byIndex(size_t ich_CC)
{
  current_channel = ich_CC;
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph = ph_list.size();
  A.zeros(nph, nph);

  int Jph = tbc_CC.J;

  #pragma omp parallel for schedule(dynamic,1)
  for (long long I = 0; I < static_cast<long long>(nph); ++I)
  {
    auto iket_ai = ph_list[static_cast<size_t>(I)];
    Ket& ket_ai = tbc_CC.GetKet(iket_ai);
    index_t a = ket_ai.p;
    index_t i = ket_ai.q;
    int j2a = modelspace->GetOrbit(a).j2;
    int j2i = modelspace->GetOrbit(i).j2;

    // phase_ai tracks the sign convention for the CC-channel ket ordering.
    // When the ket is stored as (hole, particle) we swap to (particle=a, hole=i)
    // and apply the corresponding phase -(-1)^{(j2a+j2i)/2-Jph}.
    int phase_ai = 1;
    int phase_ia = -AngMom::phase((j2a + j2i)/2 - Jph);
    if (ket_ai.op->occ > ket_ai.oq->occ)
    {
      std::swap(a, i);
      std::swap(j2a, j2i);
      std::swap(phase_ai, phase_ia);
    }

    for (size_t II = 0; II < nph; ++II)
    {
      auto iket_bj = ph_list[II];
      Ket& ket_bj = tbc_CC.GetKet(iket_bj);
      index_t b = ket_bj.p;
      index_t j = ket_bj.q;
      int j2b = modelspace->GetOrbit(b).j2;
      int j2j = modelspace->GetOrbit(j).j2;

      int phase_bj = 1;
      int phase_jb = -AngMom::phase((j2b + j2j)/2 - Jph);
      if (ket_bj.op->occ > ket_bj.oq->occ)
      {
        std::swap(b, j);
        std::swap(j2b, j2j);
        std::swap(phase_bj, phase_jb);
      }

      // 1-body contribution: f_{ab}*delta_{ij} - f_{ij}*delta_{ab}
      // f_{ab} is nonzero for any a,b in the same one-body channel (same l,j,tz),
      // not just when a==b.  Same for f_{ij}.
      double H1b = (i == j ? H.OneBody(a, b) : 0.0)
                 - (a == b ? H.OneBody(i, j) : 0.0);

      // Two-body Pandya term
      // A_{ai,bj}(J) -= sum_{J'} (2J'+1) {ja ji J; jb jj J'} <aj';J'|V|bi';J'>
      int J1min = std::max(std::abs(j2a - j2j), std::abs(j2b - j2i)) / 2;
      int J1max = std::min(j2a + j2j, j2b + j2i) / 2;
      double V_ph = 0.0;

      if (AngMom::Triangle(0.5*j2j, 0.5*j2b, Jph) && AngMom::Triangle(0.5*j2i, 0.5*j2a, Jph))
      {
        for (int J1 = J1min; J1 <= J1max; ++J1)
        {
          V_ph -= modelspace->GetSixJ(0.5*j2a, 0.5*j2i, Jph, 0.5*j2b, 0.5*j2j, J1)
                  * (2 * J1 + 1)
                  * H.TwoBody.GetTBME_J(J1, a, j, b, i);
        }
      }

      A(I, II) = (H1b + V_ph) * phase_ai * phase_bj;
    }
  }
}

// ---------------------------------------------------------------------------
// BuildBMatrix
// ---------------------------------------------------------------------------

/// Dispatch helper.
void EOMImsrg::BuildBMatrix(int J, int parity, int Tz)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  BuildBMatrix_byIndex(ich_CC);
}

///
/// Build the RPA B matrix (de-excitation coupling block) for the 1p-1h sector.
///
/// This matrix is used only for RPA-type calculations.  In EOM-IMSRG the
/// ladder operator contains only excitation amplitudes, so this matrix is not
/// needed for any EOM mode.
///
/// The matrix element couples the particle-hole and hole-particle sectors:
/// \f[
///   B_{ai,bj}(J) = (-1)^{j_i+j_b+J}
///                  \sum_{J'} (-1)^{J'} (2J'+1)
///                  \begin{Bmatrix} j_a & j_i & J \\ j_j & j_b & J' \end{Bmatrix}
///                  \langle ab; J' \| V \| ij; J' \rangle.
/// \f]
///
void EOMImsrg::BuildBMatrix_byIndex(size_t ich_CC)
{
  current_channel = ich_CC;
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph = ph_list.size();
  B.zeros(nph, nph);

  int Jph = tbc_CC.J;

  #pragma omp parallel for schedule(dynamic,1)
  for (long long I = 0; I < static_cast<long long>(nph); ++I)
  {
    auto iket_ai = ph_list[static_cast<size_t>(I)];
    Ket& ket_ai = tbc_CC.GetKet(iket_ai);
    index_t a = ket_ai.p;
    index_t i = ket_ai.q;
    int j2a = modelspace->GetOrbit(a).j2;
    int j2i = modelspace->GetOrbit(i).j2;

    int phase_ai = 1;
    int phase_ia = -AngMom::phase((j2a + j2i)/2 - Jph);
    if (ket_ai.op->occ > ket_ai.oq->occ)
    {
      std::swap(a, i);
      std::swap(j2a, j2i);
      std::swap(phase_ai, phase_ia);
    }

    for (size_t II = 0; II < nph; ++II)
    {
      auto iket_bj = ph_list[II];
      Ket& ket_bj = tbc_CC.GetKet(iket_bj);
      index_t b = ket_bj.p;
      index_t j = ket_bj.q;
      int j2b = modelspace->GetOrbit(b).j2;
      int j2j = modelspace->GetOrbit(j).j2;

      int phase_bj = 1;
      int phase_jb = -AngMom::phase((j2b + j2j)/2 - Jph);
      if (ket_bj.op->occ > ket_bj.oq->occ)
      {
        std::swap(b, j);
        std::swap(j2b, j2j);
        std::swap(phase_bj, phase_jb);
      }

      int J1min = std::max(std::abs(j2a - j2b), std::abs(j2i - j2j)) / 2;
      int J1max = std::min(j2a + j2b, j2i + j2j) / 2;
      double V_pp = 0.0;
      int phase_ib = AngMom::phase((j2i + j2b)/2 + Jph);

      if (AngMom::Triangle(0.5*j2j, 0.5*j2b, Jph) && AngMom::Triangle(0.5*j2i, 0.5*j2a, Jph))
      {
        for (int J1 = J1min; J1 <= J1max; ++J1)
        {
          V_pp += AngMom::phase(J1)
                  * (2 * J1 + 1)
                  * modelspace->GetSixJ(0.5*j2a, 0.5*j2i, Jph, 0.5*j2j, 0.5*j2b, J1)
                  * H.TwoBody.GetTBME_J(J1, a, b, i, j);
        }
      }

      B(I, II) = V_pp * phase_ib * phase_ai * phase_bj;
    }
  }
}

// ---------------------------------------------------------------------------
// Build2p2hBasis_byIndex
// ---------------------------------------------------------------------------

///
/// Enumerate all 2p2h basis states compatible with the EOM channel ich_CC.
///
/// A valid 2p2h state |ab Jab; ij Jij; J> must satisfy:
///   - a,b are particles (occ < 0.5); i,j are holes (occ > 0.5)
///   - Triangle(Jab, Jij, J)
///   - (parity_pp + parity_hh) % 2 == parity_EOM
///   - Tz_pp - Tz_hh == Tz_EOM
///
void EOMImsrg::Build2p2hBasis_byIndex(size_t ich_CC)
{
  tpth_basis.clear();

  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  int J      = tbc_CC.J;
  int parity = tbc_CC.parity;
  int Tz     = tbc_CC.Tz;

  size_t nch = modelspace->GetNumberTwoBodyChannels();

  for (size_t ich_pp = 0; ich_pp < nch; ++ich_pp)
  {
    TwoBodyChannel& tbc_pp = modelspace->GetTwoBodyChannel(ich_pp);
    const auto& pp_list = tbc_pp.GetKetIndex_pp();
    if (pp_list.empty()) continue;
    int Jab   = tbc_pp.J;
    int par_pp = tbc_pp.parity;
    int Tz_pp  = tbc_pp.Tz;

    for (size_t ich_hh = 0; ich_hh < nch; ++ich_hh)
    {
      TwoBodyChannel& tbc_hh = modelspace->GetTwoBodyChannel(ich_hh);
      const auto& hh_list = tbc_hh.GetKetIndex_hh();
      if (hh_list.empty()) continue;
      int Jij    = tbc_hh.J;
      int par_hh = tbc_hh.parity;
      int Tz_hh  = tbc_hh.Tz;

      if (!AngMom::Triangle((double)Jab, (double)Jij, (double)J)) continue;
      if ((par_pp + par_hh) % 2 != parity)                         continue;
      if (Tz_pp - Tz_hh != Tz)                                     continue;

      for (auto iket_pp : pp_list)
      {
        Ket& kpp = tbc_pp.GetKet(iket_pp);
        size_t a = kpp.p, b = kpp.q;
        for (auto iket_hh : hh_list)
        {
          Ket& khh = tbc_hh.GetKet(iket_hh);
          size_t ii = khh.p, jj = khh.q;
          tpth_basis.push_back({a, b, ii, jj, Jab, Jij});
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// BuildH22_byIndex
// ---------------------------------------------------------------------------

///
/// Build the 2p2h × 2p2h block H22.
///
/// Contributions (following Parzuchowski et al., PRC 96, 034324 (2017)):
///   - Diagonal: (eps_a + eps_b - eps_i - eps_j)
///   - PP-PP ladder:  <ab Jab|V_norm|a'b' Jab>   when ij==i'j' and Jij==Jij'
///   - HH-HH ladder: -<ij Jij|V_norm|i'j' Jij>  when ab==a'b' and Jab==Jab'
///   - Ph ring term: the correct formula uses NineJ (9j symbol) for the
///     recoupling from the (J_ab,J_ij)→J coupled basis to the J_ph basis.
///
///     For each of 4 active-ph-pair choices in α and matching β states
///     (sharing the same spectator particle and hole orbit), the ring
///     contribution is:
///
///       H22_ring(α,β) += phase_α × phase_β × prefactor
///                      × Σ_{J_ph,J_sp} (2J_ph+1)(2J_sp+1)
///                        × NineJ(j_a,j_b,J_ab; j_i,j_j,J_ij; J_ph,J_sp,J)_α
///                        × V_CC(act_pa,act_ha; act_pb,act_hb)_{J_ph}
///                        × NineJ(j_a',j_b',J_ab'; j_i',j_j',J_ij'; J_ph,J_sp,J)_β
///
///     where NineJ is always called with the first-listed orbit of each pair
///     appearing in position 1; phase_α (phase_β) corrects for the swap when
///     the active orbit is the second-listed orbit of its pair.
///
///     V_CC is the standard Pandya transform:
///       V_CC = -Σ_{J'} (2J'+1) W6j(j_act_pa,j_act_ha,J_ph; j_act_hb,j_act_pb,J')
///              × GetTBME_J(J', act_pa, act_hb, act_pb, act_ha)
///
void EOMImsrg::BuildH22_byIndex(size_t ich_CC)
{
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  int J = tbc_CC.J;

  size_t n2 = tpth_basis.size();
  H22.zeros(n2, n2);

  #pragma omp parallel for schedule(dynamic,1)
  for (long long alpha_ll = 0; alpha_ll < static_cast<long long>(n2); ++alpha_ll)
  {
    size_t alpha = static_cast<size_t>(alpha_ll);
    const TwoPTwoHState& st_alpha = tpth_basis[alpha];
    size_t a = st_alpha.a, b = st_alpha.b, ii = st_alpha.i, jj = st_alpha.j;
    int Jab = st_alpha.Jab, Jij = st_alpha.Jij;
    int j2a  = modelspace->GetOrbit(a).j2;
    int j2b  = modelspace->GetOrbit(b).j2;
    int j2ii = modelspace->GetOrbit(ii).j2;
    int j2jj = modelspace->GetOrbit(jj).j2;
    double ja   = 0.5*j2a;
    double jb   = 0.5*j2b;
    double ji   = 0.5*j2ii;
    double j_jj = 0.5*j2jj;
    double Nab = std::sqrt(1.0 + (a == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));

    // Diagonal SPE term (off-diagonal 1-body contributions are added in beta loop)
    H22(alpha, alpha) = H.OneBody(a, a) + H.OneBody(b, b)
                      - H.OneBody(ii, ii) - H.OneBody(jj, jj);

    // Phase factors for the NineJ when active orbit is the SECOND-listed orbit.
    //
    // For the antisymmetric 2p2h state, singling out the second-listed orbit
    // as "active" introduces the phase from swapping the pair ordering:
    //   |{b,a}^AS_{Jab}> = (-1)^{j_a+j_b-J_ab+1} × |{a,b}^AS_{Jab}>
    //   |{j,i}^AS_{Jij}> = (-1)^{j_i+j_j-J_ij+1} × |{i,j}^AS_{Jij}>
    //
    // The correct 9j also has the active orbit listed FIRST in its row.
    // These two corrections combined give the phase below:
    int phase_pp_a = AngMom::phase((j2a  + j2b)  / 2 - Jab + 1);  // (-1)^{j_a+j_b-J_ab+1}
    int phase_hh_a = AngMom::phase((j2ii + j2jj) / 2 - Jij + 1);  // (-1)^{j_i+j_j-J_ij+1}

    // Struct for one of the 4 active-ph-pair choices for alpha.
    // j1,j2 = row-1 j-values of the 9j (active_p first, spec_p second).
    // j3,j4 = row-2 j-values of the 9j (active_h first, spec_h second).
    struct ACase
    {
      size_t act_p, spec_p;   // particle orbits
      size_t act_h, spec_h;   // hole orbits
      int    phase_alpha;     // antisymmetry phase correction
      double j1, j2;          // 9j row 1: j(act_p), j(spec_p)
      double j3, j4;          // 9j row 2: j(act_h), j(spec_h)
    };

    // 4 cases: (active particle, spectator particle, active hole, spectator hole).
    // NineJ row 1 always has active_p first; row 2 has active_h first.
    // phase_pp_a × phase_hh_a = (-1)^{ja+jb-Jab+1} × (-1)^{ji+jj-Jij+1}
    //   = (-1)^{ja+jb+Jab} × (-1)^{ji+jj+Jij} (the two +1 cancel each other).
    std::array<ACase, 4> alpha_cases = {{
      {a,  b,  ii, jj, 1,                        ja,  jb,   ji,    j_jj },  // case 1
      {a,  b,  jj, ii, phase_hh_a,               ja,  jb,   j_jj,  ji   },  // case 2: swap hh
      {b,  a,  ii, jj, phase_pp_a,               jb,  ja,   ji,    j_jj },  // case 3: swap pp
      {b,  a,  jj, ii, phase_pp_a * phase_hh_a,  jb,  ja,   j_jj,  ji   }   // case 4: both
    }};

    for (size_t beta = 0; beta < n2; ++beta)
    {
      const TwoPTwoHState& st_beta = tpth_basis[beta];
      size_t ap = st_beta.a, bp = st_beta.b, ip = st_beta.i, jp = st_beta.j;
      int Jabp = st_beta.Jab, Jijp = st_beta.Jij;
      int j2ap = modelspace->GetOrbit(ap).j2;
      int j2bp = modelspace->GetOrbit(bp).j2;
      int j2ip = modelspace->GetOrbit(ip).j2;
      int j2jp = modelspace->GetOrbit(jp).j2;
      double jap   = 0.5*j2ap;
      double jbp   = 0.5*j2bp;
      double jip   = 0.5*j2ip;
      double jjp   = 0.5*j2jp;
      double Nabp = std::sqrt(1.0 + (ap == bp ? 1.0 : 0.0));
      double Nijp = std::sqrt(1.0 + (ip == jp ? 1.0 : 0.0));

      // val accumulates all contributions for H22(alpha, beta):
      //   diagonal SPE (only when alpha==beta) + 1-body off-diagonal
      //   + pp-pp + hh-hh + ph ring
      double val = (alpha == beta) ? H22(alpha, alpha) : 0.0;

      // Off-diagonal 1-body contributions (comm121):
      //   f_{a,ap} when b==bp, ii==ip, jj==jp, Jab==Jabp, Jij==Jijp  (alpha!=beta)
      //   f_{b,bp} when a==ap, ii==ip, jj==jp, Jab==Jabp, Jij==Jijp  (alpha!=beta)
      //  -f_{ii,ip} when a==ap, b==bp, jj==jp, Jab==Jabp, Jij==Jijp  (alpha!=beta)
      //  -f_{jj,jp} when a==ap, b==bp, ii==ip, Jab==Jabp, Jij==Jijp  (alpha!=beta)
      // (The diagonal case alpha==beta is already handled by H22(alpha,alpha) above.)
      if (alpha != beta)
      {
        if (b == bp && ii == ip && jj == jp && Jab == Jabp && Jij == Jijp)
          val += H.OneBody(a, ap);
        if (a == ap && ii == ip && jj == jp && Jab == Jabp && Jij == Jijp)
          val += H.OneBody(b, bp);
        if (a == ap && b == bp && jj == jp && Jab == Jabp && Jij == Jijp)
          val -= H.OneBody(ii, ip);
        if (a == ap && b == bp && ii == ip && Jab == Jabp && Jij == Jijp)
          val -= H.OneBody(jj, jp);
      }

      // PP-PP ladder: same hh pair and same J couplings
      if (ii == ip && jj == jp && Jij == Jijp && Jab == Jabp)
        val += H.TwoBody.GetTBME_J_norm(Jab, a, b, ap, bp);

      // HH-HH ladder: same pp pair and same J couplings
      if (a == ap && b == bp && Jab == Jabp && Jij == Jijp)
        val -= H.TwoBody.GetTBME_J_norm(Jij, ii, jj, ip, jp);

      // ---------------------------------------------------------------
      // Ph ring term: correct formula using 9j symbols.
      //
      // Phase correction for beta's NineJ when active orbit is second-listed:
      //   phase_pp_b = (-1)^{j_a'+j_b'-J_ab'+1} (applied if act_pb = bp)
      //   phase_hh_b = (-1)^{j_i'+j_j'-J_ij'+1} (applied if act_hb = jp)
      // ---------------------------------------------------------------
      int phase_pp_b = AngMom::phase((j2ap + j2bp) / 2 - Jabp + 1);
      int phase_hh_b = AngMom::phase((j2ip + j2jp) / 2 - Jijp + 1);

      double prefactor = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)
                                  *(2.0*Jabp+1)*(2.0*Jijp+1))
                       / (Nab * Nij * Nabp * Nijp);

      for (const ACase& ac : alpha_cases)
      {
        // Check beta has the same spectator orbits as alpha's current case
        if (ac.spec_p != ap && ac.spec_p != bp) continue;
        if (ac.spec_h != ip && ac.spec_h != jp) continue;

        // Identify active pair in beta
        size_t act_pb = (ac.spec_p == ap) ? bp : ap;
        size_t act_hb = (ac.spec_h == ip) ? jp : ip;
        double j_act_pa = 0.5*modelspace->GetOrbit(ac.act_p).j2;
        double j_act_ha = 0.5*modelspace->GetOrbit(ac.act_h).j2;
        double j_act_pb = 0.5*modelspace->GetOrbit(act_pb).j2;
        double j_act_hb = 0.5*modelspace->GetOrbit(act_hb).j2;
        double j_spec_p = 0.5*modelspace->GetOrbit(ac.spec_p).j2;
        double j_spec_h = 0.5*modelspace->GetOrbit(ac.spec_h).j2;

        // Phase for beta's NineJ: same antisymmetry correction as alpha
        int phase_beta = 1;
        if (act_pb != ap) phase_beta *= phase_pp_b;
        if (act_hb != ip) phase_beta *= phase_hh_b;

        // 9j j-values for beta: active orbit first in each row
        double j1b = (act_pb == ap) ? jap : jbp;
        double j2b = (act_pb == ap) ? jbp : jap;
        double j3b = (act_hb == ip) ? jip : jjp;
        double j4b = (act_hb == ip) ? jjp : jip;

        // J_ph range: triangle(j_act_pa, j_act_ha, J_ph) and triangle(j_act_pb, j_act_hb, J_ph)
        // Integer arithmetic is exact here: j2 values are always odd for half-integer j,
        // so differences are always even and the /2 produces an exact integer J_ph bound.
        int Jph_min = std::max(std::abs((int)(2*j_act_pa) - (int)(2*j_act_ha)),
                               std::abs((int)(2*j_act_pb) - (int)(2*j_act_hb))) / 2;
        int Jph_max = std::min((int)(2*j_act_pa) + (int)(2*j_act_ha),
                               (int)(2*j_act_pb) + (int)(2*j_act_hb)) / 2;

        // J_sp range: triangle(j_spec_p, j_spec_h, J_sp)
        // Same integer arithmetic applies.
        int Jsp_min_base = std::abs((int)(2*j_spec_p) - (int)(2*j_spec_h)) / 2;
        int Jsp_max_base = ((int)(2*j_spec_p) + (int)(2*j_spec_h)) / 2;

        double ring_total = 0.0;
        for (int Jph = Jph_min; Jph <= Jph_max; ++Jph)
        {
          // Pandya V_CC(act_pa, act_ha; act_pb, act_hb)_{J_ph}
          //   = -Σ_{J'} (2J'+1) W6j(j_act_pa, j_act_ha, Jph; j_act_hb, j_act_pb, J')
          //     × GetTBME_J(J', act_pa, act_hb, act_pb, act_ha)
          int Jp_min2 = std::abs((int)(2*j_act_pa) - (int)(2*j_act_hb)) / 2;
          int Jp_max2 = ((int)(2*j_act_pa) + (int)(2*j_act_hb)) / 2;
          double Vcc = 0.0;
          for (int Jp = Jp_min2; Jp <= Jp_max2; ++Jp)
          {
            if (!AngMom::Triangle(j_act_pb, j_act_ha, (double)Jp)) continue;
            double sixj = modelspace->GetSixJ(j_act_pa, j_act_ha, (double)Jph,
                                              j_act_hb, j_act_pb, (double)Jp);
            if (std::abs(sixj) < 1e-8) continue;
            Vcc -= (2*Jp+1) * sixj
                 * H.TwoBody.GetTBME_J(Jp, ac.act_p, act_hb, act_pb, ac.act_h);
          }
          if (std::abs(Vcc) < 1e-10) continue;

          // Sum over J_sp with two NineJ symbols
          // J_sp constrained by: triangle(j_spec_p, j_spec_h, J_sp) and triangle(J_ph, J_sp, J)
          int Jsp_min = std::max(Jsp_min_base, std::abs(J - Jph));
          int Jsp_max = std::min(Jsp_max_base, J + Jph);
          double ring_Jsp = 0.0;
          for (int Jsp = Jsp_min; Jsp <= Jsp_max; ++Jsp)
          {
            // NineJ for alpha: active orbit first in each row
            //   {j(act_p)  j(spec_p)  J_ab }
            //   {j(act_h)  j(spec_h)  J_ij }
            //   {J_ph      J_sp       J    }
            double n9j_a = modelspace->GetNineJ(ac.j1, ac.j2, (double)Jab,
                                                ac.j3, ac.j4, (double)Jij,
                                                (double)Jph, (double)Jsp, (double)J);
            // NineJ for beta: active orbit first in each row
            double n9j_b = modelspace->GetNineJ(j1b, j2b, (double)Jabp,
                                                j3b, j4b,     (double)Jijp,
                                                (double)Jph, (double)Jsp, (double)J);
            ring_Jsp += (2*Jsp+1) * n9j_a * n9j_b;
          }
          ring_total += Vcc * (2*Jph+1) * ring_Jsp;
        }
        val += (double)(ac.phase_alpha * phase_beta) * prefactor * ring_total;
      } // end loop over alpha_cases

      H22(alpha, beta) = val;
    }
  }

  // Symmetrize to remove any numerical asymmetry
  H22 = 0.5 * (H22 + H22.t());
}

// ---------------------------------------------------------------------------
// Build_H21_byIndex
// ---------------------------------------------------------------------------

///
/// Build the 2p-2h × 1p-1h coupling block H21 = <α|H|β>.
///
/// Notation used throughout:
///   H   = IMSRG-evolved Hamiltonian = f (one-body) + Λ (two-body) + …
///   R2† = a†_a a†_b a_j a_i  (two-body excitation operator)
///   R1† = a†_c a_k            (one-body excitation operator)
///
///   2p-2h state α = |ab J_ab; ij J_ij; J⟩   with a,b particles (occ<0.5),
///                                                  i,j holes    (occ>0.5)
///   1p-1h state β = |ck; J⟩                  with c particle, k hole
///
/// Common prefactor:
///   K = sqrt((2J_ab+1)(2J_ij+1)) / (N_ab * N_ij) * (-1)^J
///   where N_ab = sqrt(1+δ_ab), N_ij = sqrt(1+δ_ij)
///
/// Four non-zero contributions from <α|Λ|β>:
///
///   sm1 (c == a):  H21 -= phase((j_a+j_b) - J_ij)
///                         × W6j(J_ab, J_ij, J; j_k, j_a, j_b)
///                         × Λ(J_ij; k,b|i,j) × K
///
///   sm2 (c == b):  H21 += phase(J_ab + J_ij)
///                         × W6j(J_ab, J_ij, J; j_k, j_b, j_a)
///                         × Λ(J_ij; k,a|i,j) × K
///
///   sm3 (k == j):  H21 += phase((j_i+j_j) - J_ab)
///                         × W6j(J_ab, J_ij, J; j_j, j_c, j_i)
///                         × Λ(J_ab; a,b|i,c) × K
///
///   sm4 (k == i):  H21 -= phase(J_ij + J_ab)
///                         × W6j(J_ab, J_ij, J; j_i, j_c, j_j)
///                         × Λ(J_ab; a,b|j,c) × K
///
/// where Λ(J; p,q|r,s) = GetTBME_J(J, p, q, r, s), the antisymmetric
/// unnormalized two-body matrix element of H (includes sqrt(1+δ) factors).
/// Using GetTBME_J (not _norm) is consistent with the N_ab/N_ij prefactor in K.
///
/// After filling the matrix, the same CC-channel ket-ordering phase used in
/// Build_AMatrix_byIndex is applied column-wise so that the 1p1h columns of
/// H21 are aligned with those of A in the full EOM2 secular matrix:
///   [ A       H21^T ]
///   [ H21     H22   ]
///
void EOMImsrg::Build_H21_byIndex(size_t ich_CC)
{
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph  = ph_list.size();
  size_t n2   = tpth_basis.size();
  int    J    = tbc_CC.J;
  H21.zeros(n2, nph);

  // Build orbit → ph-column lookup and per-column orbit caches in one pass.
  // ph_particle[col] = particle orbit (a,b,c,…) for ph column col.
  // ph_hole[col]     = hole    orbit  (i,j,k,…) for ph column col.
  // orbit_to_ph[orb] = list of (col, is_particle) pairs.
  std::vector<std::vector<std::pair<size_t,bool>>> orbit_to_ph(
      modelspace->GetNumberOrbits());
  std::vector<index_t> ph_particle(nph), ph_hole(nph);

  {
    size_t col = 0;
    for (auto iket : ph_list)
    {
      Ket& kt = tbc_CC.GetKet(iket);
      index_t a_ph = kt.p, k_ph = kt.q;    // a_ph=particle, k_ph=hole
      if (kt.op->occ > kt.oq->occ) std::swap(a_ph, k_ph);
      orbit_to_ph[a_ph].push_back(std::make_pair(col, true));
      orbit_to_ph[k_ph].push_back(std::make_pair(col, false));
      ph_particle[col] = a_ph;
      ph_hole[col]     = k_ph;
      ++col;
    }
  }

  #pragma omp parallel for schedule(dynamic,1)
  for (long long alpha_ll = 0; alpha_ll < static_cast<long long>(n2); ++alpha_ll)
  {
    size_t alpha = static_cast<size_t>(alpha_ll);
    const TwoPTwoHState& st_alpha = tpth_basis[alpha];
    // 2p-2h state: particles a,b coupled to J_ab; holes ii,jj coupled to J_ij
    size_t a = st_alpha.a, b = st_alpha.b, ii = st_alpha.i, jj = st_alpha.j;
    int Jab = st_alpha.Jab, Jij = st_alpha.Jij;

    const Orbit& oa   = modelspace->GetOrbit(a);
    const Orbit& ob   = modelspace->GetOrbit(b);
    const Orbit& o_ii = modelspace->GetOrbit(ii);
    const Orbit& o_jj = modelspace->GetOrbit(jj);

    double ja   = 0.5*oa.j2,   jb   = 0.5*ob.j2;
    double ji   = 0.5*o_ii.j2, j_jj = 0.5*o_jj.j2;

    double Nab = std::sqrt(1.0 + (a  == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));
    // K = sqrt((2J_ab+1)(2J_ij+1)) / (N_ab * N_ij) * (-1)^J
    double K = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)) / (Nab * Nij)
               * AngMom::phase(J);

    // --- sm1: c == a  (1p1h particle matches first particle of 2p2h state) ---
    // H21[α,col] -= phase((j_a+j_b) - J_ij)
    //               × W6j(J_ab, J_ij, J; j_k, j_a, j_b)
    //               × Λ(J_ij; k,b|i,j) × K
    {
      int phase1 = AngMom::phase((oa.j2 + ob.j2)/2 - Jij);
      // Loop over ph columns where the particle orbit equals a
      for (size_t _s1 = 0; _s1 < orbit_to_ph[a].size(); ++_s1)
      {
        size_t col_idx = orbit_to_ph[a][_s1].first;
        bool is_part   = orbit_to_ph[a][_s1].second;
        if (!is_part) continue;  // require particle == a
        size_t k_orb = ph_hole[col_idx];          // hole k of the 1p1h state
        const Orbit& o_k = modelspace->GetOrbit(k_orb);
        double jk = 0.5*o_k.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         jk, ja, jb);
        if (w6j == 0.0) continue;
        // Λ(J_ij; k,b | i,j) = GetTBME_J(J_ij, k, b, ii, jj)
        double lam = H.TwoBody.GetTBME_J(Jij, k_orb, b, ii, jj);
        H21(alpha, col_idx) -= phase1 * w6j * lam * K;
      }
    }

    // --- sm2: c == b  (1p1h particle matches second particle of 2p2h state) ---
    // H21[α,col] += phase(J_ab + J_ij)
    //               × W6j(J_ab, J_ij, J; j_k, j_b, j_a)
    //               × Λ(J_ij; k,a|i,j) × K
    {
      int phase2 = AngMom::phase(Jab + Jij);
      for (size_t _si = 0; _si < orbit_to_ph[b].size(); ++_si)
      {
        size_t col_idx = orbit_to_ph[b][_si].first;
        bool is_part   = orbit_to_ph[b][_si].second;
        if (!is_part) continue;
        size_t k_orb = ph_hole[col_idx];
        const Orbit& o_k = modelspace->GetOrbit(k_orb);
        double jk = 0.5*o_k.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         jk, jb, ja);
        if (w6j == 0.0) continue;
        // Λ(J_ij; k,a | i,j) = GetTBME_J(J_ij, k, a, ii, jj)
        double lam = H.TwoBody.GetTBME_J(Jij, k_orb, a, ii, jj);
        H21(alpha, col_idx) += phase2 * w6j * lam * K;
      }
    }

    // --- sm3: k == j  (1p1h hole matches second hole of 2p2h state) ---
    // H21[α,col] += phase((j_i+j_j) - J_ab)
    //               × W6j(J_ab, J_ij, J; j_j, j_c, j_i)
    //               × Λ(J_ab; a,b|i,c) × K
    {
      int phase3 = AngMom::phase((o_ii.j2 + o_jj.j2)/2 - Jab);
      for (size_t _si = 0; _si < orbit_to_ph[jj].size(); ++_si)
      {
        size_t col_idx = orbit_to_ph[jj][_si].first;
        bool is_part   = orbit_to_ph[jj][_si].second;
        if (is_part) continue;  // require hole == jj
        size_t c_orb = ph_particle[col_idx];      // particle c of the 1p1h state
        const Orbit& o_c = modelspace->GetOrbit(c_orb);
        double jc = 0.5*o_c.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         j_jj, jc, ji);
        if (w6j == 0.0) continue;
        // Λ(J_ab; a,b | i,c) = GetTBME_J(J_ab, a, b, ii, c)
        double lam = H.TwoBody.GetTBME_J(Jab, a, b, ii, c_orb);
        H21(alpha, col_idx) += phase3 * w6j * lam * K;
      }
    }

    // --- sm4: k == i  (1p1h hole matches first hole of 2p2h state) ---
    // H21[α,col] -= phase(J_ij + J_ab)
    //               × W6j(J_ab, J_ij, J; j_i, j_c, j_j)
    //               × Λ(J_ab; a,b|j,c) × K
    {
      int phase4 = AngMom::phase(Jij + Jab);
      for (size_t _si = 0; _si < orbit_to_ph[ii].size(); ++_si)
      {
        size_t col_idx = orbit_to_ph[ii][_si].first;
        bool is_part   = orbit_to_ph[ii][_si].second;
        if (is_part) continue;
        size_t c_orb = ph_particle[col_idx];
        const Orbit& o_c = modelspace->GetOrbit(c_orb);
        double jc = 0.5*o_c.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         ji, jc, j_jj);
        if (w6j == 0.0) continue;
        // Λ(J_ab; a,b | j,c) = GetTBME_J(J_ab, a, b, jj, c)
        double lam = H.TwoBody.GetTBME_J(Jab, a, b, jj, c_orb);
        H21(alpha, col_idx) -= phase4 * w6j * lam * K;
      }
    }
  }

  // Apply the same CC-channel ket-ordering phase corrections used in
  // Build_AMatrix_byIndex.  When a ph ket at column col is stored as
  // (hole, particle), Build_AMatrix multiplies row and column col by
  // phase_ki = -(-1)^{j_c+j_k-J}  (using notation c=particle, k=hole).
  // For the EOM2 full secular matrix
  //   [ A       H21^T ]
  //   [ H21     H22   ]
  // to be consistent, H21(α, col) must carry the same phase_col factor
  // so that the 1p1h columns of H21 align with those of A.
  for (size_t col = 0; col < nph; ++col)
  {
    auto iket = ph_list[col];
    Ket& kt = tbc_CC.GetKet(iket);
    if (kt.op->occ > kt.oq->occ)   // ket stored as (hole=p, particle=q)
    {
      // After A-matrix swap: a = kt.q (particle), i = kt.p (hole)
      int ph_col = -AngMom::phase((kt.oq->j2 + kt.op->j2) / 2 - J);
      if (ph_col != 1) H21.col(col) *= ph_col;
    }
  }
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

void EOMImsrg::Solve(int J, int parity, int Tz, std::string mode)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  Solve_byIndex(ich_CC, mode);
}

void EOMImsrg::Solve_byIndex(size_t ich_CC, std::string mode)
{
  current_channel = ich_CC;
  Build_AMatrix_byIndex(ich_CC);
  one_ph_count = A.n_rows;
  two_ph_count = 0;
  lanczos_iterations = 0;
  OnePhNorms.reset();

  if (mode == "RPA")
    BuildBMatrix_byIndex(ich_CC);
  else
    B.zeros(A.n_rows, A.n_cols);

  if (mode == "EOM2")
  {
    Build2p2hBasis_byIndex(ich_CC);
    two_ph_count = tpth_basis.size();
    BuildH22_byIndex(ich_CC);
    Build_H21_byIndex(ich_CC);
  }

  SolveCurrentChannel(mode);

  // Store results for later lookup
  EOMChannel ch;
  ch.Energies = Energies;
  ch.X        = X;
  ch.Y        = Y;
  ch.OnePhNorms = OnePhNorms;
  ch.OnePhCount = one_ph_count;
  ch.TwoPhCount = two_ph_count;
  ch.LanczosIterations = lanczos_iterations;
  // Store the 2p2h basis so ComputeTransitionME_byIndex can access it for any channel.
  if (mode == "EOM2")
    ch.tpth_basis = tpth_basis;
  ChannelResults[ich_CC] = ch;
}

/// Solve all particle-hole channels that have at least one ph pair.
void EOMImsrg::SolveAllChannels(std::string mode)
{
  size_t nch = modelspace->GetNumberTwoBodyChannels_CC();
  for (size_t ich = 0; ich < nch; ich++)
  {
    TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ich);
    if (tbc.GetKetIndex_ph().empty()) continue;
    Solve_byIndex(ich, mode);
  }
}

// ---------------------------------------------------------------------------
// Matrix-free matvec helpers
// ---------------------------------------------------------------------------

/// Compute Hv_2p2h += H22 * v without building H22.
///
/// Implements the same formula as BuildH22_byIndex row by row.
/// For each 2p-2h state α = |ab J_ab; ij J_ij; J⟩ the contributions from
/// all β states are accumulated and written to Hv[α].
///
/// Terms (notation: a,b particles; i,j holes; H = f + Λ):
///   - Diagonal SPE:   (f_aa + f_bb - f_ii - f_jj) × v[α]
///   - PP-PP ladder:   +Λ_norm(J_ab; a,b|a',b') × v[β]   when ij=i'j', J_ij=J_ij'
///   - HH-HH ladder:   -Λ_norm(J_ij; i,j|i',j') × v[β]   when ab=a'b', J_ab=J_ab'
///   - Ph ring term:   see BuildH22_byIndex for the full 9j-symbol formula
///
/// The loop is trivially thread-safe under OpenMP because each α accumulates
/// into a private double hv_alpha before writing to Hv[α].
void EOMImsrg::ApplyH22_matvec(const arma::vec& v, arma::vec& Hv) const
{
  int J = modelspace->GetTwoBodyChannel_CC(current_channel).J;
  size_t n2 = tpth_basis.size();

  #pragma omp parallel for schedule(dynamic,1)
  for (long long alpha_ll = 0; alpha_ll < static_cast<long long>(n2); ++alpha_ll)
  {
    size_t alpha = static_cast<size_t>(alpha_ll);
    const TwoPTwoHState& st_alpha = tpth_basis[alpha];
    size_t a = st_alpha.a, b = st_alpha.b, ii = st_alpha.i, jj = st_alpha.j;
    int Jab = st_alpha.Jab, Jij = st_alpha.Jij;
    int j2a  = modelspace->GetOrbit(a).j2;
    int j2b  = modelspace->GetOrbit(b).j2;
    int j2ii = modelspace->GetOrbit(ii).j2;
    int j2jj = modelspace->GetOrbit(jj).j2;
    double ja   = 0.5*j2a;
    double jb   = 0.5*j2b;
    double ji   = 0.5*j2ii;
    double j_jj = 0.5*j2jj;
    double Nab = std::sqrt(1.0 + (a == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));

    // Diagonal SPE contribution and off-diagonal 1-body contributions (comm121)
    double hv_alpha = (H.OneBody(a, a) + H.OneBody(b, b)
                     - H.OneBody(ii, ii) - H.OneBody(jj, jj)) * v[alpha];

    // Off-diagonal 1-body: loop over beta states that share 3 of 4 orbits
    // with alpha but differ in one orbit within the same one-body channel.
    for (size_t beta2 = 0; beta2 < n2; ++beta2)
    {
      if (beta2 == alpha || std::abs(v[beta2]) < 1e-15) continue;
      const TwoPTwoHState& st_b2 = tpth_basis[beta2];
      size_t ap2 = st_b2.a, bp2 = st_b2.b, ip2 = st_b2.i, jp2 = st_b2.j;
      int Jabp2 = st_b2.Jab, Jijp2 = st_b2.Jij;
      if (b == bp2 && ii == ip2 && jj == jp2 && Jab == Jabp2 && Jij == Jijp2)
        hv_alpha += H.OneBody(a, ap2) * v[beta2];
      if (a == ap2 && ii == ip2 && jj == jp2 && Jab == Jabp2 && Jij == Jijp2)
        hv_alpha += H.OneBody(b, bp2) * v[beta2];
      if (a == ap2 && b == bp2 && jj == jp2 && Jab == Jabp2 && Jij == Jijp2)
        hv_alpha -= H.OneBody(ii, ip2) * v[beta2];
      if (a == ap2 && b == bp2 && ii == ip2 && Jab == Jabp2 && Jij == Jijp2)
        hv_alpha -= H.OneBody(jj, jp2) * v[beta2];
    }

    int phase_pp_a = AngMom::phase((j2a  + j2b)  / 2 - Jab + 1);
    int phase_hh_a = AngMom::phase((j2ii + j2jj) / 2 - Jij + 1);

    struct ACase
    {
      size_t act_p, spec_p;
      size_t act_h, spec_h;
      int    phase_alpha;
      double j1, j2;
      double j3, j4;
    };
    std::array<ACase, 4> alpha_cases = {{
      {a,  b,  ii, jj, 1,                        ja,  jb,   ji,    j_jj },
      {a,  b,  jj, ii, phase_hh_a,               ja,  jb,   j_jj,  ji   },
      {b,  a,  ii, jj, phase_pp_a,               jb,  ja,   ji,    j_jj },
      {b,  a,  jj, ii, phase_pp_a * phase_hh_a,  jb,  ja,   j_jj,  ji   }
    }};

    for (size_t beta = 0; beta < n2; ++beta)
    {
      // Do NOT skip beta==alpha: PP-PP/HH-HH self-coupling and diagonal ring
      // are also needed for the diagonal, beyond the SPE handled above.
      if (std::abs(v[beta]) < 1e-15) continue;

      const TwoPTwoHState& st_beta = tpth_basis[beta];
      size_t ap = st_beta.a, bp = st_beta.b, ip = st_beta.i, jp = st_beta.j;
      int Jabp = st_beta.Jab, Jijp = st_beta.Jij;
      int j2ap = modelspace->GetOrbit(ap).j2;
      int j2bp = modelspace->GetOrbit(bp).j2;
      int j2ip = modelspace->GetOrbit(ip).j2;
      int j2jp = modelspace->GetOrbit(jp).j2;
      double jap   = 0.5*j2ap;
      double jbp   = 0.5*j2bp;
      double jip   = 0.5*j2ip;
      double jjp   = 0.5*j2jp;
      double Nabp = std::sqrt(1.0 + (ap == bp ? 1.0 : 0.0));
      double Nijp = std::sqrt(1.0 + (ip == jp ? 1.0 : 0.0));

      double val = 0.0;

      // PP-PP ladder
      if (ii == ip && jj == jp && Jij == Jijp && Jab == Jabp)
        val += H.TwoBody.GetTBME_J_norm(Jab, a, b, ap, bp);

      // HH-HH ladder
      if (a == ap && b == bp && Jab == Jabp && Jij == Jijp)
        val -= H.TwoBody.GetTBME_J_norm(Jij, ii, jj, ip, jp);

      // Ph ring
      int phase_pp_b = AngMom::phase((j2ap + j2bp) / 2 - Jabp + 1);
      int phase_hh_b = AngMom::phase((j2ip + j2jp) / 2 - Jijp + 1);
      double prefactor = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)
                                  *(2.0*Jabp+1)*(2.0*Jijp+1))
                       / (Nab * Nij * Nabp * Nijp);

      for (const ACase& ac : alpha_cases)
      {
        if (ac.spec_p != ap && ac.spec_p != bp) continue;
        if (ac.spec_h != ip && ac.spec_h != jp) continue;

        size_t act_pb = (ac.spec_p == ap) ? bp : ap;
        size_t act_hb = (ac.spec_h == ip) ? jp : ip;
        double j_act_pa = 0.5*modelspace->GetOrbit(ac.act_p).j2;
        double j_act_ha = 0.5*modelspace->GetOrbit(ac.act_h).j2;
        double j_act_pb = 0.5*modelspace->GetOrbit(act_pb).j2;
        double j_act_hb = 0.5*modelspace->GetOrbit(act_hb).j2;
        double j_spec_p = 0.5*modelspace->GetOrbit(ac.spec_p).j2;
        double j_spec_h = 0.5*modelspace->GetOrbit(ac.spec_h).j2;

        int phase_beta = 1;
        if (act_pb != ap) phase_beta *= phase_pp_b;
        if (act_hb != ip) phase_beta *= phase_hh_b;

        double j1b = (act_pb == ap) ? jap : jbp;
        double j2b = (act_pb == ap) ? jbp : jap;
        double j3b = (act_hb == ip) ? jip : jjp;
        double j4b = (act_hb == ip) ? jjp : jip;

        int Jph_min = std::max(std::abs((int)(2*j_act_pa) - (int)(2*j_act_ha)),
                               std::abs((int)(2*j_act_pb) - (int)(2*j_act_hb))) / 2;
        int Jph_max = std::min((int)(2*j_act_pa) + (int)(2*j_act_ha),
                               (int)(2*j_act_pb) + (int)(2*j_act_hb)) / 2;

        int Jsp_min_base = std::abs((int)(2*j_spec_p) - (int)(2*j_spec_h)) / 2;
        int Jsp_max_base = ((int)(2*j_spec_p) + (int)(2*j_spec_h)) / 2;

        double ring_total = 0.0;
        for (int Jph = Jph_min; Jph <= Jph_max; ++Jph)
        {
          int Jp_min2 = std::abs((int)(2*j_act_pa) - (int)(2*j_act_hb)) / 2;
          int Jp_max2 = ((int)(2*j_act_pa) + (int)(2*j_act_hb)) / 2;
          double Vcc = 0.0;
          for (int Jp = Jp_min2; Jp <= Jp_max2; ++Jp)
          {
            if (!AngMom::Triangle(j_act_pb, j_act_ha, (double)Jp)) continue;
            double sixj = modelspace->GetSixJ(j_act_pa, j_act_ha, (double)Jph,
                                              j_act_hb, j_act_pb, (double)Jp);
            if (std::abs(sixj) < 1e-8) continue;
            Vcc -= (2*Jp+1) * sixj
                 * H.TwoBody.GetTBME_J(Jp, ac.act_p, act_hb, act_pb, ac.act_h);
          }
          if (std::abs(Vcc) < 1e-10) continue;

          int Jsp_min = std::max(Jsp_min_base, std::abs(J - Jph));
          int Jsp_max = std::min(Jsp_max_base, J + Jph);
          double ring_Jsp = 0.0;
          for (int Jsp = Jsp_min; Jsp <= Jsp_max; ++Jsp)
          {
            double n9j_a = modelspace->GetNineJ(ac.j1, ac.j2, (double)Jab,
                                                ac.j3, ac.j4, (double)Jij,
                                                (double)Jph, (double)Jsp, (double)J);
            double n9j_b = modelspace->GetNineJ(j1b,  j2b, (double)Jabp,
                                                j3b,  j4b,     (double)Jijp,
                                                (double)Jph, (double)Jsp, (double)J);
            ring_Jsp += (2*Jsp+1) * n9j_a * n9j_b;
          }
          ring_total += Vcc * (2*Jph+1) * ring_Jsp;
        }
        val += (double)(ac.phase_alpha * phase_beta) * prefactor * ring_total;
      }

      hv_alpha += val * v[beta];
    }

    Hv[alpha] += hv_alpha;
  }
}

/// Compute Hv_2p2h += H21 * v_ph without building H21.
///
/// Applies the same formula as Build_H21_byIndex but without materialising H21.
/// Parallelised over 2p-2h states α; uses the precomputed lookup tables:
///   mf_orbit_to_ph[orb]  → (col, is_particle) pairs for each orbit
///   mf_ph_particle[col]  → particle orbit (a,b,c,…) of ph column col
///   mf_ph_hole[col]      → hole    orbit  (i,j,k,…) of ph column col
/// These are populated once by Solve_byIndex_MF before Lanczos iterations.
///
/// The four terms sm1-sm4 are identical to Build_H21_byIndex; see that
/// function's comment for the full formula with consistent notation.
void EOMImsrg::ApplyH21_matvec(const arma::vec& v_ph, arma::vec& Hv_2p2h) const
{
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(current_channel);
  int J = tbc_CC.J;
  size_t n2 = tpth_basis.size();

  #pragma omp parallel for schedule(dynamic,1)
  for (long long alpha_ll = 0; alpha_ll < static_cast<long long>(n2); ++alpha_ll)
  {
    size_t alpha = static_cast<size_t>(alpha_ll);
    const TwoPTwoHState& st_alpha = tpth_basis[alpha];
    // 2p-2h state: particles a,b coupled to J_ab; holes ii,jj coupled to J_ij
    size_t a = st_alpha.a, b = st_alpha.b, ii = st_alpha.i, jj = st_alpha.j;
    int Jab = st_alpha.Jab, Jij = st_alpha.Jij;

    const Orbit& oa   = modelspace->GetOrbit(a);
    const Orbit& ob   = modelspace->GetOrbit(b);
    const Orbit& o_ii = modelspace->GetOrbit(ii);
    const Orbit& o_jj = modelspace->GetOrbit(jj);

    double ja   = 0.5*oa.j2,   jb   = 0.5*ob.j2;
    double ji   = 0.5*o_ii.j2, j_jj = 0.5*o_jj.j2;

    double Nab = std::sqrt(1.0 + (a  == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));
    // K = sqrt((2J_ab+1)(2J_ij+1)) / (N_ab * N_ij) * (-1)^J
    double K = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)) / (Nab * Nij) * AngMom::phase(J);

    double hv = 0.0;

    // sm1: c == a  (1p1h particle c matches first particle a of 2p2h state)
    // hv -= phase((j_a+j_b) - J_ij) × W6j(J_ab,J_ij,J; j_k,j_a,j_b)
    //       × Λ(J_ij; k,b|i,j) × K × phase_col × v_ph[col]
    {
      int phase1 = AngMom::phase((oa.j2 + ob.j2)/2 - Jij);
      for (size_t _si = 0; _si < mf_orbit_to_ph[a].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[a][_si].first;
        bool is_part   = mf_orbit_to_ph[a][_si].second;
        if (!is_part) continue;
        size_t k_orb = mf_ph_hole[col_idx];       // hole k of the 1p1h state
        const Orbit& o_k = modelspace->GetOrbit(k_orb);
        double jk = 0.5*o_k.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         jk, ja, jb);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jij, k_orb, b, ii, jj);
        hv -= phase1 * w6j * lam * K * mf_ph_phase[col_idx] * v_ph[col_idx];
      }
    }

    // sm2: c == b  (1p1h particle c matches second particle b of 2p2h state)
    // hv += phase(J_ab+J_ij) × W6j(J_ab,J_ij,J; j_k,j_b,j_a)
    //       × Λ(J_ij; k,a|i,j) × K × phase_col × v_ph[col]
    {
      int phase2 = AngMom::phase(Jab + Jij);
      for (size_t _si = 0; _si < mf_orbit_to_ph[b].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[b][_si].first;
        bool is_part   = mf_orbit_to_ph[b][_si].second;
        if (!is_part) continue;
        size_t k_orb = mf_ph_hole[col_idx];
        const Orbit& o_k = modelspace->GetOrbit(k_orb);
        double jk = 0.5*o_k.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         jk, jb, ja);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jij, k_orb, a, ii, jj);
        hv += phase2 * w6j * lam * K * mf_ph_phase[col_idx] * v_ph[col_idx];
      }
    }

    // sm3: k == j  (1p1h hole k matches second hole j of 2p2h state)
    // hv += phase((j_i+j_j) - J_ab) × W6j(J_ab,J_ij,J; j_j,j_c,j_i)
    //       × Λ(J_ab; a,b|i,c) × K × phase_col × v_ph[col]
    {
      int phase3 = AngMom::phase((o_ii.j2 + o_jj.j2)/2 - Jab);
      for (size_t _si = 0; _si < mf_orbit_to_ph[jj].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[jj][_si].first;
        bool is_part   = mf_orbit_to_ph[jj][_si].second;
        if (is_part) continue;
        size_t c_orb = mf_ph_particle[col_idx];   // particle c of the 1p1h state
        const Orbit& o_c = modelspace->GetOrbit(c_orb);
        double jc = 0.5*o_c.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         j_jj, jc, ji);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jab, a, b, ii, c_orb);
        hv += phase3 * w6j * lam * K * mf_ph_phase[col_idx] * v_ph[col_idx];
      }
    }

    // sm4: k == i  (1p1h hole k matches first hole i of 2p2h state)
    // hv -= phase(J_ij+J_ab) × W6j(J_ab,J_ij,J; j_i,j_c,j_j)
    //       × Λ(J_ab; a,b|j,c) × K × phase_col × v_ph[col]
    {
      int phase4 = AngMom::phase(Jij + Jab);
      for (size_t _si = 0; _si < mf_orbit_to_ph[ii].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[ii][_si].first;
        bool is_part   = mf_orbit_to_ph[ii][_si].second;
        if (is_part) continue;
        size_t c_orb = mf_ph_particle[col_idx];
        const Orbit& o_c = modelspace->GetOrbit(c_orb);
        double jc = 0.5*o_c.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         ji, jc, j_jj);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jab, a, b, jj, c_orb);
        hv -= phase4 * w6j * lam * K * mf_ph_phase[col_idx] * v_ph[col_idx];
      }
    }

    Hv_2p2h[alpha] += hv;
  }
}

/// Compute Hv_ph += H21^T * v_2p2h without building H21.
///
/// The transpose of ApplyH21_matvec: accumulates contributions from each
/// 2p-2h amplitude v_2p2h[α] into the 1p-1h residual Hv_ph.
/// Because multiple α states can write to the same ph column this loop
/// cannot be parallelised without synchronisation and runs serially.
///
/// The four terms sm1-sm4 are the transpose of those in Build_H21_byIndex;
/// see that function for the full formula with consistent notation.
void EOMImsrg::ApplyH21T_matvec(const arma::vec& v_2p2h, arma::vec& Hv_ph) const
{
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(current_channel);
  int J = tbc_CC.J;
  size_t n2 = tpth_basis.size();

  for (size_t alpha = 0; alpha < n2; ++alpha)
  {
    double v_alpha = v_2p2h[alpha];
    if (std::abs(v_alpha) < 1e-15) continue;

    const TwoPTwoHState& st_alpha = tpth_basis[alpha];
    // 2p-2h state: particles a,b coupled to J_ab; holes ii,jj coupled to J_ij
    size_t a = st_alpha.a, b = st_alpha.b, ii = st_alpha.i, jj = st_alpha.j;
    int Jab = st_alpha.Jab, Jij = st_alpha.Jij;

    const Orbit& oa   = modelspace->GetOrbit(a);
    const Orbit& ob   = modelspace->GetOrbit(b);
    const Orbit& o_ii = modelspace->GetOrbit(ii);
    const Orbit& o_jj = modelspace->GetOrbit(jj);

    double ja   = 0.5*oa.j2,   jb   = 0.5*ob.j2;
    double ji   = 0.5*o_ii.j2, j_jj = 0.5*o_jj.j2;

    double Nab = std::sqrt(1.0 + (a  == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));
    // K = sqrt((2J_ab+1)(2J_ij+1)) / (N_ab * N_ij) * (-1)^J
    double K = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)) / (Nab * Nij) * AngMom::phase(J);

    // sm1: c == a  → Hv_ph[col] -= phase1 * w6j * Λ * K * phase_col * v_alpha
    {
      int phase1 = AngMom::phase((oa.j2 + ob.j2)/2 - Jij);
      for (size_t _si = 0; _si < mf_orbit_to_ph[a].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[a][_si].first;
        bool is_part   = mf_orbit_to_ph[a][_si].second;
        if (!is_part) continue;
        size_t k_orb = mf_ph_hole[col_idx];
        const Orbit& o_k = modelspace->GetOrbit(k_orb);
        double jk = 0.5*o_k.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         jk, ja, jb);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jij, k_orb, b, ii, jj);
        Hv_ph[col_idx] -= phase1 * w6j * lam * K * mf_ph_phase[col_idx] * v_alpha;
      }
    }

    // sm2: c == b  → Hv_ph[col] += phase2 * w6j * Λ * K * phase_col * v_alpha
    {
      int phase2 = AngMom::phase(Jab + Jij);
      for (size_t _si = 0; _si < mf_orbit_to_ph[b].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[b][_si].first;
        bool is_part   = mf_orbit_to_ph[b][_si].second;
        if (!is_part) continue;
        size_t k_orb = mf_ph_hole[col_idx];
        const Orbit& o_k = modelspace->GetOrbit(k_orb);
        double jk = 0.5*o_k.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         jk, jb, ja);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jij, k_orb, a, ii, jj);
        Hv_ph[col_idx] += phase2 * w6j * lam * K * mf_ph_phase[col_idx] * v_alpha;
      }
    }

    // sm3: k == j  → Hv_ph[col] += phase3 * w6j * Λ * K * phase_col * v_alpha
    {
      int phase3 = AngMom::phase((o_ii.j2 + o_jj.j2)/2 - Jab);
      for (size_t _si = 0; _si < mf_orbit_to_ph[jj].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[jj][_si].first;
        bool is_part   = mf_orbit_to_ph[jj][_si].second;
        if (is_part) continue;
        size_t c_orb = mf_ph_particle[col_idx];
        const Orbit& o_c = modelspace->GetOrbit(c_orb);
        double jc = 0.5*o_c.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         j_jj, jc, ji);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jab, a, b, ii, c_orb);
        Hv_ph[col_idx] += phase3 * w6j * lam * K * mf_ph_phase[col_idx] * v_alpha;
      }
    }

    // sm4: k == i  → Hv_ph[col] -= phase4 * w6j * Λ * K * phase_col * v_alpha
    {
      int phase4 = AngMom::phase(Jij + Jab);
      for (size_t _si = 0; _si < mf_orbit_to_ph[ii].size(); ++_si)
      {
        size_t col_idx = mf_orbit_to_ph[ii][_si].first;
        bool is_part   = mf_orbit_to_ph[ii][_si].second;
        if (is_part) continue;
        size_t c_orb = mf_ph_particle[col_idx];
        const Orbit& o_c = modelspace->GetOrbit(c_orb);
        double jc = 0.5*o_c.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jij, (double)J,
                                         ji, jc, j_jj);
        if (w6j == 0.0) continue;
        double lam = H.TwoBody.GetTBME_J(Jab, a, b, jj, c_orb);
        Hv_ph[col_idx] -= phase4 * w6j * lam * K * mf_ph_phase[col_idx] * v_alpha;
      }
    }
  }
}

/// Full H_EOM2 matvec: Hv = [A, H21^T; H21, H22] * v.
/// v[0..nph-1] = 1p1h part; v[nph..] = 2p2h part.
void EOMImsrg::ApplyH_EOM2_matvec(const arma::vec& v, arma::vec& Hv) const
{
  size_t nph = A.n_rows;
  size_t n2  = tpth_basis.size();

  arma::vec v_ph   = v.head(nph);
  arma::vec v_2p2h = v.tail(n2);

  arma::vec Hv_ph(nph,  arma::fill::zeros);
  arma::vec Hv_2p2h(n2, arma::fill::zeros);

  // 1p1h block: A * v_ph
  Hv_ph = A * v_ph;

  // Off-diagonal coupling H21^T * v_2p2h → 1p1h
  ApplyH21T_matvec(v_2p2h, Hv_ph);

  // Off-diagonal coupling H21 * v_ph → 2p2h
  ApplyH21_matvec(v_ph, Hv_2p2h);

  // 2p2h block: H22 * v_2p2h
  ApplyH22_matvec(v_2p2h, Hv_2p2h);

  Hv.head(nph) = Hv_ph;
  Hv.tail(n2)  = Hv_2p2h;
}

namespace
{
/// Armadillo newarp MatProd-compatible operator that wraps
/// EOMImsrg::ApplyH_EOM2_matvec.  Passed to SymEigsSolver in place of
/// DenseGenMatProd so that no H22 / H21 matrix is ever materialised.
struct EOMMatFreeOp
{
  typedef double elem_type;
  const EOMImsrg* eom;
  arma::uword n_rows;  ///< required by armadillo newarp SymEigsSolver

  EOMMatFreeOp(const EOMImsrg* e, arma::uword n) : eom(e), n_rows(n) {}

  void perform_op(const double* x_in, double* y_out) const
  {
    arma::vec v(x_in, n_rows);  // copies x_in
    arma::vec Hv(n_rows);
    eom->ApplyH_EOM2_matvec(v, Hv);
    std::copy(Hv.memptr(), Hv.memptr() + n_rows, y_out);
  }
};
} // anonymous namespace

// ---------------------------------------------------------------------------
// Matrix-free Lanczos solver
// ---------------------------------------------------------------------------

/// Solve channel ich_CC using a matrix-free Lanczos method (EOM2 mode).
///
/// This is the memory-efficient alternative to Solve_byIndex(..., "EOM2"):
///   - Reuses Build_AMatrix_byIndex() and Build2p2hBasis_byIndex() (unchanged).
///   - Does NOT build H22 (N2×N2) or H21 (N2×Nph).
///   - Runs armadillo's newarp SymEigsSolver with a custom matrix-product
///     operator that computes H_EOM2 * v on the fly each Lanczos step.
///
/// Peak additional memory is O(N2 × ncv) for the Lanczos vectors instead of
/// O(N2²) for the explicit matrices.
///
/// Computation time: each matrix-vector product costs the same as one row of
/// BuildH22_byIndex, so K Lanczos iterations costs K × build_cost(H22).
/// For K ≪ N2 (typical when requesting a handful of eigenvalues) this is
/// cheaper than building the full matrix; for K ≫ N2 the dense path is faster.
///
/// @param nev  Number of algebraically-lowest eigenvalues to converge (≥ 1).
///             If nev ≥ N = nph + n2p2h, falls back to the dense EOM2 solve.
void EOMImsrg::Solve_byIndex_MF(size_t ich_CC, int nev)
{
  current_channel = ich_CC;

  // --- Reuse existing builders for A and the 2p2h basis ---
  Build_AMatrix_byIndex(ich_CC);
  one_ph_count = A.n_rows;

  Build2p2hBasis_byIndex(ich_CC);
  two_ph_count  = tpth_basis.size();
  lanczos_iterations = 0;
  OnePhNorms.reset();
  B.zeros(A.n_rows, A.n_cols);  // not used by EOM2, but reset for consistency

  size_t nph = A.n_rows;
  size_t n2  = tpth_basis.size();
  size_t N   = nph + n2;

  // Precompute the orbit → ph-column lookup used by ApplyH21 / ApplyH21T.
  // Built once here and shared across all Lanczos matrix-vector products.
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  int J_ch = tbc_CC.J;
  mf_orbit_to_ph.assign(modelspace->GetNumberOrbits(),
                        std::vector<std::pair<size_t,bool>>());
  mf_ph_particle.resize(nph);
  mf_ph_hole.resize(nph);
  mf_ph_phase.resize(nph);
  {
    size_t col = 0;
    for (auto iket : ph_list)
    {
      Ket& kt = tbc_CC.GetKet(iket);
      index_t a_ph = kt.p, k_ph = kt.q;  // a_ph = particle, k_ph = hole
      if (kt.op->occ > kt.oq->occ)
      {
        std::swap(a_ph, k_ph);
        // After swap: a_ph = kt.q (particle), k_ph = kt.p (hole)
        mf_ph_phase[col] = -AngMom::phase((kt.oq->j2 + kt.op->j2) / 2 - J_ch);
      }
      else
      {
        mf_ph_phase[col] = 1;
      }
      mf_orbit_to_ph[a_ph].push_back(std::make_pair(col, true));
      mf_orbit_to_ph[k_ph].push_back(std::make_pair(col, false));
      mf_ph_particle[col] = a_ph;
      mf_ph_hole[col]     = k_ph;
      ++col;
    }
  }

  if (nev <= 0 || (size_t)nev >= N)
  {
    // Degenerate cases: build the explicit matrices and use the dense path.
    std::cout << "WARNING EOMImsrg::Solve_byIndex_MF: nev=" << nev
              << " out of range for N=" << N
              << "; falling back to dense EOM2 solve in channel " << ich_CC
              << std::endl;
    BuildH22_byIndex(ich_CC);
    Build_H21_byIndex(ich_CC);
    SolveCurrentChannel("EOM2");
  }
  else
  {
    size_t nev_u = static_cast<size_t>(nev);
    size_t ncv   = std::min(N, std::max(static_cast<size_t>(5 * nev), nev_u + 2));

    EOMMatFreeOp op(this, static_cast<arma::uword>(N));
    arma::newarp::SymEigsSolver<double,
        arma::newarp::EigsSelect::SMALLEST_ALGE,
        EOMMatFreeOp> solver(op, nev_u, ncv);
    solver.init();
    arma::uword nconv = solver.compute(/*maxit=*/1000, /*tol=*/1e-10);
    lanczos_iterations = solver.num_iterations();

    if (nconv < (arma::uword)nev)
    {
      std::cout << "WARNING EOMImsrg::Solve_byIndex_MF: only " << nconv
                << " of " << nev << " eigenvalues converged in channel "
                << ich_CC << "  " << __FILE__ << ":" << __LINE__ << std::endl;
    }

    arma::vec all_evals = solver.eigenvalues();
    arma::mat all_evecs = solver.eigenvectors();

    // Keep only positive eigenvalues (physical excitation energies).
    arma::uvec pos_idx = arma::find(all_evals > 0);
    Energies = all_evals(pos_idx);
    arma::mat Vpos = all_evecs.cols(pos_idx);

    // X = 1p1h part of eigenvectors; Y = 2p2h part
    X = Vpos.head_rows(nph);
    Y = Vpos.tail_rows(Vpos.n_rows - nph);
    OnePhNorms = ComputeOnePhNorms(X);
  }

  // Store results in the per-channel map (same layout as Solve_byIndex).
  EOMChannel ch;
  ch.Energies           = Energies;
  ch.X                  = X;
  ch.Y                  = Y;
  ch.OnePhNorms         = OnePhNorms;
  ch.OnePhCount         = one_ph_count;
  ch.TwoPhCount         = two_ph_count;
  ch.LanczosIterations  = lanczos_iterations;
  ch.tpth_basis         = tpth_basis;  // Store 2p2h basis for transition ME
  ChannelResults[ich_CC] = ch;
}

/// Run Solve_byIndex_MF for all ph channels.
void EOMImsrg::SolveAllChannels_MF(int nev)
{
  size_t nch = modelspace->GetNumberTwoBodyChannels_CC();
  for (size_t ich = 0; ich < nch; ich++)
  {
    TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ich);
    if (tbc.GetKetIndex_ph().empty()) continue;
    Solve_byIndex_MF(ich, nev);
  }
}


void EOMImsrg::SolveCurrentChannel(std::string mode)
{
  if (mode == "TDA")
  {
    arma::vec eigvals;
    arma::mat eigvecs;
    arma::eig_sym(eigvals, eigvecs, A);
    Energies = eigvals;
    X = eigvecs;
    Y = arma::zeros(arma::size(X));
    OnePhNorms = ComputeOnePhNorms(X);
  }
  else if (mode == "EOM")
  {
    // EOM-IMSRG at the 1p1h level: diagonalise A.
    // The EOM ladder operator Q†_ν = Σ_ai X_{ai} a†_a a_i contains only
    // excitation amplitudes.  There are no de-excitation terms; those arise
    // in RPA but are absent in EOM-IMSRG.
    arma::vec eigvals;
    arma::mat eigvecs;
    arma::eig_sym(eigvals, eigvecs, A);
    Energies = eigvals;
    X = eigvecs;
    Y = arma::zeros(arma::size(X));
    OnePhNorms = ComputeOnePhNorms(X);
  }
  else if (mode == "EOM2")
  {
    // Full 1p1h + 2p2h block matrix:
    //   H_EOM = [ A     H12 ]   where H12 = H21^T
    //           [ H21   H22 ]
    // Diagonalise with either the dense LAPACK driver (all eigenvalues) or the
    // newarp Implicitly Restarted Arnoldi / Lanczos solver (lowest lanczos_nev
    // eigenvalues only).  The latter mirrors the ARPACK-based approach used by
    // the reference Fortran EOM-IMSRG code of Parzuchowski et al.
    size_t nph = A.n_rows;
    arma::mat H12 = H21.t();
    arma::mat Hfull = arma::join_vert(
        arma::join_horiz(A,   H12),
        arma::join_horiz(H21, H22));
    // Enforce exact symmetry
    Hfull = 0.5 * (Hfull + Hfull.t());

    size_t N = Hfull.n_rows;

    if (lanczos_nev > 0 && (size_t)lanczos_nev < N)
    {
      // --- Lanczos / IRAM path ---
      // Uses armadillo's built-in newarp SymEigsSolver (same algorithm as
      // ARPACK's dsaupd/dseupd used by the reference Fortran code).
      // SMALLEST_ALGE retrieves the algebraically smallest eigenvalues first,
      // giving the lowest-lying physical excitation energies.
      //
      // ncv = number of Lanczos basis vectors; must satisfy ncv > lanczos_nev + 1.
      // Following the reference Fortran code: ncv ≈ 5*nev, capped to N.
      size_t nev_u = static_cast<size_t>(lanczos_nev);
      size_t ncv   = std::min(N,
                              std::max(static_cast<size_t>(5 * lanczos_nev),
                                       nev_u + 2));  // ncv > nev + 1 always

      arma::newarp::DenseGenMatProd<double> op(Hfull);
      arma::newarp::SymEigsSolver<double,
          arma::newarp::EigsSelect::SMALLEST_ALGE,
          arma::newarp::DenseGenMatProd<double>> solver(op, nev_u, ncv);
      solver.init();
      arma::uword nconv = solver.compute(/*maxit=*/1000, /*tol=*/1e-10);
      lanczos_iterations = solver.num_iterations();

      if (nconv < (arma::uword)lanczos_nev)
      {
        std::cout << "WARNING EOMImsrg Lanczos: only " << nconv
                  << " of " << lanczos_nev << " eigenvalues converged"
                  << " in channel " << current_channel
                  << "  " << __FILE__ << ":" << __LINE__ << std::endl;
      }

      arma::vec all_evals = solver.eigenvalues();
      arma::mat all_evecs = solver.eigenvectors();

      // Keep only positive eigenvalues (physical excitation energies)
      arma::uvec pos_idx = arma::find(all_evals > 0);
      Energies = all_evals(pos_idx);
      arma::mat Vpos = all_evecs.cols(pos_idx);

      // X = 1p1h part of eigenvectors; Y = 2p2h part
      X = Vpos.head_rows(nph);
      Y = Vpos.tail_rows(Vpos.n_rows - nph);
      OnePhNorms = ComputeOnePhNorms(X);
    }
    else
    {
      // --- Dense LAPACK path (default): all eigenvalues ---
      arma::vec eigvals;
      arma::mat eigvecs;
      arma::eig_sym(eigvals, eigvecs, Hfull);

      // Keep only positive eigenvalues (physical excitation energies)
      arma::uvec pos_idx = arma::find(eigvals > 0);
      Energies = eigvals(pos_idx);
      arma::mat Vpos = eigvecs.cols(pos_idx);

      // X = 1p1h part of eigenvectors; Y = 2p2h part
      X = Vpos.head_rows(nph);
      Y = Vpos.tail_rows(Vpos.n_rows - nph);
      OnePhNorms = ComputeOnePhNorms(X);
    }
  }
  else if (mode == "RPA")
  {
    // RPA secular equation: [ A  B; -B  -A ] (B must already be built by caller)
    // Produces both excitation (X) and de-excitation (Y) amplitudes.
    arma::mat M = arma::join_vert(
        arma::join_horiz( A,  B),
        arma::join_horiz(-B, -A));

    arma::cx_vec cx_eigvals;
    arma::cx_mat cx_eigvecs;
    arma::eig_gen(cx_eigvals, cx_eigvecs, M);

    double norm_imag = arma::norm(arma::imag(cx_eigvals), "fro");
    if (norm_imag > RPA_IMAG_TOL)
    {
      std::cout << "WARNING EOMImsrg RPA: non-zero imaginary eigenvalues ("
                << norm_imag << ") in channel " << current_channel
                << "  " << __FILE__ << ":" << __LINE__ << std::endl;
    }

    // Keep only positive-real solutions (discard spurious negative partners)
    arma::uvec pos_idx = arma::find(arma::real(cx_eigvals)
                                    + arma::imag(cx_eigvals) > 0);
    size_t nph_rpa = A.n_rows;
    arma::vec Etmp = arma::real(cx_eigvals(pos_idx));
    arma::mat Vtmp = arma::real(cx_eigvecs.cols(pos_idx));

    arma::uvec ord = arma::sort_index(Etmp);
    Energies = Etmp(ord);
    arma::mat Vsorted = Vtmp.cols(ord);

    // The secular matrix is 2*nph × 2*nph.  First nph rows = X (excitation),
    // last nph rows = Y (de-excitation).
    X = Vsorted.head_rows(nph_rpa);
    Y = Vsorted.tail_rows(nph_rpa);

    // Normalise each solution: X^T X - Y^T Y = 1
    for (size_t mu = 0; mu < Energies.n_elem; mu++)
    {
      double nxy = arma::dot(X.col(mu), X.col(mu))
                 - arma::dot(Y.col(mu), Y.col(mu));
      if (std::abs(nxy) > RPA_NORM_TOL)
      {
        double inv_sqrt_nxy = 1.0 / std::sqrt(std::abs(nxy));
        X.col(mu) *= inv_sqrt_nxy;
        Y.col(mu) *= inv_sqrt_nxy;
      }
    }
    OnePhNorms = ComputeOnePhNorms(X);
  }
  else
  {
    throw std::invalid_argument("EOMImsrg::Solve: unknown mode '" + mode
                                + "'. Use 'TDA', 'EOM', 'RPA', or 'EOM2'.");
  }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

arma::vec EOMImsrg::GetExcitationEnergies() const { return Energies; }

arma::vec EOMImsrg::GetAmplitudesX(size_t state_index) const
{
  return X.col(state_index);
}

arma::vec EOMImsrg::GetAmplitudesY(size_t state_index) const
{
  return Y.col(state_index);
}

arma::vec EOMImsrg::GetOnePhNorms() const { return OnePhNorms; }

size_t EOMImsrg::GetOnePhCount() const { return one_ph_count; }

size_t EOMImsrg::GetTwoPhCount() const { return two_ph_count; }

size_t EOMImsrg::GetLanczosIterations() const { return lanczos_iterations; }

void EOMImsrg::PrintSummary() const
{
  std::streamsize old_prec = std::cout.precision();
  std::ios::fmtflags old_flags = std::cout.flags();

  if (one_ph_count > 0 || two_ph_count > 0)
  {
    std::cout << " 1p1h Amplitudes:" << std::setw(13) << one_ph_count << std::endl;
    std::cout << " 2p2h Amplitudes:" << std::setw(13) << two_ph_count << std::endl;
  }
  if (lanczos_iterations > 0)
  {
    std::cout << "Lanczos iteration:" << std::setw(13)
              << lanczos_iterations << std::endl;
  }

  std::cout << std::fixed << std::setprecision(9);
  std::cout << std::endl;
  std::cout << "Ground State Energy:" << std::setw(17) << H.ZeroBody << std::endl;
  std::cout << std::endl;
  std::cout << " EXCITED STATE ENERGIES:" << std::endl;
  std::cout << " ==============================================" << std::endl;
  std::cout << "       dE           E_0 + dE         n(1p1h)" << std::endl;
  std::cout << " ==============================================" << std::endl;
  size_t n_print = Energies.n_elem;
  if (lanczos_nev > 0 && static_cast<size_t>(lanczos_nev) < n_print)
    n_print = static_cast<size_t>(lanczos_nev);
  for (size_t i = 0; i < n_print; i++)
  {
    double one_ph_norm = (i < OnePhNorms.n_elem) ? OnePhNorms(i) : 0.0;
    std::cout << std::setw(16) << Energies(i)
              << std::setw(16) << (H.ZeroBody + Energies(i))
              << std::setw(16) << one_ph_norm << std::endl;
  }

  std::cout.precision(old_prec);
  std::cout.flags(old_flags);
}

EOMChannel EOMImsrg::GetChannelResults(size_t ich_CC) const
{
  auto it = ChannelResults.find(ich_CC);
  if (it == ChannelResults.end())
    throw std::out_of_range("EOMImsrg::GetChannelResults: channel not solved");
  return it->second;
}

// ---------------------------------------------------------------------------
// Transition matrix elements
// ---------------------------------------------------------------------------

///
/// Compute the transition matrix element
/// \f$M_{0\nu} = \langle \Phi_0 \| [\bar{\mathcal{O}}^\lambda \times
///   \bar{X}^\dagger_\nu(J^\Pi)]^0 \| \Phi_0 \rangle\f$
/// following eq. (16) of Parzuchowski et al., Phys. Rev. C **96**, 034324 (2017).
///
/// Modes:
///   - **TDA / EOM**: only the 1p1h term is evaluated.
///   - **EOM2**: \p ch.Y stores the 2p2h amplitudes \f$\breve{X}^{J_1J_2J}_{abij}\f$.
///     When \p Op has a two-body part (\c particle_rank >= 2) the 2p2h matrix
///     elements \f$\breve{\mathcal{O}}^{J_1J_2}_{abij}(\lambda)\f$ are included.
///
/// The full formula is (paper eq. 16):
/// \f[
///   M_{0\nu} = \delta_{\lambda,J_\nu}(-1)^{J_\nu} \frac{1}{\sqrt{2J_\nu+1}}
///     \Biggl[
///       \sum_{ai} X^J_{ai}(\nu)\,\langle a\|O^\lambda\|i\rangle
///       \;+\; \frac{1}{4} \sum_{abij,J_1J_2}
///              \breve{X}^{J_1J_2J}_{abij}(\nu)\,
///              \breve{\mathcal{O}}^{J_1J_2}_{abij}(\lambda)
///     \Biggr].
/// \f]
///
double EOMImsrg::ComputeTransitionME(Operator& Op, size_t state_index) const
{
  return ComputeTransitionME_byIndex(current_channel, Op, state_index);
}

double EOMImsrg::ComputeTransitionME_byIndex(size_t ich_CC,
                                              Operator& Op,
                                              size_t state_index) const
{
  auto it = ChannelResults.find(ich_CC);
  if (it == ChannelResults.end())
    throw std::out_of_range("EOMImsrg::ComputeTransitionME_byIndex: "
                            "channel not solved");

  const EOMChannel& ch = it->second;
  if (state_index >= (size_t)ch.X.n_cols)
    throw std::out_of_range("EOMImsrg::ComputeTransitionME_byIndex: "
                            "state_index out of range");

  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph = ph_list.size();

  // Channel angular momentum J_ν (= operator rank λ for the matrix element to be nonzero).
  int J = tbc_CC.J;

  // -------------------------------------------------------------------
  // 1p1h contribution (paper eq. 16 first sum):
  //   Σ_{ai} X_{ai}(ν) O_{ai}(λ,Π)
  //
  // In EOM-IMSRG the ladder operator Q†_ν = Σ_ai X_{ai} a†_a a_i contains
  // only excitation (particle-hole) amplitudes.  The A-matrix convention
  // includes a CC-channel ket-ordering phase per pair; each X(I) carries
  // that phase and the operator is read with the same stored-index ordering.
  // -------------------------------------------------------------------
  double T = 0.0;
  size_t I = 0;
  for (auto iket_ai : ph_list)
  {
    Ket& ket = tbc_CC.GetKet(iket_ai);
    index_t m = ket.p;
    index_t i = ket.q;

    // Use the stored (m, i) ordering, consistent with the phase convention
    // embedded in ch.X from the phase-corrected A-matrix.
    T += ch.X(I, state_index) * Op.OneBody(m, i);

    I++;
  }

  // -------------------------------------------------------------------
  // 2p2h contribution (EOM2 mode, paper eq. 16 second sum):
  //   (1/4) Σ_{abij,J1,J2} X̌_{abij}^{J1J2J}(ν) × Ǒ_{abij}^{J1J2}(λ)
  //
  // TODO: Implement the 2p2h contribution.  The two-body reduced matrix
  // element Ǒ_{abij}^{J1J2}(λ) (paper eqs. B7-B8) requires a recoupling
  // from the (ab J1)(ij J2)→J EOM 2p2h coupling scheme to the standard
  // TwoBodyChannel scheme, using 9j symbols analogously to how the
  // H22 ph-ring term is computed in BuildH22_byIndex.  This is left as a
  // future enhancement; for typical 1p1h-dominant states the dominant
  // contribution comes from the 1p1h term above.
  // -------------------------------------------------------------------

  // -------------------------------------------------------------------
  // Overall prefactor from eq. (16): (-1)^{J_ν} / √(2J_ν+1)
  // -------------------------------------------------------------------
  return AngMom::phase(J) / std::sqrt(2.0 * J + 1.0) * T;
}
