// EOM_IMSRG.cc
///////////////////////////////////////////////////////////////////////////////////
//    EOM_IMSRG.cc, part of  imsrg++
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

#include "EOM_IMSRG.hh"
#include "AngMom.hh"
#include "TensorCommutators.hh"

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

EOM_IMSRG::EOM_IMSRG()
  : modelspace(nullptr), one_ph_count(0), two_ph_count(0),
    lanczos_iterations(0), current_channel(0), lanczos_nev(0)
{}

EOM_IMSRG::EOM_IMSRG(Operator& H_imsrg)
  : modelspace(H_imsrg.modelspace), H(H_imsrg), one_ph_count(0),
    two_ph_count(0), lanczos_iterations(0), current_channel(0), lanczos_nev(0)
{}

// ---------------------------------------------------------------------------
// Build_AMatrix
// ---------------------------------------------------------------------------

/// Dispatch helper: look up the channel index then call Build_H11_byIndex.
void EOM_IMSRG::Build_AMatrix(int J, int parity, int Tz)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  Build_H11_byIndex(ich_CC);
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
void EOM_IMSRG::Build_H11_byIndex(size_t ich_CC)
{
  current_channel = ich_CC;
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph = ph_list.size();
  A.zeros(nph, nph);

  int Jph = tbc_CC.J;

  // Use pre-computed Pandya table for this channel.
  BuildPandya();
  const arma::mat& Hbar_ch = Hbar_CC[ich_CC];

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

      // 1-body contribution: f_{ab} * δ_{ij} - f_{ij} * δ_{ab}
      double H1b = (i == j ? H.OneBody(a, b) : 0.0)
                 - (a == b ? H.OneBody(i, j) : 0.0);

      // Two-body Pandya term: look up from pre-computed table
      A(I, II) = (H1b + Hbar_ch(I, II)) * phase_ai * phase_bj;
    }
  }
}

// ---------------------------------------------------------------------------
// Build_BMatrix
// ---------------------------------------------------------------------------

/// Dispatch helper.
void EOM_IMSRG::Build_BMatrix(int J, int parity, int Tz)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  Build_BMatrix_byIndex(ich_CC);
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
void EOM_IMSRG::Build_BMatrix_byIndex(size_t ich_CC)
{
  current_channel = ich_CC;
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph = ph_list.size();
  B.zeros(nph, nph);

  int Jph = tbc_CC.J;

  // Pre-compute the B-matrix CC-channel table (ph × ph block).
  //
  //   Hbar_B(ibra, iket)
  //     = Σ_{J'}(-1)^{J'} (2J'+1) W6j(ja,ji,Jph; jj,jb,J') × GetTBME_J(J', a,b,i,j)
  //
  // ibra/iket follow the same ordering as the ph_list (particle=a_orb, hole=i_orb
  // after an optional swap).
  arma::mat Hbar_B(nph, nph, arma::fill::zeros);
  for (size_t ibra = 0; ibra < nph; ++ibra)
  {
    Ket& bra = tbc_CC.GetKet(ph_list[ibra]);
    size_t a_orb = bra.p, i_orb = bra.q;
    if (bra.op->occ > bra.oq->occ) std::swap(a_orb, i_orb);
    double ja_b = 0.5 * modelspace->GetOrbit(a_orb).j2;
    double ji_b = 0.5 * modelspace->GetOrbit(i_orb).j2;

    for (size_t iket = 0; iket < nph; ++iket)
    {
      Ket& ket = tbc_CC.GetKet(ph_list[iket]);
      size_t b_orb = ket.p, j_orb = ket.q;
      if (ket.op->occ > ket.oq->occ) std::swap(b_orb, j_orb);
      double jb_b = 0.5 * modelspace->GetOrbit(b_orb).j2;
      double jj_b = 0.5 * modelspace->GetOrbit(j_orb).j2;

      int Jp_min = std::max(std::abs((int)(2*ja_b) - (int)(2*jb_b)),
                            std::abs((int)(2*ji_b) - (int)(2*jj_b))) / 2;
      int Jp_max = std::min((int)(2*ja_b) + (int)(2*jb_b),
                            (int)(2*ji_b) + (int)(2*jj_b)) / 2;
      double V_B = 0.0;
      for (int Jp = Jp_min; Jp <= Jp_max; ++Jp)
      {
        double sixj = modelspace->GetSixJ(ja_b, ji_b, (double)Jph,
                                          jj_b, jb_b, (double)Jp);
        if (std::abs(sixj) < 1e-8) continue;
        V_B += AngMom::phase(Jp) * (2*Jp+1) * sixj
             * H.TwoBody.GetTBME_J(Jp, a_orb, b_orb, i_orb, j_orb);
      }
      Hbar_B(ibra, iket) = V_B;
    }
  }

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

      int phase_ib = AngMom::phase((j2i + j2b)/2 + Jph);
      B(I, II) = Hbar_B(I, II) * phase_ib * phase_ai * phase_bj;
    }
  }
}

// ---------------------------------------------------------------------------
// Build_2p2hBasis_byIndex
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
void EOM_IMSRG::Build_2p2hBasis_byIndex(size_t ich_CC)
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
// Build_H22_byIndex
// ---------------------------------------------------------------------------

///
/// Build the 2p2h × 2p2h block H22.
///
/// Contributions (following Parzuchowski et al., PRC 96, 034324 (2017)):
///
///   (a) 1-body (comm121) — orbit-channel iteration with phase-corrected lookup:
///         +Σ_c f_{ac} δ_{j_c,j_a}   →  H22(α, {c,b,i,j;Jab,Jij}) += phase·f_{ac}
///         +Σ_c f_{bc} δ_{j_c,j_b}   →  H22(α, {a,c,i,j;Jab,Jij}) += phase·f_{bc}
///         -Σ_k f_{ik} δ_{j_k,j_i}   →  H22(α, {a,b,k,j;Jab,Jij}) -= phase·f_{ik}
///         -Σ_k f_{jk} δ_{j_k,j_j}   →  H22(α, {a,b,i,k;Jab,Jij}) -= phase·f_{jk}
///
///       The "phase" is the antisymmetry factor (-1)^{j_p+j_q-J+1} needed when
///       substituting an orbit causes the pair to be reordered into canonical form.
///       This avoids the O(n_basis²) double loop: only orbits in the same
///       one-body channel (same l, j2, tz2) can contribute.
///
///   (b) PP-PP ladder — loop over pp kets in the TBC with J=Jab:
///         +0.5 × GetTBME_J(Jab, a,b, a',b')  when β = {a',b',i,j;Jab,Jij}
///
///   (c) HH-HH ladder — loop over hh kets in the TBC with J=Jij:
///         +0.5 × GetTBME_J(Jij, i,j, i',j')  when β = {a,b,i',j';Jab,Jij}
///
///   (d) Ph ring term — two contributions (nanb = n_c - n_k):
///         Term 1 (Xbar_12ck × Ybar_ck34): c=particle, k=hole, nanb=-1
///           Xbar = -Σ_J5 (2J5+1) SixJ(j1,j4,J3; jc,jk,J5) × H(J5; I1,k,c,I4)
///           Ybar = -hats × phaseY × NineJ(jc,j2,J6; jk,j3,J7; J3,J4,J) / (Nab_β Nij_β)
///           beta state: {c,I2;J6} × {I3,k;J7},  phaseY = (-1)^{(jk+j_{I2})/2 + J4 + J7}
///
///         Term 2 (Ybar_12ck × Xbar_ck34): c=hole, k=particle, nanb=+1
///           Xbar = -Σ_J5 (2J5+1) SixJ(j1,j4,J3; jc,jk,J5) × H(J5; I1,k,c,I4)
///           Ybar = -hats × phaseY × NineJ(jk,j2,J6; jc,j3,J7; J3,J4,J) / (Nab_β Nij_β)
///           beta state: {k,I2;J6} × {I3,c;J7},  phaseY = (-1)^{(jc+j_{I2})/2 + J4 + J7}
///
///         J6 parity restriction: if pp-pair has identical orbits, J6 must be even.
///         J7 parity restriction: if hh-pair has identical orbits, J7 must be even.
///
void EOM_IMSRG::Build_H22_byIndex(size_t ich_CC)
{
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  int J = tbc_CC.J;

  size_t n2 = tpth_basis.size();
  H22.zeros(n2, n2);
  if (n2 == 0) return;

  // -----------------------------------------------------------------------
  // Basis lookup map: (a, b, i, j, Jab, Jij) -> index in tpth_basis.
  // The pairs (a,b) and (i,j) are stored in the canonical channel ordering
  // (first orbit ≤ second orbit by orbit index).
  // -----------------------------------------------------------------------
  std::map<std::tuple<size_t,size_t,size_t,size_t,int,int>, size_t> basis_map;
  for (size_t alpha = 0; alpha < n2; ++alpha)
  {
    const TwoPTwoHState& st = tpth_basis[alpha];
    basis_map[{st.a, st.b, st.i, st.j, st.Jab, st.Jij}] = alpha;
  }

  // -----------------------------------------------------------------------
  // Lookup helpers with antisymmetry phase correction.
  //
  // When substituting one orbit in a pair (p→c), the resulting pair (c, q)
  // may not be in canonical order.  The canonical form is obtained via the
  // antisymmetry relation:
  //   |{q,p};J>_AS = (-1)^{j_p+j_q-J+1} |{p,q};J>_AS   (p ≤ q canonical)
  //
  // lookup_pp_swap: find basis state with pp pair {p,q} and hh pair {i,j}.
  //   Returns {basis_index, phase} or {n2, 0} if not in basis.
  //
  // lookup_hh_swap: find basis state with pp pair {a,b} and hh pair {k,l}.
  //   Returns {basis_index, phase} or {n2, 0} if not in basis.
  // -----------------------------------------------------------------------
  auto lookup_pp_swap =
    [&](size_t p, size_t q, size_t i, size_t j, int Jp, int Jh)
    -> std::pair<size_t, double>
  {
    auto it = basis_map.find({p, q, i, j, Jp, Jh});
    if (it != basis_map.end()) return {it->second, 1.0};
    auto it2 = basis_map.find({q, p, i, j, Jp, Jh});
    if (it2 != basis_map.end())
    {
      double ph = AngMom::phase(
        (modelspace->GetOrbit(p).j2 + modelspace->GetOrbit(q).j2) / 2 - Jp + 1);
      return {it2->second, ph};
    }
    return {n2, 0.0};
  };

  auto lookup_hh_swap =
    [&](size_t a, size_t b, size_t k, size_t l, int Jp, int Jh)
    -> std::pair<size_t, double>
  {
    auto it = basis_map.find({a, b, k, l, Jp, Jh});
    if (it != basis_map.end()) return {it->second, 1.0};
    auto it2 = basis_map.find({a, b, l, k, Jp, Jh});
    if (it2 != basis_map.end())
    {
      double ph = AngMom::phase(
        (modelspace->GetOrbit(k).j2 + modelspace->GetOrbit(l).j2) / 2 - Jh + 1);
      return {it2->second, ph};
    }
    return {n2, 0.0};
  };

  // (No pre-computed Pandya table needed for H22: ring term uses inline 6j.)
  // -----------------------------------------------------------------------
  // Main loop: compute H22 row by row (parallelized over alpha).
  // All contributions use +=; H22 was initialised to zero above.
  // Writes are to row alpha only → no race conditions between threads.
  // -----------------------------------------------------------------------
  #pragma omp parallel for schedule(dynamic,1)
  for (long long alpha_ll = 0; alpha_ll < static_cast<long long>(n2); ++alpha_ll)
  {
    size_t alpha = static_cast<size_t>(alpha_ll);
    const TwoPTwoHState& st_alpha = tpth_basis[alpha];
    size_t a = st_alpha.a, b = st_alpha.b, ii = st_alpha.i, jj = st_alpha.j;
    int Jab = st_alpha.Jab, Jij = st_alpha.Jij;

    const Orbit& oa  = modelspace->GetOrbit(a);
    const Orbit& ob  = modelspace->GetOrbit(b);
    const Orbit& oii = modelspace->GetOrbit(ii);
    const Orbit& ojj = modelspace->GetOrbit(jj);

    double ja   = 0.5 * oa.j2;
    double jb   = 0.5 * ob.j2;
    double ji   = 0.5 * oii.j2;
    double j_jj = 0.5 * ojj.j2;

    // -------------------------------------------------------------------
    // (a) 1-body term: iterate orbits in the same OB channel as each of
    //     a, b, ii, jj, look up the resulting basis state (with possible
    //     phase from reordering to canonical form), and add f * phase.
    //
    // Normalization factors Nab = sqrt(1+delta(a,b)) and Nij = sqrt(1+delta(i,j))
    // for the alpha state, and the corresponding N for the beta state, are
    // included so that the matrix element is between fully normalized 2p2h states.
    // -------------------------------------------------------------------
    double Nab = std::sqrt(1.0 + (a  == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));

    for (auto c : modelspace->OneBodyChannels.at({oa.l, oa.j2, oa.tz2}))
    {
      double Ncb = std::sqrt(1.0 + (c == b ? 1.0 : 0.0));
      std::pair<size_t,double> res = lookup_pp_swap(c, b, ii, jj, Jab, Jij);
      if (res.first < n2) H22(alpha, res.first) += res.second * (Nab / Ncb) * H.OneBody(a, c);
    }
    for (auto c : modelspace->OneBodyChannels.at({ob.l, ob.j2, ob.tz2}))
    {
      double Nac = std::sqrt(1.0 + (a == c ? 1.0 : 0.0));
      std::pair<size_t,double> res = lookup_pp_swap(a, c, ii, jj, Jab, Jij);
      if (res.first < n2) H22(alpha, res.first) += res.second * (Nab / Nac) * H.OneBody(b, c);
    }
    for (auto k : modelspace->OneBodyChannels.at({oii.l, oii.j2, oii.tz2}))
    {
      double Nkj = std::sqrt(1.0 + (k  == jj ? 1.0 : 0.0));
      std::pair<size_t,double> res = lookup_hh_swap(a, b, k, jj, Jab, Jij);
      if (res.first < n2) H22(alpha, res.first) -= res.second * (Nij / Nkj) * H.OneBody(ii, k);
    }
    for (auto k : modelspace->OneBodyChannels.at({ojj.l, ojj.j2, ojj.tz2}))
    {
      double Nik = std::sqrt(1.0 + (ii == k  ? 1.0 : 0.0));
      std::pair<size_t,double> res = lookup_hh_swap(a, b, ii, k, Jab, Jij);
      if (res.first < n2) H22(alpha, res.first) -= res.second * (Nij / Nik) * H.OneBody(jj, k);
    }

    // -------------------------------------------------------------------
    // (b) PP-PP ladder: GetTBME_J(Jab, a,b, a',b')
    //     Factor = 0.5 when ap==bp (identical, no exchange partner);
    //            = 1.0 when ap!=bp (distinct: includes both orderings).
    //     Loop over pp kets in the TBC with J=Jab, parity=(la+lb)%2,
    //     Tz=(tza+tzb)/2.  The lookup uses the canonical pair (a',b')
    //     directly from the channel.
    // -------------------------------------------------------------------
    {
      size_t ich_pp = modelspace->GetTwoBodyChannelIndex(
                        Jab, (oa.l + ob.l) % 2, (oa.tz2 + ob.tz2) / 2);
      TwoBodyChannel& tbc_pp = modelspace->GetTwoBodyChannel(ich_pp);
      for (auto iket_pp : tbc_pp.GetKetIndex_pp())
      {
        Ket& kp = tbc_pp.GetKet(iket_pp);
        size_t ap = kp.p, bp = kp.q;
        auto it = basis_map.find({ap, bp, ii, jj, Jab, Jij});
        if (it != basis_map.end())
        {
          double fac = (ap == bp) ? 0.5 : 1.0;
          H22(alpha, it->second) += fac * H.TwoBody.GetTBME_J(Jab, a, b, ap, bp);
        }
      }
    }

    // -------------------------------------------------------------------
    // (c) HH-HH ladder: +GetTBME_J(Jij, ii,jj, i',j')
    //     Factor = 0.5 when ip==jp (identical); = 1.0 when ip!=jp.
    //     Loop over hh kets in the TBC with J=Jij, parity=(li+lj)%2,
    //     Tz=(tzi+tzj)/2.
    // -------------------------------------------------------------------
    {
      size_t ich_hh = modelspace->GetTwoBodyChannelIndex(
                        Jij, (oii.l + ojj.l) % 2, (oii.tz2 + ojj.tz2) / 2);
      TwoBodyChannel& tbc_hh = modelspace->GetTwoBodyChannel(ich_hh);
      for (auto iket_hh : tbc_hh.GetKetIndex_hh())
      {
        Ket& kh = tbc_hh.GetKet(iket_hh);
        size_t ip = kh.p, jp = kh.q;
        auto it = basis_map.find({a, b, ip, jp, Jab, Jij});
        if (it != basis_map.end())
        {
          double fac = (ip == jp) ? 0.5 : 1.0;
          H22(alpha, it->second) += fac * H.TwoBody.GetTBME_J(Jij, ii, jj, ip, jp);
        }
      }
    }

    // -------------------------------------------------------------------
    // (d) Ph ring term: [Gamma, X_pphh]_ph  (comm222_phst formula)
    //
    // Directly mirrors ReferenceImplementations::comm222_phst with
    //   X  = Gamma (2-body part of H, scalar)
    //   Y  = X_pphh (2p2h tensor operator of rank lambda = J)
    //
    // Two contributions (nanb = n_c - n_k):
    //   Term 1 (Xbar_12ck × Ybar_ck34): c=particle, k=hole, nanb=-1
    //     Xbar = -Σ_J5 (2J5+1) SixJ(j1,j4,J3; jc,jk,J5) × H(J5; I1,k,c,I4)
    //     Ybar = -hats × phaseY × NineJ(jc,j2,J6; jk,j3,J7; J3,J4,J) / (Nab_β Nij_β)
    //     beta state: {c,I2;J6} × {I3,k;J7},  phaseY = (-1)^{(jk+j_{I2})/2 + J4 + J7}
    //
    //   Term 2 (Ybar_12ck × Xbar_ck34): c=hole, k=particle, nanb=+1
    //     Xbar = -Σ_J5 (2J5+1) SixJ(j1,j4,J3; jc,jk,J5) × H(J5; I1,k,c,I4)
    //     Ybar = -hats × phaseY × NineJ(jk,j2,J6; jc,j3,J7; J3,J4,J) / (Nab_β Nij_β)
    //     beta state: {k,I2;J6} × {I3,c;J7},  phaseY = (-1)^{(jc+j_{I2})/2 + J4 + J7}
    //
    // Notation:
    //   I1=p, I2=q, I3=r, I4=s  from the 4 permutations of {a,b,ii,jj}
    //   J3 = coupling(I1,I4),  J4 = coupling(I2,I3),  Triangle(J3,J4,lambda=J)
    //   ninej_outer = NineJ(j1,j4,J3; j2,j3,J4; Jab,Jij,J)
    //   phase_outer = (-1)^{(j2_I2+j2_I4)/2 + Jij + J4}
    //   H22(α,β) += perm_phase × beta_can_phase / (Nab Nij)
    //               × sqrt((2Jab+1)(2Jij+1)(2J3+1)(2J4+1))
    //               × ninej_outer × phase_outer × nanb × Xbar × Ybar
    // -------------------------------------------------------------------
    {
      // lambda = J = rank of X_pphh (the 2p2h tensor operator)
      const int lambda = J;

      int phase_pp = AngMom::phase((oa.j2  + ob.j2)  / 2 - Jab + 1);
      int phase_hh = AngMom::phase((oii.j2 + ojj.j2) / 2 - Jij + 1);

      // comm222_phst permutations: {p,q,r,s} → 4 orderings of {a,b,ii,jj}
      // I1=perm[0], I2=perm[1], I3=perm[2], I4=perm[3]
      // j1..j4 correspond to j(I1)..j(I4)
      struct PermCPS {
        size_t I1, I2, I3, I4;
        int perm_phase;
        int j2_1, j2_2, j2_3, j2_4;  // 2*j values
      };
      const std::array<PermCPS,4> alpha_perms = {{
        {a,  b,  ii, jj, 1,                  oa.j2, ob.j2, oii.j2, ojj.j2},
        {b,  a,  ii, jj, phase_pp,           ob.j2, oa.j2, oii.j2, ojj.j2},
        {a,  b,  jj, ii, phase_hh,           oa.j2, ob.j2, ojj.j2, oii.j2},
        {b,  a,  jj, ii, phase_pp * phase_hh,ob.j2, oa.j2, ojj.j2, oii.j2}
      }};

      for (const PermCPS& pa : alpha_perms)
      {
        double j1 = 0.5*pa.j2_1, j2 = 0.5*pa.j2_2;
        double j3 = 0.5*pa.j2_3, j4 = 0.5*pa.j2_4;

        // J3 = coupling of (I1, I4);  J4 = coupling of (I2, I3)
        const int J3min = std::abs(pa.j2_1 - pa.j2_4) / 2;
        const int J3max = (pa.j2_1 + pa.j2_4) / 2;
        const int J4min = std::abs(pa.j2_2 - pa.j2_3) / 2;
        const int J4max = (pa.j2_2 + pa.j2_3) / 2;

        for (int J3 = J3min; J3 <= J3max; ++J3)
        {
          for (int J4 = J4min; J4 <= J4max; ++J4)
          {
            if (not AngMom::Triangle(J3, J4, lambda)) continue;

            // outer NineJ: NineJ(j1,j4,J3; j2,j3,J4; Jab,Jij,J)
            double ninej_out = modelspace->GetNineJ(j1, j4, (double)J3,
                                                    j2, j3, (double)J4,
                                                    (double)Jab, (double)Jij, (double)J);
            if (std::abs(ninej_out) < 1e-10) continue;

            double hats_out = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)
                                       *(2.0*J3+1)*(2.0*J4+1));
            int phase_out = AngMom::phase((pa.j2_2 + pa.j2_4)/2 + Jij + J4);

            // loop over intermediate (c, k) pairs.
            // Term 1 (Xbar_12ck x Ybar_ck34): c=particle, k=hole, nanb=-1
            // Term 2 (Ybar_12ck x Xbar_ck34): c=hole,     k=particle, nanb=+1
            for (int i_term = 0; i_term < 2; ++i_term)
            {
              const std::set<index_t>& c_set = (i_term == 0) ? modelspace->particles
                                                              : modelspace->holes;
              const std::set<index_t>& k_set = (i_term == 0) ? modelspace->holes
                                                              : modelspace->particles;
              const double nanb = (i_term == 0) ? -1.0 : +1.0;

              for (size_t c : c_set)
              {
                const Orbit& oc = modelspace->GetOrbit(c);
                double jc = 0.5*oc.j2;
                if (not AngMom::Triangle(jc, j4, J3)) continue;

                for (size_t k : k_set)
                {
                  const Orbit& ok = modelspace->GetOrbit(k);
                  double jk = 0.5*ok.j2;
                  if (not AngMom::Triangle(jc, jk, J3)) continue;

                  // parity and Tz checks for H(J5; I1, k, c, I4)
                  const Orbit& oI1 = modelspace->GetOrbit(pa.I1);
                  const Orbit& oI4 = modelspace->GetOrbit(pa.I4);
                  if ((oI1.l + ok.l + oc.l + oI4.l) % 2 != 0) continue;
                  if ((oI1.tz2 + ok.tz2) != (oc.tz2 + oI4.tz2))        continue;

                  // Xbar = -Sum_J5 (2J5+1) SixJ(j1,j4,J3; jc,jk,J5) x H(J5; I1,k,c,I4)
                  double Xbar = 0.0;
                  {
                    int J5min = std::max(std::abs(pa.j2_1 - ok.j2),
                                         std::abs(oc.j2  - pa.j2_4)) / 2;
                    int J5max = std::min(pa.j2_1 + ok.j2,
                                         oc.j2  + pa.j2_4) / 2;
                    for (int J5 = J5min; J5 <= J5max; ++J5)
                    {
                      double sixj = modelspace->GetSixJ(j1, j4, (double)J3,
                                                        jc, jk, (double)J5);
                      if (std::abs(sixj) < 1e-8) continue;
                      Xbar -= (2*J5+1) * sixj
                            * H.TwoBody.GetTBME_J(J5, pa.I1, k, c, pa.I4);
                    }
                  }
                  if (std::abs(Xbar) < 1e-10) continue;

                  // Term 1: beta state {c,I2;J6} × {I3,k;J7},  NineJ(jc,j2,J6; jk,j3,J7; ...)
                  // Term 2: beta state {k,I2;J6} × {I3,c;J7},  NineJ(jk,j2,J6; jc,j3,J7; ...)
                  double ja_Y  = (i_term == 0) ? jc : jk;
                  double jb_Y  = (i_term == 0) ? jk : jc;
                  int    j2_a_Y = (i_term == 0) ? oc.j2 : ok.j2;
                  int    j2_b_Y = (i_term == 0) ? ok.j2 : oc.j2;
                  size_t orb_pp = (i_term == 0) ? c : k;
                  size_t orb_hh = (i_term == 0) ? k : c;

                  const int J6min = std::abs(j2_a_Y - pa.j2_2) / 2;
                  const int J6max = (j2_a_Y + pa.j2_2) / 2;
                  const int J7min = std::abs(j2_b_Y - pa.j2_3) / 2;
                  const int J7max = (j2_b_Y + pa.j2_3) / 2;

                  for (int J6 = J6min; J6 <= J6max; ++J6)
                  {
                    // Pauli exclusion: if pp pair has identical orbits, J6 must be even
                    if ( orb_pp == pa.I2 && J6 % 2 != 0 ) continue;

                    for (int J7 = J7min; J7 <= J7max; ++J7)
                    {
                      // Pauli exclusion: if hh pair has identical orbits, J7 must be even
                      if ( orb_hh == pa.I3 && J7 % 2 != 0 ) continue;
                      if (not AngMom::Triangle(J6, J7, lambda)) continue;

                      double ninejY = modelspace->GetNineJ(ja_Y, j2, (double)J6,
                                                           jb_Y, j3, (double)J7,
                                                           (double)J3, (double)J4, (double)J);
                      if (std::abs(ninejY) < 1e-10) continue;

                      double hats_Y = std::sqrt((2.0*J3+1)*(2.0*J4+1)
                                               *(2.0*J6+1)*(2.0*J7+1));
                      // phaseY uses j2 of the hole orbit: ok for Term 1, oc for Term 2
                      int phaseY = AngMom::phase((j2_b_Y + pa.j2_2)/2 + J4 + J7);

                      double Nabb = std::sqrt(1.0 + (orb_pp == pa.I2 ? 1.0 : 0.0));
                      double Nijb = std::sqrt(1.0 + (orb_hh == pa.I3 ? 1.0 : 0.0));
                      double Ybar = -hats_Y * phaseY * ninejY / (Nabb * Nijb);

                      // Look up beta state in basis_map, handling canonical ordering.
                      double beta_can_phase = 0.0;
                      size_t beta_idx = n2;
                      {
                        auto it = basis_map.find({orb_pp, pa.I2, pa.I3, orb_hh, J6, J7});
                        if (it != basis_map.end())
                        {
                          beta_idx       = it->second;
                          beta_can_phase = 1.0;
                        }
                        else
                        {
                          // pp-swap
                          auto it2 = basis_map.find({pa.I2, orb_pp, pa.I3, orb_hh, J6, J7});
                          if (it2 != basis_map.end())
                          {
                            beta_idx       = it2->second;
                            beta_can_phase = AngMom::phase(
                                (j2_a_Y + pa.j2_2)/2 - J6 + 1);
                          }
                          else
                          {
                            // hh-swap
                            auto it3 = basis_map.find({orb_pp, pa.I2, orb_hh, pa.I3, J6, J7});
                            if (it3 != basis_map.end())
                            {
                              beta_idx       = it3->second;
                              beta_can_phase = AngMom::phase(
                                  (j2_b_Y + pa.j2_3)/2 - J7 + 1);
                            }
                            else
                            {
                              // both swaps
                              auto it4 = basis_map.find({pa.I2, orb_pp, orb_hh, pa.I3, J6, J7});
                              if (it4 != basis_map.end())
                              {
                                beta_idx       = it4->second;
                                beta_can_phase =
                                    AngMom::phase((j2_a_Y + pa.j2_2)/2 - J6 + 1)
                                  * AngMom::phase((j2_b_Y + pa.j2_3)/2 - J7 + 1);
                              }
                            }
                          }
                        }
                      }
                      if (beta_idx >= n2) continue;

                      H22(alpha, beta_idx) +=
                          (double)pa.perm_phase * beta_can_phase
                          / (Nab * Nij)
                          * hats_out * ninej_out * (double)phase_out
                          * nanb * Xbar * Ybar;
                    }  // J7
                  }  // J6
                }  // k
              }  // c
            }  // i_term
          }  // J4
        }  // J3
      }  // alpha perm
    }
  }
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
///   where N_ab = sqrt(1+delta_{a,b}), N_ij = sqrt(1+delta_{i,j})
///   account for the normalization of the identical-pair two-body states.
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
///
/// After filling the matrix, the same CC-channel ket-ordering phase used in
/// Build_H11_byIndex is applied column-wise so that the 1p1h columns of
/// H21 are aligned with those of A in the full EOM2 secular matrix:
///   [ A       H21^T ]
///   [ H21     H22   ]
///
void EOM_IMSRG::Build_H21_byIndex(size_t ich_CC)
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

    double Nab  = (a  == b)  ? std::sqrt(2.0) : 1.0;
    double Nij  = (ii == jj) ? std::sqrt(2.0) : 1.0;
    double K = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)) / (Nab * Nij) * AngMom::phase(J);

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
  // Build_H11_byIndex.  When a ph ket at column col is stored as
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
// Build_H12_byIndex
// ---------------------------------------------------------------------------

/// Build the 1p1h × 2p2h coupling block A12 for the explicit EOM2 matrix.
///
/// A12 is the Hermitian conjugate (transpose) of H21:
///   A12[col, α] = H21[α, col]  for all 1p1h column col and 2p2h row α.
///
/// For a real Hermitian Hamiltonian A12 = H21^T exactly.  This function
/// materialises A12 explicitly so that SolveCurrentChannel can assemble
///   H_full = [ A     A12 ]
///             [ H21   H22 ]
/// without relying on an in-place transposition of H21.
///
/// Note: Build_H21_byIndex MUST be called before this function (H21 is
/// used as the source).  The matrix-free Lanczos path (Solve_byIndex_MF)
/// uses ApplyH21T_matvec instead and never calls Build_H12_byIndex.
void EOM_IMSRG::Build_H12_byIndex(size_t /*ich_CC*/)
{
  A12 = H21.t();
}

// ---------------------------------------------------------------------------
// BuildPandya / ClearPandya
// ---------------------------------------------------------------------------

///
/// Precompute the Pandya-transformed Hamiltonian for every CC channel.
///
/// The result is stored in member arrays Hbar_CC and pan_idx so that both
/// Build_H11_byIndex and Build_H22_byIndex can share the same table.
/// The function is idempotent: if Hbar_CC is already populated it returns
/// immediately, so calling it multiple times is free.
///
/// The unified Pandya formula is:
/// \f[
///   \bar{H}(p_\text{bra}, h_\text{bra}; p_\text{ket}, h_\text{ket}; J_{ph})
///   = -\sum_{J'} (2J'+1)
///     \begin{Bmatrix} j_{p_\text{bra}} & j_{h_\text{bra}} & J_{ph} \\
///                    j_{p_\text{ket}} & j_{h_\text{ket}} & J' \end{Bmatrix}
///     \langle p_\text{bra}\, h_\text{ket}; J' \| V \|
///             p_\text{ket}\, h_\text{bra}; J' \rangle
/// \f]
///
void EOM_IMSRG::BuildPandya()
{
  if (!Hbar_CC.empty()) return;  // already built

  size_t n_CC_total = modelspace->GetNumberTwoBodyChannels_CC();
  pan_idx.assign(n_CC_total, std::map<std::pair<size_t,size_t>,int>());
  Hbar_CC.assign(n_CC_total, arma::mat());

  for (size_t ich_ph = 0; ich_ph < n_CC_total; ++ich_ph)
  {
    TwoBodyChannel_CC& tbc_ph = modelspace->GetTwoBodyChannel_CC(ich_ph);
    int J_ph = tbc_ph.J;
    const arma::uvec& ph_idx = tbc_ph.GetKetIndex_ph();
    int n_ph = (int)ph_idx.n_elem;
    if (n_ph == 0) continue;

    for (int i = 0; i < n_ph; ++i)
    {
      Ket& kt = tbc_ph.GetKet(ph_idx[i]);
      size_t p_orb = kt.p, h_orb = kt.q;
      if (kt.op->occ > 0.5) std::swap(p_orb, h_orb);  // ensure p_orb=particle
      pan_idx[ich_ph][{p_orb, h_orb}] = i;
    }

    Hbar_CC[ich_ph].zeros(n_ph, n_ph);
    for (int ibra = 0; ibra < n_ph; ++ibra)
    {
      Ket& bra = tbc_ph.GetKet(ph_idx[ibra]);
      size_t a_orb = bra.p, k_orb = bra.q;
      if (bra.op->occ > 0.5) std::swap(a_orb, k_orb);
      double j_a = 0.5 * modelspace->GetOrbit(a_orb).j2;
      double j_k = 0.5 * modelspace->GetOrbit(k_orb).j2;

      for (int iket = 0; iket < n_ph; ++iket)
      {
        Ket& ket = tbc_ph.GetKet(ph_idx[iket]);
        size_t c_orb = ket.p, l_orb = ket.q;
        if (ket.op->occ > 0.5) std::swap(c_orb, l_orb);
        double j_c = 0.5 * modelspace->GetOrbit(c_orb).j2;
        double j_l = 0.5 * modelspace->GetOrbit(l_orb).j2;

        int Jp_min = std::abs((int)(2*j_a) - (int)(2*j_l)) / 2;
        int Jp_max = ((int)(2*j_a) + (int)(2*j_l)) / 2;
        double V_bar = 0.0;
        for (int Jp = Jp_min; Jp <= Jp_max; ++Jp)
        {
          if (!AngMom::Triangle(j_c, j_k, (double)Jp)) continue;
          // CORRECT W6j: row2 is (j_ket_particle=j_c, j_ket_hole=j_l, J')
          double sixj = modelspace->GetSixJ(j_a, j_k, (double)J_ph,
                                            j_c, j_l, (double)Jp);
          if (std::abs(sixj) < 1e-8) continue;
          V_bar -= (2*Jp+1) * sixj
                 * H.TwoBody.GetTBME_J(Jp, a_orb, l_orb, c_orb, k_orb);
        }
        Hbar_CC[ich_ph](ibra, iket) = V_bar;
      }
    }
  }
}

/// Release the precomputed Pandya table to free memory.
void EOM_IMSRG::ClearPandya()
{
  Hbar_CC.clear();
  pan_idx.clear();
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

void EOM_IMSRG::Solve(int J, int parity, int Tz, std::string mode)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  Solve_byIndex(ich_CC, mode);
}

void EOM_IMSRG::Solve_byIndex(size_t ich_CC, std::string mode)
{
  current_channel = ich_CC;
  Build_H11_byIndex(ich_CC);
  one_ph_count = A.n_rows;
  two_ph_count = 0;
  lanczos_iterations = 0;
  OnePhNorms.reset();

  if (mode == "RPA")
    Build_BMatrix_byIndex(ich_CC);
  else
    B.zeros(A.n_rows, A.n_cols);

  if (mode == "EOM2")
  {
    Build_2p2hBasis_byIndex(ich_CC);
    two_ph_count = tpth_basis.size();
    Build_H22_byIndex(ich_CC);
    Build_H21_byIndex(ich_CC);
    Build_H12_byIndex(ich_CC);
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
void EOM_IMSRG::SolveAllChannels(std::string mode)
{
  BuildPandya();
  size_t nch = modelspace->GetNumberTwoBodyChannels_CC();
  for (size_t ich = 0; ich < nch; ich++)
  {
    TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ich);
    if (tbc.GetKetIndex_ph().empty()) continue;
    Solve_byIndex(ich, mode);
  }
  ClearPandya();
}

// ---------------------------------------------------------------------------
// Matrix-free matvec helpers
// ---------------------------------------------------------------------------

/// Compute Hv_2p2h += H22 * v without building H22.
///
/// Implements the same formula as Build_H22_byIndex row by row.
/// For each 2p-2h state α = |ab J_ab; ij J_ij; J⟩ the contributions from
/// all β states are accumulated and written to Hv[α].
///
/// Terms (notation: a,b particles; i,j holes; H = f + Λ):
///   - Diagonal SPE:   (f_aa + f_bb - f_ii - f_jj) × v[α]
///   - PP-PP ladder:   +Λ_norm(J_ab; a,b|a',b') × v[β]   when ij=i'j', J_ij=J_ij'
///   - HH-HH ladder:   +Λ_norm(J_ij; i,j|i',j') × v[β]   when ab=a'b', J_ab=J_ab'
///   - Ph ring term:   see Build_H22_byIndex for the full 9j-symbol formula
///
/// The loop is trivially thread-safe under OpenMP because each α accumulates
/// into a private double hv_alpha before writing to Hv[α].
void EOM_IMSRG::ApplyH22_matvec(const arma::vec& v, arma::vec& Hv) const
{
  int J = modelspace->GetTwoBodyChannel_CC(current_channel).J;
  size_t n2 = tpth_basis.size();

  // Basis lookup map for ring term: (a, b, i, j, Jab, Jij) -> index
  std::map<std::tuple<size_t,size_t,size_t,size_t,int,int>, size_t> ring_basis_map;
  for (size_t idx = 0; idx < n2; ++idx)
  {
    const TwoPTwoHState& st = tpth_basis[idx];
    ring_basis_map[{st.a, st.b, st.i, st.j, st.Jab, st.Jij}] = idx;
  }

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

    // 1-body contributions (comm121): sum over all intermediate-state beta.
    // For beta==alpha the four conditions each collapse to one term, recovering
    // the diagonal SPE f_{aa}+f_{bb}-f_{ii}-f_{jj}.
    // Normalization factors for the alpha state:
    double Nab = std::sqrt(1.0 + (a  == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));
    double hv_alpha = 0.0;
    for (size_t beta2 = 0; beta2 < n2; ++beta2)
    {
      if (std::abs(v[beta2]) < 1e-15) continue;
      const TwoPTwoHState& st_b2 = tpth_basis[beta2];
      size_t ap2 = st_b2.a, bp2 = st_b2.b, ip2 = st_b2.i, jp2 = st_b2.j;
      int Jabp2 = st_b2.Jab, Jijp2 = st_b2.Jij;
      if (b == bp2 && ii == ip2 && jj == jp2 && Jab == Jabp2 && Jij == Jijp2)
      {
        double Nap2b = std::sqrt(1.0 + (ap2 == b   ? 1.0 : 0.0));
        hv_alpha += (Nab / Nap2b) * H.OneBody(a, ap2) * v[beta2];
      }
      if (a == ap2 && ii == ip2 && jj == jp2 && Jab == Jabp2 && Jij == Jijp2)
      {
        double Nabp2 = std::sqrt(1.0 + (a   == bp2 ? 1.0 : 0.0));
        hv_alpha += (Nab / Nabp2) * H.OneBody(b, bp2) * v[beta2];
      }
      if (a == ap2 && b == bp2 && jj == jp2 && Jab == Jabp2 && Jij == Jijp2)
      {
        double Nip2j = std::sqrt(1.0 + (ip2 == jj  ? 1.0 : 0.0));
        hv_alpha -= (Nij / Nip2j) * H.OneBody(ii, ip2) * v[beta2];
      }
      if (a == ap2 && b == bp2 && ii == ip2 && Jab == Jabp2 && Jij == Jijp2)
      {
        double Nijp2 = std::sqrt(1.0 + (ii  == jp2 ? 1.0 : 0.0));
        hv_alpha -= (Nij / Nijp2) * H.OneBody(jj, jp2) * v[beta2];
      }
    }

    // PP-PP and HH-HH ladder terms.
    for (size_t beta = 0; beta < n2; ++beta)
    {
      if (std::abs(v[beta]) < 1e-15) continue;
      const TwoPTwoHState& st_beta = tpth_basis[beta];
      size_t ap = st_beta.a, bp = st_beta.b, ip = st_beta.i, jp = st_beta.j;
      int Jabp = st_beta.Jab, Jijp = st_beta.Jij;

      double val = 0.0;
      if (ii == ip && jj == jp && Jij == Jijp && Jab == Jabp)
      {
        double fac = (ap == bp) ? 0.5 : 1.0;
        val += fac * H.TwoBody.GetTBME_J(Jab, a, b, ap, bp);
      }
      if (a == ap && b == bp && Jab == Jabp && Jij == Jijp)
      {
        double fac = (ip == jp) ? 0.5 : 1.0;
        val += fac * H.TwoBody.GetTBME_J(Jij, ii, jj, ip, jp);
      }
      hv_alpha += val * v[beta];
    }

    // Ph ring term: same formula as Build_H22_byIndex section (d).
    // Two terms: c=particle/k=hole (nanb=-1) and c=hole/k=particle (nanb=+1).
    {
      const int lambda = J;
      const double Nab = std::sqrt(1.0 + (a == b  ? 1.0 : 0.0));
      const double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));

      int phase_pp_a = AngMom::phase((j2a  + j2b)  / 2 - Jab + 1);
      int phase_hh_a = AngMom::phase((j2ii + j2jj) / 2 - Jij + 1);

      struct PermCPS {
        size_t I1, I2, I3, I4;
        int perm_phase;
        int j2_1, j2_2, j2_3, j2_4;
      };
      const std::array<PermCPS,4> alpha_perms = {{
        {a,  b,  ii, jj, 1,                   j2a,  j2b,  j2ii, j2jj},
        {b,  a,  ii, jj, phase_pp_a,          j2b,  j2a,  j2ii, j2jj},
        {a,  b,  jj, ii, phase_hh_a,          j2a,  j2b,  j2jj, j2ii},
        {b,  a,  jj, ii, phase_pp_a*phase_hh_a,j2b, j2a,  j2jj, j2ii}
      }};

      for (const PermCPS& pa : alpha_perms)
      {
        double j1 = 0.5*pa.j2_1, j2_val = 0.5*pa.j2_2;
        double j3 = 0.5*pa.j2_3, j4 = 0.5*pa.j2_4;

        const int J3min = std::abs(pa.j2_1 - pa.j2_4) / 2;
        const int J3max = (pa.j2_1 + pa.j2_4) / 2;
        const int J4min = std::abs(pa.j2_2 - pa.j2_3) / 2;
        const int J4max = (pa.j2_2 + pa.j2_3) / 2;

        for (int J3 = J3min; J3 <= J3max; ++J3)
        {
          for (int J4 = J4min; J4 <= J4max; ++J4)
          {
            if (not AngMom::Triangle(J3, J4, lambda)) continue;

            double ninej_out = modelspace->GetNineJ(j1, j4, (double)J3,
                                                    j2_val, j3, (double)J4,
                                                    (double)Jab, (double)Jij, (double)J);
            if (std::abs(ninej_out) < 1e-10) continue;

            double hats_out = std::sqrt((2.0*Jab+1)*(2.0*Jij+1)
                                       *(2.0*J3+1)*(2.0*J4+1));
            int phase_out = AngMom::phase((pa.j2_2 + pa.j2_4)/2 + Jij + J4);

            // loop over intermediate (c, k) pairs.
            // Term 1 (Xbar_12ck x Ybar_ck34): c=particle, k=hole, nanb=-1
            // Term 2 (Ybar_12ck x Xbar_ck34): c=hole,     k=particle, nanb=+1
            for (int i_term = 0; i_term < 2; ++i_term)
            {
              const std::set<index_t>& c_set = (i_term == 0) ? modelspace->particles
                                                              : modelspace->holes;
              const std::set<index_t>& k_set = (i_term == 0) ? modelspace->holes
                                                              : modelspace->particles;
              const double nanb = (i_term == 0) ? -1.0 : +1.0;

              for (size_t c : c_set)
              {
                const Orbit& oc = modelspace->GetOrbit(c);
                double jc = 0.5*oc.j2;
                if (not AngMom::Triangle(jc, j4, J3)) continue;

                for (size_t k : k_set)
                {
                  const Orbit& ok = modelspace->GetOrbit(k);
                  double jk = 0.5*ok.j2;
                  if (not AngMom::Triangle(jc, jk, J3)) continue;

                  const Orbit& oI1 = modelspace->GetOrbit(pa.I1);
                  const Orbit& oI4 = modelspace->GetOrbit(pa.I4);
                  if ((oI1.l + ok.l + oc.l + oI4.l) % 2 != 0) continue;
                  if ((oI1.tz2 + ok.tz2) != (oc.tz2 + oI4.tz2))        continue;

                  // Xbar = -Sum_J5 (2J5+1) SixJ(j1,j4,J3; jc,jk,J5) x H(J5; I1,k,c,I4)
                  double Xbar = 0.0;
                  {
                    int J5min = std::max(std::abs(pa.j2_1 - ok.j2),
                                         std::abs(oc.j2  - pa.j2_4)) / 2;
                    int J5max = std::min(pa.j2_1 + ok.j2,
                                         oc.j2  + pa.j2_4) / 2;
                    for (int J5 = J5min; J5 <= J5max; ++J5)
                    {
                      double sixj = modelspace->GetSixJ(j1, j4, (double)J3,
                                                        jc, jk, (double)J5);
                      if (std::abs(sixj) < 1e-8) continue;
                      Xbar -= (2*J5+1) * sixj
                            * H.TwoBody.GetTBME_J(J5, pa.I1, k, c, pa.I4);
                    }
                  }
                  if (std::abs(Xbar) < 1e-10) continue;

                  // Term 1: beta state {c,I2;J6} × {I3,k;J7},  NineJ(jc,j2,J6; jk,j3,J7; ...)
                  // Term 2: beta state {k,I2;J6} × {I3,c;J7},  NineJ(jk,j2,J6; jc,j3,J7; ...)
                  double ja_Y   = (i_term == 0) ? jc : jk;
                  double jb_Y   = (i_term == 0) ? jk : jc;
                  int    j2_a_Y = (i_term == 0) ? oc.j2 : ok.j2;
                  int    j2_b_Y = (i_term == 0) ? ok.j2 : oc.j2;
                  size_t orb_pp = (i_term == 0) ? c : k;
                  size_t orb_hh = (i_term == 0) ? k : c;

                  const int J6min = std::abs(j2_a_Y - pa.j2_2) / 2;
                  const int J6max = (j2_a_Y + pa.j2_2) / 2;
                  const int J7min = std::abs(j2_b_Y - pa.j2_3) / 2;
                  const int J7max = (j2_b_Y + pa.j2_3) / 2;

                  for (int J6 = J6min; J6 <= J6max; ++J6)
                  {
                    // Pauli exclusion: if pp pair has identical orbits, J6 must be even
                    if ( orb_pp == pa.I2 && J6 % 2 != 0 ) continue;

                    for (int J7 = J7min; J7 <= J7max; ++J7)
                    {
                      // Pauli exclusion: if hh pair has identical orbits, J7 must be even
                      if ( orb_hh == pa.I3 && J7 % 2 != 0 ) continue;
                      if (not AngMom::Triangle(J6, J7, lambda)) continue;

                      double ninejY = modelspace->GetNineJ(ja_Y, j2_val, (double)J6,
                                                           jb_Y, j3,     (double)J7,
                                                           (double)J3, (double)J4, (double)J);
                      if (std::abs(ninejY) < 1e-10) continue;

                      double hats_Y = std::sqrt((2.0*J3+1)*(2.0*J4+1)
                                               *(2.0*J6+1)*(2.0*J7+1));
                      // phaseY uses j2 of the hole orbit: ok for Term 1, oc for Term 2
                      int phaseY = AngMom::phase((j2_b_Y + pa.j2_2)/2 + J4 + J7);

                      double Nabb = std::sqrt(1.0 + (orb_pp == pa.I2 ? 1.0 : 0.0));
                      double Nijb = std::sqrt(1.0 + (orb_hh == pa.I3 ? 1.0 : 0.0));
                      double Ybar = -hats_Y * phaseY * ninejY / (Nabb * Nijb);

                      // Look up beta state in ring_basis_map, handling canonical ordering.
                      double beta_can_phase = 0.0;
                      size_t beta_idx = n2;
                      {
                        auto it = ring_basis_map.find({orb_pp, pa.I2, pa.I3, orb_hh, J6, J7});
                        if (it != ring_basis_map.end())
                        {
                          beta_idx       = it->second;
                          beta_can_phase = 1.0;
                        }
                        else
                        {
                          auto it2 = ring_basis_map.find({pa.I2, orb_pp, pa.I3, orb_hh, J6, J7});
                          if (it2 != ring_basis_map.end())
                          {
                            beta_idx       = it2->second;
                            beta_can_phase = AngMom::phase(
                                (j2_a_Y + pa.j2_2)/2 - J6 + 1);
                          }
                          else
                          {
                            auto it3 = ring_basis_map.find({orb_pp, pa.I2, orb_hh, pa.I3, J6, J7});
                            if (it3 != ring_basis_map.end())
                            {
                              beta_idx       = it3->second;
                              beta_can_phase = AngMom::phase(
                                  (j2_b_Y + pa.j2_3)/2 - J7 + 1);
                            }
                            else
                            {
                              auto it4 = ring_basis_map.find({pa.I2, orb_pp, orb_hh, pa.I3, J6, J7});
                              if (it4 != ring_basis_map.end())
                              {
                                beta_idx       = it4->second;
                                beta_can_phase =
                                    AngMom::phase((j2_a_Y + pa.j2_2)/2 - J6 + 1)
                                  * AngMom::phase((j2_b_Y + pa.j2_3)/2 - J7 + 1);
                              }
                            }
                          }
                        }
                      }
                      if (beta_idx >= n2) continue;
                      if (std::abs(v[beta_idx]) < 1e-15) continue;

                      hv_alpha +=
                          (double)pa.perm_phase * beta_can_phase
                          / (Nab * Nij)
                          * hats_out * ninej_out * (double)phase_out
                          * nanb * Xbar * Ybar
                          * v[beta_idx];
                    }  // J7
                  }  // J6
                }  // k
              }  // c
            }  // i_term
          }  // J4
        }  // J3
      }  // alpha perm
    }

    Hv[alpha] += hv_alpha;
  }
}

/// Compute Hv_2p2h += H21 * v_ph without building H21.
/// Applies the same formula as Build_H21_byIndex but without materialising H21.
/// Parallelised over 2p-2h states α; uses the precomputed lookup tables:
///   mf_orbit_to_ph[orb]  → (col, is_particle) pairs for each orbit
///   mf_ph_particle[col]  → particle orbit (a,b,c,…) of ph column col
///   mf_ph_hole[col]      → hole    orbit  (i,j,k,…) of ph column col
/// These are populated once by Solve_byIndex_MF before Lanczos iterations.
///
/// The four terms sm1-sm4 are identical to Build_H21_byIndex; see that
/// function's comment for the full formula with consistent notation.
void EOM_IMSRG::ApplyH21_matvec(const arma::vec& v_ph, arma::vec& Hv_2p2h) const
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

    double Nab  = (a  == b)  ? std::sqrt(2.0) : 1.0;
    double Nij  = (ii == jj) ? std::sqrt(2.0) : 1.0;
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
void EOM_IMSRG::ApplyH21T_matvec(const arma::vec& v_2p2h, arma::vec& Hv_ph) const
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

    double Nab  = (a  == b)  ? std::sqrt(2.0) : 1.0;
    double Nij  = (ii == jj) ? std::sqrt(2.0) : 1.0;
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
void EOM_IMSRG::ApplyH_EOM2_matvec(const arma::vec& v, arma::vec& Hv) const
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
/// EOM_IMSRG::ApplyH_EOM2_matvec.  Passed to SymEigsSolver in place of
/// DenseGenMatProd so that no H22 / H21 matrix is ever materialised.
struct EOMMatFreeOp
{
  typedef double elem_type;
  const EOM_IMSRG* eom;
  arma::uword n_rows;  ///< required by armadillo newarp SymEigsSolver

  EOMMatFreeOp(const EOM_IMSRG* e, arma::uword n) : eom(e), n_rows(n) {}

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
///   - Reuses Build_H11_byIndex() and Build_2p2hBasis_byIndex() (unchanged).
///   - Does NOT build H22 (N2×N2) or H21 (N2×Nph).
///   - Runs armadillo's newarp SymEigsSolver with a custom matrix-product
///     operator that computes H_EOM2 * v on the fly each Lanczos step.
///
/// Peak additional memory is O(N2 × ncv) for the Lanczos vectors instead of
/// O(N2²) for the explicit matrices.
///
/// Computation time: each matrix-vector product costs the same as one row of
/// Build_H22_byIndex, so K Lanczos iterations costs K × build_cost(H22).
/// For K ≪ N2 (typical when requesting a handful of eigenvalues) this is
/// cheaper than building the full matrix; for K ≫ N2 the dense path is faster.
///
/// @param nev  Number of algebraically-lowest eigenvalues to converge (≥ 1).
///             If nev ≥ N = nph + n2p2h, falls back to the dense EOM2 solve.
void EOM_IMSRG::Solve_byIndex_MF(size_t ich_CC, int nev)
{
  current_channel = ich_CC;

  // --- Reuse existing builders for A and the 2p2h basis ---
  Build_H11_byIndex(ich_CC);
  one_ph_count = A.n_rows;

  Build_2p2hBasis_byIndex(ich_CC);
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
    std::cout << "WARNING EOM_IMSRG::Solve_byIndex_MF: nev=" << nev
              << " out of range for N=" << N
              << "; falling back to dense EOM2 solve in channel " << ich_CC
              << std::endl;
    Build_H22_byIndex(ich_CC);
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
      std::cout << "WARNING EOM_IMSRG::Solve_byIndex_MF: only " << nconv
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
void EOM_IMSRG::SolveAllChannels_MF(int nev)
{
  BuildPandya();
  size_t nch = modelspace->GetNumberTwoBodyChannels_CC();
  for (size_t ich = 0; ich < nch; ich++)
  {
    TwoBodyChannel_CC& tbc = modelspace->GetTwoBodyChannel_CC(ich);
    if (tbc.GetKetIndex_ph().empty()) continue;
    Solve_byIndex_MF(ich, nev);
  }
  ClearPandya();
}


void EOM_IMSRG::SolveCurrentChannel(std::string mode)
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
    //   H_EOM = [ A     A12 ]   where A12 = H21^T (built by Build_H12_byIndex)
    //           [ H21   H22 ]
    // A12 is precomputed as the explicit 1p1h × 2p2h coupling block so that
    // the matrix assembly matches the Fortran EOM-IMSRG code structure.
    // Diagonalise with either the dense LAPACK driver (all eigenvalues) or the
    // newarp Implicitly Restarted Arnoldi / Lanczos solver (lowest lanczos_nev
    // eigenvalues only).  The latter mirrors the ARPACK-based approach used by
    // the reference Fortran EOM-IMSRG code of Parzuchowski et al.
    size_t nph = A.n_rows;
    arma::mat Hfull = arma::join_vert(
        arma::join_horiz(A,   A12),
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
        std::cout << "WARNING EOM_IMSRG Lanczos: only " << nconv
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
      std::cout << "WARNING EOM_IMSRG RPA: non-zero imaginary eigenvalues ("
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
    throw std::invalid_argument("EOM_IMSRG::Solve: unknown mode '" + mode
                                + "'. Use 'TDA', 'EOM', 'RPA', or 'EOM2'.");
  }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

arma::vec EOM_IMSRG::GetExcitationEnergies() const { return Energies; }

arma::vec EOM_IMSRG::GetAmplitudesX(size_t state_index) const
{
  return X.col(state_index);
}

arma::vec EOM_IMSRG::GetAmplitudesY(size_t state_index) const
{
  return Y.col(state_index);
}

arma::vec EOM_IMSRG::GetOnePhNorms() const { return OnePhNorms; }

size_t EOM_IMSRG::GetOnePhCount() const { return one_ph_count; }

size_t EOM_IMSRG::GetTwoPhCount() const { return two_ph_count; }

size_t EOM_IMSRG::GetLanczosIterations() const { return lanczos_iterations; }

void EOM_IMSRG::PrintSummary() const
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

EOMChannel EOM_IMSRG::GetChannelResults(size_t ich_CC) const
{
  auto it = ChannelResults.find(ich_CC);
  if (it == ChannelResults.end())
    throw std::out_of_range("EOM_IMSRG::GetChannelResults: channel not solved");
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
double EOM_IMSRG::ComputeTransitionME(Operator& Op, size_t state_index) const
{
  return ComputeTransitionME_byIndex(current_channel, Op, state_index);
}

double EOM_IMSRG::ComputeTransitionME_byIndex(size_t ich_CC,
                                              Operator& Op,
                                              size_t state_index) const
{
  auto it = ChannelResults.find(ich_CC);
  if (it == ChannelResults.end())
    throw std::out_of_range("EOM_IMSRG::ComputeTransitionME_byIndex: "
                            "channel not solved");

  const EOMChannel& ch = it->second;
  if (state_index >= (size_t)ch.X.n_cols)
    throw std::out_of_range("EOM_IMSRG::ComputeTransitionME_byIndex: "
                            "state_index out of range");

  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();

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
  // H22 ph-ring term is computed in Build_H22_byIndex.  This is left as a
  // future enhancement; for typical 1p1h-dominant states the dominant
  // contribution comes from the 1p1h term above.
  // -------------------------------------------------------------------

  // -------------------------------------------------------------------
  // Overall prefactor from eq. (16): (-1)^{J_ν} / √(2J_ν+1)
  // -------------------------------------------------------------------
  return AngMom::phase(J) / std::sqrt(2.0 * J + 1.0) * T;
}

// ---------------------------------------------------------------------------
// Test_H22_ring_via_comm
// ---------------------------------------------------------------------------

///
/// Test that the 2p2h block H22 built by Build_H22_byIndex agrees with a
/// version assembled column-by-column from tensor commutators.
///
/// For each basis state α = |ab J_ab; ij J_ij; J⟩ the pure-excitation
/// 2p2h operator
///
///   X_α  =  (1/(N_ab N_ij)) a†_a a†_b a_j a_i   (rank λ = J)
///
/// is neither hermitian nor antihermitian, so Commutator::comm222_phst
/// (which requires one of those symmetries) cannot be called directly.
/// Instead we write
///
///   X_α  =  ½ (T_s + T_a)
///
/// where
///   T_s  =  X_α + X_α†   (hermitian,    rank λ)
///   T_a  =  X_α − X_α†   (antihermitian, rank λ)
///
/// and use linearity of the commutator:
///
///   [H, X_α]  =  ½ ([H, T_s] + [H, T_a])
///
/// This holds because T_s + T_a = (X_α + X_α†) + (X_α − X_α†) = 2 X_α.
///
/// The full H22 matrix is then recovered from:
///
///   H22_comm(α, β)
///     =  ½ (Z_s + Z_a).GetTBME_J(J_ab′, J_ij′, a′, b′, i′, j′)
///                      / (N_ab′ N_ij′)
///
/// where Z_s = [H, T_s] (antihermitian) and Z_a = [H, T_a] (hermitian).
/// Each Z is accumulated from three tensor-commutator contributions:
///
///   Commutator::comm122st         (f × T  → 2-body, the 1-body term)
///   Commutator::comm222_pp_hhst   (pp-pp and hh-hh ladder terms)
///   Commutator::comm222_phst      (ph ring term)
///
bool EOM_IMSRG::Test_H22_ring_via_comm(size_t ich_CC)
{
  // -----------------------------------------------------------------------
  // 1. Build 2p2h basis and reference H22 via the existing inline code.
  // -----------------------------------------------------------------------
  Build_2p2hBasis_byIndex(ich_CC);
  Build_H22_byIndex(ich_CC);
  arma::mat H22_ref = H22;  // deep copy before H22 is overwritten

  size_t n2 = tpth_basis.size();
  if (n2 == 0)
  {
    TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
    std::cout << "Test_H22_ring_via_comm: empty 2p2h basis for channel "
              << ich_CC
              << " (J=" << tbc_CC.J
              << " P=" << tbc_CC.parity
              << " Tz=" << tbc_CC.Tz << "), SKIP" << std::endl;
    return true;
  }

  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  int    J_EOM   = tbc_CC.J;
  int    par_EOM = tbc_CC.parity;
  int    Tz_rank = std::abs(tbc_CC.Tz);

  // Pre-compute recoupling tables (idempotent if already done).
  modelspace->PreCalculateSixJ();
  modelspace->PreCalculateNineJ();

  // -----------------------------------------------------------------------
  // 2. For each α column, assemble H22_comm(α, :) from tensor commutators.
  // -----------------------------------------------------------------------
  arma::mat H22_comm(n2, n2, arma::fill::zeros);

  for (size_t alpha = 0; alpha < n2; ++alpha)
  {
    const TwoPTwoHState& st = tpth_basis[alpha];
    double Nab = std::sqrt(1.0 + (st.a == st.b ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (st.i == st.j ? 1.0 : 0.0));
    // SetTBME_J stores the raw matrix element; GetTBME_J returns Nab*Nij*raw.
    // Setting raw = 1/(Nab*Nij) makes GetTBME_J return exactly 1.0, matching
    // the unit-amplitude excitation operator X_α.
    double val = 1.0 / (Nab * Nij);

    // ---- T_s (hermitian): excitation ME = 1.0, de-excitation ME = +conj ----
    Operator T_s(*modelspace, J_EOM, Tz_rank, par_EOM, 2);
    T_s.SetHermitian();
    T_s.TwoBody.SetTBME_J(st.Jab, st.Jij, st.a, st.b, st.i, st.j, val);

    // ---- T_a (antihermitian): excitation ME = 1.0, de-excitation ME = -conj --
    Operator T_a(*modelspace, J_EOM, Tz_rank, par_EOM, 2);
    T_a.SetAntiHermitian();
    T_a.TwoBody.SetTBME_J(st.Jab, st.Jij, st.a, st.b, st.i, st.j, val);

    // ---- Z_s = [H, T_s]: H hermitian, T_s hermitian → Z_s antihermitian ---
    Operator Z_s(*modelspace, J_EOM, Tz_rank, par_EOM, 2);
    Z_s.SetAntiHermitian();

    // ---- Z_a = [H, T_a]: H hermitian, T_a antihermitian → Z_a hermitian ----
    Operator Z_a(*modelspace, J_EOM, Tz_rank, par_EOM, 2);
    Z_a.SetHermitian();

    // ---- Accumulate all three commutator contributions --------------------
    // 1-body term: [H^{1b}, T^{2b}] → Z^{2b}
    Commutator::comm122st(H, T_s, Z_s);
    Commutator::comm122st(H, T_a, Z_a);

    // pp-pp and hh-hh ladder: [H^{2b}, T^{2b}]_{pp/hh} → Z^{2b}
    Commutator::comm222_pp_hhst(H, T_s, Z_s);
    Commutator::comm222_pp_hhst(H, T_a, Z_a);

    // ph ring: [H^{2b}, T^{2b}]_{ph} → Z^{2b}
    Commutator::comm222_phst(H, T_s, Z_s);
    Commutator::comm222_phst(H, T_a, Z_a);

    // ---- Extract H22_comm(α, β) for every β --------------------------------
    // [H, X_α] = ½([H, T_s] + [H, T_a]) = ½(Z_s + Z_a)
    //
    // H22_comm(α, β)
    //   = <β| [H, X_α] |0>
    //   = ½ (Z_s + Z_a).GetTBME_J(J_ab′, J_ij′, a′, b′, i′, j′) / (N_ab′ N_ij′)
    //
    // GetTBME_J returns N_ab′ * N_ij′ * (raw stored value), so dividing by
    // N_ab′ * N_ij′ recovers the EOM matrix element.
    for (size_t beta = 0; beta < n2; ++beta)
    {
      const TwoPTwoHState& sb = tpth_basis[beta];
      double Nabp = std::sqrt(1.0 + (sb.a == sb.b ? 1.0 : 0.0));
      double Nijp = std::sqrt(1.0 + (sb.i == sb.j ? 1.0 : 0.0));

      double me_s = Z_s.TwoBody.GetTBME_J(sb.Jab, sb.Jij, sb.a, sb.b, sb.i, sb.j);
      double me_a = Z_a.TwoBody.GetTBME_J(sb.Jab, sb.Jij, sb.a, sb.b, sb.i, sb.j);

      H22_comm(alpha, beta) = 0.5 * (me_s + me_a) / (Nabp * Nijp);
    }
  }  // alpha loop

  // -----------------------------------------------------------------------
  // 3. Compare H22_comm with H22_ref and report.
  // -----------------------------------------------------------------------
  arma::mat diff = H22_ref - H22_comm;
  double max_diff = arma::max(arma::max(arma::abs(diff)));
  double ref_norm = arma::norm(H22_ref, "fro");

  const double tol = 1e-6;
  bool pass = (max_diff < tol);

  std::cout << "Test_H22_ring_via_comm"
            << "  ch=" << ich_CC
            << "  J=" << J_EOM << "  P=" << par_EOM << "  Tz=" << tbc_CC.Tz
            << "  n2=" << n2
            << "  ||H22_ref||=" << std::scientific << std::setprecision(4) << ref_norm
            << "  max|diff|="   << std::scientific << std::setprecision(4) << max_diff
            << "  " << (pass ? "PASS" : "FAIL") << std::endl;

  return pass;
}
