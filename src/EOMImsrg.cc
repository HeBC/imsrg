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
#include <iostream>
#include <stdexcept>

// Tolerance for warning about non-zero imaginary parts of EOM eigenvalues
static const double EOM_IMAG_TOL = 1e-3;
// Tolerance below which we skip renormalization of an EOM amplitude pair
static const double EOM_NORM_TOL = 1e-10;

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

EOMImsrg::EOMImsrg()
  : modelspace(nullptr), current_channel(0), lanczos_nev(0)
{}

EOMImsrg::EOMImsrg(Operator& H_imsrg)
  : modelspace(H_imsrg.modelspace), H(H_imsrg), current_channel(0), lanczos_nev(0)
{}

// ---------------------------------------------------------------------------
// BuildAMatrix
// ---------------------------------------------------------------------------

/// Dispatch helper: look up the channel index then call BuildAMatrix_byIndex.
void EOMImsrg::BuildAMatrix(int J, int parity, int Tz)
{
  size_t ich_CC = modelspace->GetTwoBodyChannelIndex(J, parity, Tz);
  BuildAMatrix_byIndex(ich_CC);
}

///
/// Build the EOM A matrix (forward/TDA block) for the 1p-1h sector.
///
/// The matrix element is
/// \f[
///   A_{ai,bj}(J) = (\varepsilon_a - \varepsilon_i)\delta_{ab}\delta_{ij}
///                  - \sum_{J'} (2J'+1)
///                    \begin{Bmatrix} j_a & j_i & J \\ j_b & j_j & J' \end{Bmatrix}
///                    \langle a j; J' \| V \| b i; J' \rangle,
/// \f]
/// which is the Pandya transform of the antisymmetrised two-body interaction.
/// This is identical to the TDA/RPA A matrix, but evaluated with the
/// IMSRG-evolved \f$\tilde{H}\f$.
///
void EOMImsrg::BuildAMatrix_byIndex(size_t ich_CC)
{
  current_channel = ich_CC;
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph = ph_list.size();
  A.zeros(nph, nph);

  int Jph = tbc_CC.J;

  size_t I = 0;
  for (auto iket_ai : ph_list)
  {
    Ket& ket_ai = tbc_CC.GetKet(iket_ai);
    index_t a = ket_ai.p;
    index_t i = ket_ai.q;
    double ja = 0.5 * modelspace->GetOrbit(a).j2;
    double ji = 0.5 * modelspace->GetOrbit(i).j2;

    // Ensure a is particle (less occupied) and i is hole (more occupied)
    if (ket_ai.op->occ > ket_ai.oq->occ)
    {
      std::swap(a, i);
      std::swap(ja, ji);
    }

    size_t II = 0;
    for (auto iket_bj : ph_list)
    {
      Ket& ket_bj = tbc_CC.GetKet(iket_bj);
      index_t b = ket_bj.p;
      index_t j = ket_bj.q;
      double jb = 0.5 * modelspace->GetOrbit(b).j2;
      double jj = 0.5 * modelspace->GetOrbit(j).j2;

      if (ket_bj.op->occ > ket_bj.oq->occ)
      {
        std::swap(b, j);
        std::swap(jb, jj);
      }

      // Diagonal SPE term: (eps_a - eps_i) * delta_{ai,bj}
      double H1b = (iket_ai == iket_bj) ? H.OneBody(a, a) - H.OneBody(i, i) : 0.0;

      // Two-body Pandya term
      // A_{ai,bj}(J) -= sum_{J'} (2J'+1) {ja ji J; jb jj J'} <aj';J'|V|bi';J'>
      int J1min = std::max(std::abs(ja - jj), std::abs(jb - ji));
      int J1max = std::min(ja + jj, jb + ji);
      double V_ph = 0.0;

      if (AngMom::Triangle(jj, jb, Jph) && AngMom::Triangle(ji, ja, Jph))
      {
        for (int J1 = J1min; J1 <= J1max; ++J1)
        {
          V_ph -= modelspace->GetSixJ(ja, ji, Jph, jb, jj, J1)
                  * (2 * J1 + 1)
                  * H.TwoBody.GetTBME_J(J1, a, j, b, i);
        }
      }

      A(I, II) = H1b + V_ph;
      II++;
    }
    I++;
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
/// Build the EOM B matrix (backward coupling block) for the 1p-1h sector.
///
/// The matrix element is the same as in the RPA B matrix and couples the
/// forward (particle-hole) and backward (hole-particle) sectors:
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

  size_t I = 0;
  for (auto iket_ai : ph_list)
  {
    Ket& ket_ai = tbc_CC.GetKet(iket_ai);
    index_t a = ket_ai.p;
    index_t i = ket_ai.q;
    double ja = 0.5 * modelspace->GetOrbit(a).j2;
    double ji = 0.5 * modelspace->GetOrbit(i).j2;

    if (ket_ai.op->occ > ket_ai.oq->occ)
    {
      std::swap(a, i);
      std::swap(ja, ji);
    }

    size_t II = 0;
    for (auto iket_bj : ph_list)
    {
      Ket& ket_bj = tbc_CC.GetKet(iket_bj);
      index_t b = ket_bj.p;
      index_t j = ket_bj.q;
      double jb = 0.5 * modelspace->GetOrbit(b).j2;
      double jj = 0.5 * modelspace->GetOrbit(j).j2;

      if (ket_bj.op->occ > ket_bj.oq->occ)
      {
        std::swap(b, j);
        std::swap(jb, jj);
      }

      int J1min = std::max(std::abs(ja - jb), std::abs(ji - jj));
      int J1max = std::min(ja + jb, ji + jj);
      double V_pp = 0.0;
      int phase_ib = AngMom::phase(ji + jb + Jph);

      if (AngMom::Triangle(jj, jb, Jph) && AngMom::Triangle(ji, ja, Jph))
      {
        for (int J1 = J1min; J1 <= J1max; ++J1)
        {
          V_pp += AngMom::phase(J1)
                  * (2 * J1 + 1)
                  * modelspace->GetSixJ(ja, ji, Jph, jj, jb, J1)
                  * H.TwoBody.GetTBME_J(J1, a, b, i, j);
        }
      }

      B(I, II) = V_pp * phase_ib;
      II++;
    }
    I++;
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

  for (size_t alpha = 0; alpha < n2; ++alpha)
  {
    const TwoPTwoHState& sa = tpth_basis[alpha];
    size_t a = sa.a, b = sa.b, ii = sa.i, jj = sa.j;
    int Jab = sa.Jab, Jij = sa.Jij;
    int j2a  = modelspace->GetOrbit(a).j2;
    int j2b  = modelspace->GetOrbit(b).j2;
    int j2ii = modelspace->GetOrbit(ii).j2;
    int j2jj = modelspace->GetOrbit(jj).j2;
    double ja   = 0.5*j2a;
    double jb   = 0.5*j2b;
    double ji   = 0.5*j2ii;
    double jj_v = 0.5*j2jj;
    double Nab = std::sqrt(1.0 + (a == b  ? 1.0 : 0.0));
    double Nij = std::sqrt(1.0 + (ii == jj ? 1.0 : 0.0));

    // Diagonal SPE term
    H22(alpha, alpha) = H.OneBody(a, a) + H.OneBody(b, b)
                      - H.OneBody(ii, ii) - H.OneBody(jj, jj);

    // Phase factors for the NineJ when active orbit is NOT the first-listed orbit:
    //   phase_pp_a = (-1)^{j_a+j_b+J_ab}  (applied when spec_p = a → active_p = b)
    //   phase_hh_a = (-1)^{j_i+j_j+J_ij}  (applied when spec_h = i → active_h = j)
    int phase_pp_a = AngMom::phase((j2a  + j2b)  / 2 + Jab);
    int phase_hh_a = AngMom::phase((j2ii + j2jj) / 2 + Jij);

    // Struct for one of the 4 active-ph-pair choices for alpha
    struct ACase
    {
      size_t act_p, spec_p;   // particle orbits
      size_t act_h, spec_h;   // hole orbits
      int    phase_alpha;     // 9j sign correction
    };

    // 4 cases: (active particle, spectator particle, active hole, spectator hole)
    std::array<ACase, 4> alpha_cases = {{
      {a,  b,  ii, jj, 1},                              // case 1
      {a,  b,  jj, ii, phase_hh_a},                     // case 2
      {b,  a,  ii, jj, phase_pp_a},                     // case 3
      {b,  a,  jj, ii, phase_pp_a * phase_hh_a}         // case 4
    }};

    for (size_t beta = 0; beta < n2; ++beta)
    {
      const TwoPTwoHState& sb = tpth_basis[beta];
      size_t ap = sb.a, bp = sb.b, ip = sb.i, jp = sb.j;
      int Jabp = sb.Jab, Jijp = sb.Jij;
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
      //   diagonal SPE (only when alpha==beta) + pp-pp + hh-hh + ph ring
      double val = (alpha == beta) ? H22(alpha, alpha) : 0.0;

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
      //   phase_pp_b = (-1)^{j_a'+j_b'+J_ab'} (applied if act_pb = bp)
      //   phase_hh_b = (-1)^{j_i'+j_j'+J_ij'} (applied if act_hb = jp)
      // ---------------------------------------------------------------
      int phase_pp_b = AngMom::phase((j2ap + j2bp) / 2 + Jabp);
      int phase_hh_b = AngMom::phase((j2ip + j2jp) / 2 + Jijp);

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

        // Phase for beta's NineJ
        int phase_beta = 1;
        if (act_pb != ap) phase_beta *= phase_pp_b;  // act_pb is second p of beta
        if (act_hb != ip) phase_beta *= phase_hh_b;  // act_hb is second h of beta

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
            // NineJ for alpha (first-listed orbits first):
            //   {j_a  j_b  J_ab }
            //   {j_i  j_j  J_ij }
            //   {J_ph J_sp J    }
            double n9j_a = AngMom::NineJ(ja,  jb,   (double)Jab,
                                         ji,  jj_v, (double)Jij,
                                         (double)Jph, (double)Jsp, (double)J);
            // NineJ for beta (first-listed orbits first):
            //   {j_a' j_b'  J_ab' }
            //   {j_i' j_j'  J_ij' }
            //   {J_ph J_sp  J     }
            double n9j_b = AngMom::NineJ(jap, jbp,  (double)Jabp,
                                         jip, jjp,  (double)Jijp,
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
// BuildH21_byIndex
// ---------------------------------------------------------------------------

///
/// Build the 2p2h × 1p1h coupling block H21.
///
/// From [Γ, Q_{ph}^†] → 2p2h (commutator_212 in Parzuchowski notation).
/// Four contributions for each 2p2h state α=(ab Jab; cd Jcd; J) and
/// 1p1h state β=(e,f;J):
///
///   K = sqrt((2Jab+1)(2Jcd+1)) / sqrt((1+δ_ab)(1+δ_cd)) × (-1)^J
///
///   sm1 (e==a):  H21 -= phase((j2_a+j2_b)/2-Jcd)
///                       × W6j(Jab,Jcd,J; jf,ja,jb) × V_norm(Jcd,f,b,c,d) × K
///   sm2 (e==b):  H21 += phase(Jab+Jcd)
///                       × W6j(Jab,Jcd,J; jf,jb,ja) × V_norm(Jcd,f,a,c,d) × K
///   sm3 (f==d):  H21 += phase((j2_c+j2_d)/2-Jab)
///                       × W6j(Jab,Jcd,J; jd,je,jc) × V_norm(Jab,a,b,c,e) × K
///   sm4 (f==c):  H21 -= phase(Jcd+Jab)
///                       × W6j(Jab,Jcd,J; jc,je,jd) × V_norm(Jab,a,b,d,e) × K
///
void EOMImsrg::BuildH21_byIndex(size_t ich_CC)
{
  TwoBodyChannel_CC& tbc_CC = modelspace->GetTwoBodyChannel_CC(ich_CC);
  const auto& ph_list = tbc_CC.GetKetIndex_ph();
  size_t nph  = ph_list.size();
  size_t n2   = tpth_basis.size();
  int    J    = tbc_CC.J;
  H21.zeros(n2, nph);

  // Build a fast lookup: (orbit_index) → list of column indices in ph_list
  // where that orbit is the particle (e) or hole (f).
  // Key: orbit index; Value: vector of (col, is_particle) pairs
  std::vector<std::vector<std::pair<size_t,bool>>> orbit_to_ph(
      modelspace->GetNumberOrbits());

  size_t col = 0;
  for (auto iket : ph_list)
  {
    Ket& kt = tbc_CC.GetKet(iket);
    index_t e = kt.p, f = kt.q;
    if (kt.op->occ > kt.oq->occ) std::swap(e, f);
    orbit_to_ph[e].push_back({col, true});
    orbit_to_ph[f].push_back({col, false});
    ++col;
  }

  // Precompute orbit quantum numbers for each ph state
  std::vector<index_t> ph_e(nph), ph_f(nph);
  col = 0;
  for (auto iket : ph_list)
  {
    Ket& kt = tbc_CC.GetKet(iket);
    index_t e = kt.p, f = kt.q;
    if (kt.op->occ > kt.oq->occ) std::swap(e, f);
    ph_e[col] = e;
    ph_f[col] = f;
    ++col;
  }

  for (size_t alpha = 0; alpha < n2; ++alpha)
  {
    const TwoPTwoHState& sa = tpth_basis[alpha];
    size_t a = sa.a, b = sa.b, c = sa.i, d = sa.j;
    int Jab = sa.Jab, Jcd = sa.Jij;

    const Orbit& oa = modelspace->GetOrbit(a);
    const Orbit& ob = modelspace->GetOrbit(b);
    const Orbit& oc = modelspace->GetOrbit(c);
    const Orbit& od = modelspace->GetOrbit(d);

    double ja = 0.5*oa.j2, jb = 0.5*ob.j2;
    double jc = 0.5*oc.j2, jd = 0.5*od.j2;

    double Nab = std::sqrt(1.0 + (a == b ? 1.0 : 0.0));
    double Ncd = std::sqrt(1.0 + (c == d ? 1.0 : 0.0));
    // Common factor K = sqrt((2Jab+1)(2Jcd+1)) / (Nab*Ncd) * (-1)^J
    double K = std::sqrt((2.0*Jab+1)*(2.0*Jcd+1)) / (Nab * Ncd)
               * AngMom::phase(J);

    // --- sm1: e == a ---
    // H21[alpha,col] -= phase((j2_a+j2_b)/2 - Jcd)
    //                   * W6j(Jab,Jcd,J; jf,ja,jb) * V_norm(Jcd,f,b,c,d) * K
    {
      int phase1 = AngMom::phase((oa.j2 + ob.j2)/2 - Jcd);
      // Loop over ph states where e == a
      for (auto& [c_idx, is_part] : orbit_to_ph[a])
      {
        if (!is_part) continue;  // need e == a (particle)
        size_t f_orb = ph_f[c_idx];
        const Orbit& of = modelspace->GetOrbit(f_orb);
        double jf = 0.5*of.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jcd, (double)J,
                                         jf, ja, jb);
        if (w6j == 0.0) continue;
        double v = H.TwoBody.GetTBME_J_norm(Jcd, f_orb, b, c, d);
        H21(alpha, c_idx) -= phase1 * w6j * v * K;
      }
    }

    // --- sm2: e == b ---
    // H21[alpha,col] += phase(Jab+Jcd)
    //                   * W6j(Jab,Jcd,J; jf,jb,ja) * V_norm(Jcd,f,a,c,d) * K
    {
      int phase2 = AngMom::phase(Jab + Jcd);
      for (auto& [c_idx, is_part] : orbit_to_ph[b])
      {
        if (!is_part) continue;
        size_t f_orb = ph_f[c_idx];
        const Orbit& of = modelspace->GetOrbit(f_orb);
        double jf = 0.5*of.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jcd, (double)J,
                                         jf, jb, ja);
        if (w6j == 0.0) continue;
        double v = H.TwoBody.GetTBME_J_norm(Jcd, f_orb, a, c, d);
        H21(alpha, c_idx) += phase2 * w6j * v * K;
      }
    }

    // --- sm3: f == d ---
    // H21[alpha,col] += phase((j2_c+j2_d)/2 - Jab)
    //                   * W6j(Jab,Jcd,J; jd,je,jc) * V_norm(Jab,a,b,c,e) * K
    {
      int phase3 = AngMom::phase((oc.j2 + od.j2)/2 - Jab);
      for (auto& [c_idx, is_part] : orbit_to_ph[d])
      {
        if (is_part) continue;  // need f == d (hole)
        size_t e_orb = ph_e[c_idx];
        const Orbit& oe = modelspace->GetOrbit(e_orb);
        double je = 0.5*oe.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jcd, (double)J,
                                         jd, je, jc);
        if (w6j == 0.0) continue;
        double v = H.TwoBody.GetTBME_J_norm(Jab, a, b, c, e_orb);
        H21(alpha, c_idx) += phase3 * w6j * v * K;
      }
    }

    // --- sm4: f == c ---
    // H21[alpha,col] -= phase(Jcd+Jab)
    //                   * W6j(Jab,Jcd,J; jc,je,jd) * V_norm(Jab,a,b,d,e) * K
    {
      int phase4 = AngMom::phase(Jcd + Jab);
      for (auto& [c_idx, is_part] : orbit_to_ph[c])
      {
        if (is_part) continue;
        size_t e_orb = ph_e[c_idx];
        const Orbit& oe = modelspace->GetOrbit(e_orb);
        double je = 0.5*oe.j2;
        double w6j = modelspace->GetSixJ((double)Jab, (double)Jcd, (double)J,
                                         jc, je, jd);
        if (w6j == 0.0) continue;
        double v = H.TwoBody.GetTBME_J_norm(Jab, a, b, d, e_orb);
        H21(alpha, c_idx) -= phase4 * w6j * v * K;
      }
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
  BuildAMatrix_byIndex(ich_CC);
  if (mode == "EOM")
    BuildBMatrix_byIndex(ich_CC);
  else
    B.zeros(A.n_rows, A.n_cols);

  if (mode == "EOM2")
  {
    Build2p2hBasis_byIndex(ich_CC);
    BuildH22_byIndex(ich_CC);
    BuildH21_byIndex(ich_CC);
  }

  SolveCurrentChannel(mode);

  // Store results for later lookup
  EOMChannel ch;
  ch.Energies = Energies;
  ch.X        = X;
  ch.Y        = Y;
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

/// Core solver: operates on the already-filled A (and B or H22/H21) matrices.
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
  }
  else if (mode == "EOM")
  {
    // Secular matrix: [ A  B; -B -A ]
    arma::mat M = arma::join_vert(
        arma::join_horiz( A,  B),
        arma::join_horiz(-B, -A));

    arma::cx_vec cx_eigvals;
    arma::cx_mat cx_eigvecs;
    arma::eig_gen(cx_eigvals, cx_eigvecs, M);

    double norm_imag = arma::norm(arma::imag(cx_eigvals), "fro");
    if (norm_imag > EOM_IMAG_TOL)
    {
      std::cout << "WARNING EOMImsrg: non-zero imaginary eigenvalues ("
                << norm_imag << ") in channel " << current_channel
                << "  " << __FILE__ << ":" << __LINE__ << std::endl;
    }

    // Keep only positive-real solutions (discard spurious negatives)
    arma::uvec pos_idx = arma::find(arma::real(cx_eigvals)
                                    + arma::imag(cx_eigvals) > 0);
    size_t len = pos_idx.n_elem;
    arma::vec Etmp = arma::real(cx_eigvals(pos_idx));
    arma::mat Vtmp = arma::real(cx_eigvecs.cols(pos_idx));

    arma::uvec ord = arma::sort_index(Etmp);
    Energies = Etmp(ord);
    arma::mat Vsorted = Vtmp.cols(ord);

    X = Vsorted.head_rows(len);
    Y = Vsorted.tail_rows(len);

    // Normalise: X^T X - Y^T Y = 1
    for (size_t mu = 0; mu < len; mu++)
    {
      double nxy = arma::dot(X.col(mu), X.col(mu))
                   - arma::dot(Y.col(mu), Y.col(mu));
      if (std::abs(nxy) > EOM_NORM_TOL)
      {
        double inv_sqrt_nxy = 1.0 / std::sqrt(std::abs(nxy));
        X.col(mu) *= inv_sqrt_nxy;
        Y.col(mu) *= inv_sqrt_nxy;
      }
    }
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
    }
  }
  else
  {
    throw std::invalid_argument("EOMImsrg::Solve: unknown mode '" + mode
                                + "'. Use 'TDA', 'EOM', or 'EOM2'.");
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
/// Compute the one-body transition matrix element
/// \f$\langle 0 \| \hat{O} \| \mu \rangle\f$ for excited state \p state_index.
///
/// For a one-body scalar/tensor operator \f$\hat{O}\f$ of rank \f$\lambda\f$:
/// \f[
///   \langle 0 \| \hat{O} \| \mu \rangle
///     = \sum_{ai}
///         \bigl[ X_{ai}^{(\mu)} \langle a \| O \| i \rangle
///              + (-1)^{\lambda+1} Y_{ai}^{(\mu)} \langle i \| O \| a \rangle
///         \bigr].
/// \f]
/// Here the sum runs over 1p-1h pairs (a=particle, i=hole) in the solved channel.
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

  // Phase factor from operator rank: (-1)^{lambda+1}
  int lambda = Op.GetJRank();
  double phase_lambda = AngMom::phase(lambda + 1);

  double T = 0.0;
  size_t I = 0;
  for (auto iket_ai : ph_list)
  {
    Ket& ket = tbc_CC.GetKet(iket_ai);
    index_t a = ket.p;
    index_t i = ket.q;

    if (ket.op->occ > ket.oq->occ)
      std::swap(a, i);

    double Oai = Op.OneBody(a, i);
    double Oia = Op.OneBody(i, a);

    T += ch.X(I, state_index) * Oai
         + phase_lambda * ch.Y(I, state_index) * Oia;
    I++;
  }
  return T;
}
