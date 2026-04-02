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
  : modelspace(nullptr), current_channel(0)
{}

EOMImsrg::EOMImsrg(Operator& H_imsrg)
  : modelspace(H_imsrg.modelspace), H(H_imsrg), current_channel(0)
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

/// Core solver: operates on the already-filled A (and B) matrices.
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
  else
  {
    throw std::invalid_argument("EOMImsrg::Solve: unknown mode '" + mode
                                + "'. Use 'TDA' or 'EOM'.");
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
