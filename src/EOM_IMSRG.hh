///////////////////////////////////////////////////////////////////////////////////
//    EOM_IMSRG.hh, part of  imsrg++
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

///
/// \file EOM_IMSRG.hh
///
/// Equation-of-Motion IMSRG (EOM-IMSRG) solver.
///
/// EOM-IMSRG computes nuclear excited-state energies and transition matrix
/// elements on top of a IMSRG ground-state solution.  The key idea is that the
/// similarity-transformed Hamiltonian \f$\tilde{H}\f$ obtained from an IMSRG
/// flow already encodes ground-state correlations, so one can obtain excited
/// states cheaply by diagonalising in the 1p-1h (particle-hole) sector of
/// \f$\tilde{H}\f$.
///
/// Three levels of approximation are supported:
///   - **TDA** (Tamm-Dancoff Approximation): diagonalises the A matrix in the
///     1p-1h sector. Equivalent to EOM-IMSRG(1).
///   - **EOM** (Equation of Motion at 1p-1h level): diagonalises the A matrix
///     in the 1p-1h sector (same as TDA).  The EOM ladder operator contains
///     only excitation amplitudes X; there are no de-excitation terms.
///   - **EOM2** (EOM-IMSRG with 2p-2h): diagonalises the full
///     [H_11 H_12; H_21 H_22] block matrix in the combined 1p-1h + 2p-2h space.
///     This is the proper EOM-IMSRG following Parzuchowski et al. (PRC 2017).
///     H_11 is the TDA A matrix; H_22 contains the 2p-2h diagonal energies
///     plus pp-pp, hh-hh ladder and ph ring interactions; H_21 is the
///     coupling from [Γ, Q_{ph}] → 2p-2h (4 terms from commutator_212).
///     By default the full matrix is diagonalised with LAPACK's dsyev (all
///     eigenvalues).  Set `lanczos_nev > 0` before calling Solve() to instead
///     use an Implicitly Restarted Lanczos/Arnoldi method (IRAM, identical in
///     spirit to ARPACK's dsaupd/dseupd used by the reference Fortran code) that
///     computes only the `lanczos_nev` algebraically smallest eigenvalues.  This
///     is much cheaper for large model spaces where only a few low-lying
///     excitation energies are needed.
///
/// The A matrix is
/// \f[
///   A_{ai,bj}(J) = (\varepsilon_a - \varepsilon_i)\delta_{ab}\delta_{ij}
///                  + \overline{V}^J_{aibj},
/// \f]
/// where \f$\overline{V}^J\f$ is the Pandya-transformed two-body matrix
/// element of \f$\tilde{H}\f$, evaluated with the IMSRG-evolved Hamiltonian.
///
/// Usage example:
/// \code
///   EOM_IMSRG eom( imsrg_solver.GetH_s() );
///   eom.Solve(2, 1, 0, "EOM2");    // full 1p1h+2p2h EOM-IMSRG
///   arma::vec E = eom.GetExcitationEnergies();
///   double B_E2 = eom.ComputeTransitionME(E2op, 0); // lowest 2+ state
/// \endcode

#ifndef EOM_IMSRG_hh
#define EOM_IMSRG_hh 1

#include <armadillo>
#include <map>
#include <string>

#include "ModelSpace.hh"
#include "Operator.hh"

/// Compact representation of one 2p-2h basis state.
struct TwoPTwoHState
{
  size_t a; ///< First particle orbit index  (a <= b by TwoBodyChannel convention)
  size_t b; ///< Second particle orbit index
  size_t i; ///< First hole orbit index      (i <= j)
  size_t j; ///< Second hole orbit index
  int Jab;  ///< Angular momentum coupling of the pp pair (actual value, e.g. 0,1,2,…)
  int Jij;  ///< Angular momentum coupling of the hh pair
};

/// Results stored for one (J, parity, Tz) channel after a Solve() call.
struct EOMChannel
{
  arma::vec Energies; ///< Excitation energies (MeV), sorted ascending
  arma::mat X;        ///< 1p1h excitation amplitudes  (nph x nstates)
  /// Mode-dependent:
  ///   - TDA/EOM: zero matrix (nph x nstates) — no de-excitation amplitudes
  ///   - EOM2:    2p2h amplitudes X̌_{abij}^{Jab,Jij,J}(ν) (n2p2h x nstates)
  arma::mat Y;
  arma::vec OnePhNorms; ///< n(1p1h) = sum_ai |X_ai|^2 for each state
  size_t OnePhCount = 0; ///< Number of 1p1h amplitudes in the solved channel
  size_t TwoPhCount = 0; ///< Number of 2p2h amplitudes in the solved channel
  size_t LanczosIterations = 0; ///< Number of Lanczos iterations used (0 for dense solve)
  /// 2p2h basis states for EOM2 mode (empty for TDA/EOM).
  /// Stored here so ComputeTransitionME_byIndex can use them for all channels.
  std::vector<TwoPTwoHState> tpth_basis;
};

///
/// \class EOM_IMSRG
/// \brief EOM-IMSRG excited-state solver (1p-1h and 1p1h+2p2h sectors).
///
class EOM_IMSRG
{
 public:
  // -----------------------------------------------------------------------
  // Fields
  // -----------------------------------------------------------------------
  ModelSpace* modelspace; ///< Pointer to the model space
  Operator    H;          ///< Copy of the IMSRG-evolved Hamiltonian

  /// Current working matrices (populated by Build_AMatrix / Build_BMatrix)
  arma::mat A;
  arma::mat B;

  /// 2p2h working matrices (populated when mode == "EOM2")
  arma::mat H22; ///< 2p2h × 2p2h block
  arma::mat H21; ///< 2p2h × 1p1h coupling block (rows = 2p2h states, cols = 1p1h states)
  arma::mat A12; ///< 1p1h × 2p2h coupling block (rows = 1p1h states, cols = 2p2h states; = H21^T)
  std::vector<TwoPTwoHState> tpth_basis; ///< 2p2h basis for current channel

  /// Results for the most recently solved channel
  arma::vec Energies;
  arma::mat X;
  arma::mat Y;
  arma::vec OnePhNorms;
  size_t one_ph_count;
  size_t two_ph_count;
  size_t lanczos_iterations;

  /// Index of the most recently solved TwoBodyChannel_CC
  size_t current_channel;

  /// Number of eigenvalues to compute with Lanczos (EOM2 mode only).
  /// When 0 (default) the full dense LAPACK driver is used and all eigenvalues
  /// are returned.  When > 0 the armadillo newarp Implicitly Restarted Arnoldi
  /// solver computes only the \p lanczos_nev algebraically smallest eigenvalues,
  /// which is much faster for large (1p1h + 2p2h) spaces where only a handful of
  /// low-lying excited states are needed.
  int lanczos_nev;

  /// Results indexed by TwoBodyChannel_CC index
  std::map<size_t, EOMChannel> ChannelResults;

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------
  EOM_IMSRG();
  /// Construct from an IMSRG-evolved Hamiltonian (deep copy).
  explicit EOM_IMSRG(Operator& H_imsrg);

  // -----------------------------------------------------------------------
  // Matrix construction
  // -----------------------------------------------------------------------
  /// Build the A matrix for the given quantum numbers.
  void Build_AMatrix(int J, int parity, int Tz);
  /// Build the RPA B matrix (de-excitation coupling, for RPA use only).
  void Build_BMatrix(int J, int parity, int Tz);

  /// Build the A matrix by TwoBodyChannel_CC index.
  void Build_AMatrix_byIndex(size_t ich_CC);
  /// Build the RPA B matrix by TwoBodyChannel_CC index (for RPA use only).
  void Build_BMatrix_byIndex(size_t ich_CC);

  /// Enumerate the 2p2h basis states compatible with EOM channel ich_CC.
  void Build2p2hBasis_byIndex(size_t ich_CC);
  /// Build the 2p2h × 2p2h block H22 (diagonal + pp-pp + hh-hh + ph ring).
  void BuildH22_byIndex(size_t ich_CC);
  /// Build the 2p2h × 1p1h coupling block H21 from [Γ, Q_ph] → 2p2h.
  void Build_H21_byIndex(size_t ich_CC);
  /// Build the 1p1h × 2p2h coupling block A12 (= H21^T) for the matrix EOM2 solver.
  /// Called only by the explicit-matrix path (Solve_byIndex with mode "EOM2").
  /// The matrix-free Lanczos path uses ApplyH21T_matvec instead.
  void Build_A12_byIndex(size_t ich_CC);

  /// Precompute the Pandya-transformed H for ALL CC channels and store in
  /// Hbar_CC / pan_idx.  Idempotent: returns immediately if already built.
  void BuildPandya();
  /// Release the Pandya table (Hbar_CC and pan_idx).
  void ClearPandya();

  // -----------------------------------------------------------------------
  // Solvers
  // -----------------------------------------------------------------------
  /// Solve for excitation energies and amplitudes in one (J, parity, Tz) channel.
  /// @param mode  "TDA"   – diagonalise A only in the 1p1h sector.
  ///              "EOM"   – diagonalise A in the 1p1h sector (same as TDA;
  ///                        excitation amplitudes only, no de-excitation terms).
  ///              "RPA"   – solve the RPA secular equation [A B; -B -A] (includes
  ///                        de-excitation amplitudes Y).
  ///              "EOM2"  – full EOM-IMSRG with 1p1h+2p2h block matrix.
  void Solve(int J, int parity, int Tz, std::string mode = "TDA");
  /// Same as Solve() but addressed by channel index.
  void Solve_byIndex(size_t ich_CC, std::string mode = "TDA");

  /// Solve all (J, parity, Tz) channels that have at least one 1p-1h pair.
  void SolveAllChannels(std::string mode = "TDA");

  /// Matrix-free Lanczos EOM2 solver for channel ich_CC.
  ///
  /// Reuses Build_AMatrix_byIndex() and Build2p2hBasis_byIndex() but never
  /// materialises the H22 (N2×N2) or H21 (N2×Nph) matrices.  The action of
  /// the full H_EOM2 = [A, H21^T; H21, H22] on a Lanczos vector is computed
  /// on the fly each iteration, reducing peak memory from O(N2²) to O(N2·ncv).
  ///
  /// @param nev  Number of algebraically-lowest eigenvalues to converge.
  ///             Must satisfy 1 ≤ nev < N = nph + n2p2h.
  void Solve_byIndex_MF(size_t ich_CC, int nev);

  /// Run Solve_byIndex_MF for all ph channels.
  void SolveAllChannels_MF(int nev);

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------
  /// Excitation energies for the most recently solved channel (ascending order).
  arma::vec GetExcitationEnergies() const;
  /// 1p1h excitation amplitudes X_{ai} for state \p state_index in the current channel.
  arma::vec GetAmplitudesX(size_t state_index) const;
  /// For EOM2: 2p2h amplitudes for state \p state_index.  Zero for TDA/EOM.
  arma::vec GetAmplitudesY(size_t state_index) const;
  /// 1p1h norm weights n(1p1h)=sum_ai |X_ai|^2 for the solved states.
  arma::vec GetOnePhNorms() const;
  /// Number of 1p1h amplitudes in the current channel.
  size_t GetOnePhCount() const;
  /// Number of 2p2h amplitudes in the current channel.
  size_t GetTwoPhCount() const;
  /// Number of Lanczos iterations used in the most recent EOM2 solve.
  size_t GetLanczosIterations() const;
  /// Print a compact spectrum summary matching the reference EOM-IMSRG output.
  void PrintSummary() const;

  /// Return the stored EOMChannel for channel index \p ich_CC, if it exists.
  EOMChannel GetChannelResults(size_t ich_CC) const;

  // -----------------------------------------------------------------------
  // Transition matrix elements
  // -----------------------------------------------------------------------
  /// Compute the one-body transition matrix element
  /// \f$\langle 0 \| \hat{O} \| \mu \rangle\f$ for state \p state_index
  /// in the most recently solved channel.
  /// @param Op        One-body operator (e.g. E2, M1, Gamow-Teller).
  /// @param state_index  Index into the sorted excitation spectrum (0 = lowest).
  double ComputeTransitionME(Operator& Op, size_t state_index) const;

  /// Compute the transition matrix element using stored channel results.
  double ComputeTransitionME_byIndex(size_t ich_CC, Operator& Op, size_t state_index) const;

  /// Full H_EOM2 = [A, H21^T; H21, H22] matvec, used by EOMMatFreeOp.
  /// May also be called directly (e.g. for testing or custom iterative solvers).
  void ApplyH_EOM2_matvec(const arma::vec& v, arma::vec& Hv) const;

 private:
  /// Implementation shared by Solve() overloads; operates on current A, B, channel.
  void SolveCurrentChannel(std::string mode);

  // -----------------------------------------------------------------------
  // Matrix-free matvec helpers (used by Solve_byIndex_MF / EOMMatFreeOp).
  // Populate mf_orbit_to_ph / mf_ph_particle / mf_ph_hole before calling
  // Apply*_matvec.
  // Notation: a,b,c,… for particle orbits; i,j,k,… for hole orbits.
  // -----------------------------------------------------------------------

  /// Precomputed ph-orbit lookup for matrix-free path.
  /// mf_orbit_to_ph[orb] = list of (col, is_particle) pairs for the current channel.
  std::vector<std::vector<std::pair<size_t,bool>>> mf_orbit_to_ph;
  std::vector<index_t> mf_ph_particle; ///< particle orbit (a,b,c,…) of each ph column
  std::vector<index_t> mf_ph_hole;     ///< hole    orbit  (i,j,k,…) of each ph column
  /// CC-channel ket-ordering phase for each ph column, matching Build_AMatrix_byIndex.
  /// phase = 1 if ket stored as (particle,hole); -(-1)^{ja+ji-J} if stored as (hole,particle).
  std::vector<int>     mf_ph_phase;

  // -----------------------------------------------------------------------
  // Precomputed Pandya-transformed H for ALL CC channels.
  // Call BuildPandya() once before any matrix construction.
  // Release with ClearPandya().
  //
  //   Hbar_CC[ich_ph](ibra, iket)
  //     = H̄(p_bra, h_bra; p_ket, h_ket; J_ph)
  //     = -Σ_{J'}(2J'+1) W6j(j_p_bra, j_h_bra, J_ph; j_p_ket, j_h_ket, J')
  //               × GetTBME_J(J', p_bra, h_ket, p_ket, h_bra)
  //
  // pan_idx[ich_ph][(p_orb, h_orb)] → local row/col index inside Hbar_CC.
  // -----------------------------------------------------------------------
  std::vector<arma::mat>                            Hbar_CC;
  std::vector<std::map<std::pair<size_t,size_t>,int>> pan_idx;

  /// Compute Hv_2p2h += H22 * v_2p2h without building H22.
  void ApplyH22_matvec(const arma::vec& v, arma::vec& Hv) const;

  /// Compute Hv_2p2h += H21 * v_ph without building H21.
  void ApplyH21_matvec(const arma::vec& v_ph, arma::vec& Hv_2p2h) const;

  /// Compute Hv_ph += H21^T * v_2p2h without building H21.
  void ApplyH21T_matvec(const arma::vec& v_2p2h, arma::vec& Hv_ph) const;
};

#endif
