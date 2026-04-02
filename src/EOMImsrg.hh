///////////////////////////////////////////////////////////////////////////////////
//    EOMImsrg.hh, part of  imsrg++
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
/// \file EOMImsrg.hh
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
/// Two levels of approximation are supported:
///   - **TDA** (Tamm-Dancoff Approximation): diagonalises the A matrix in the
///     1p-1h sector. Equivalent to IMSRG-TDA.
///   - **EOM** (full Equation of Motion): solves the RPA-like secular equation
///     including the B matrix that couples forward and backward amplitudes.
///     Equivalent to IMSRG-RPA.
///
/// The A matrix is
/// \f[
///   A_{ai,bj}(J) = (\varepsilon_a - \varepsilon_i)\delta_{ab}\delta_{ij}
///                  + \overline{V}^J_{aibj},
/// \f]
/// where \f$\overline{V}^J\f$ is the Pandya-transformed two-body matrix
/// element of \f$\tilde{H}\f$.  This is identical to the RPA A matrix, but
/// evaluated with the IMSRG-evolved \f$\tilde{H}\f$ rather than the bare
/// Hamiltonian.
///
/// Usage example:
/// \code
///   EOMImsrg eom( imsrg_solver.GetH_s() );
///   eom.Solve(2, 1, 0);            // J=2, positive parity, Tz=0
///   arma::vec E = eom.GetExcitationEnergies();
///   double B_E2 = eom.ComputeTransitionME(E2op, 0); // lowest 2+ state
/// \endcode

#ifndef EOMImsrg_hh
#define EOMImsrg_hh 1

#include <armadillo>
#include <map>
#include <string>

#include "ModelSpace.hh"
#include "Operator.hh"

/// Results stored for one (J, parity, Tz) channel after a Solve() call.
struct EOMChannel
{
  arma::vec Energies; ///< Excitation energies (MeV), sorted ascending
  arma::mat X;        ///< Forward amplitudes  (nph x nstates)
  arma::mat Y;        ///< Backward amplitudes (nph x nstates); zero for TDA
};

///
/// \class EOMImsrg
/// \brief EOM-IMSRG excited-state solver (1p-1h sector).
///
class EOMImsrg
{
 public:
  // -----------------------------------------------------------------------
  // Fields
  // -----------------------------------------------------------------------
  ModelSpace* modelspace; ///< Pointer to the model space
  Operator    H;          ///< Copy of the IMSRG-evolved Hamiltonian

  /// Current working matrices (populated by BuildAMatrix / BuildBMatrix)
  arma::mat A;
  arma::mat B;

  /// Results for the most recently solved channel
  arma::vec Energies;
  arma::mat X;
  arma::mat Y;

  /// Index of the most recently solved TwoBodyChannel_CC
  size_t current_channel;

  /// Results indexed by TwoBodyChannel_CC index
  std::map<size_t, EOMChannel> ChannelResults;

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------
  EOMImsrg();
  /// Construct from an IMSRG-evolved Hamiltonian (deep copy).
  explicit EOMImsrg(Operator& H_imsrg);

  // -----------------------------------------------------------------------
  // Matrix construction
  // -----------------------------------------------------------------------
  /// Build the A matrix (TDA / EOM forward block) for the given quantum numbers.
  void BuildAMatrix(int J, int parity, int Tz);
  /// Build the B matrix (backward coupling block) for the given quantum numbers.
  void BuildBMatrix(int J, int parity, int Tz);

  /// Build the A matrix by TwoBodyChannel_CC index.
  void BuildAMatrix_byIndex(size_t ich_CC);
  /// Build the B matrix by TwoBodyChannel_CC index.
  void BuildBMatrix_byIndex(size_t ich_CC);

  // -----------------------------------------------------------------------
  // Solvers
  // -----------------------------------------------------------------------
  /// Solve for excitation energies and amplitudes in one (J, parity, Tz) channel.
  /// @param mode  "TDA"  – diagonalise A only (no B matrix).
  ///              "EOM"  – solve the full EOM secular equation (A and B).
  void Solve(int J, int parity, int Tz, std::string mode = "TDA");
  /// Same as Solve() but addressed by channel index.
  void Solve_byIndex(size_t ich_CC, std::string mode = "TDA");

  /// Solve all (J, parity, Tz) channels that have at least one 1p-1h pair.
  void SolveAllChannels(std::string mode = "TDA");

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------
  /// Excitation energies for the most recently solved channel (ascending order).
  arma::vec GetExcitationEnergies() const;
  /// Forward amplitudes X_{ai} for state \p state_index in the current channel.
  arma::vec GetAmplitudesX(size_t state_index) const;
  /// Backward amplitudes Y_{ai} for state \p state_index in the current channel.
  arma::vec GetAmplitudesY(size_t state_index) const;

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

 private:
  /// Implementation shared by Solve() overloads; operates on current A, B, channel.
  void SolveCurrentChannel(std::string mode);
};

#endif
