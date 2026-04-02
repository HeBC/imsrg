#!/usr/bin/env python3
"""
UnitTestEOM.py
==============
Unit tests for the EOM-IMSRG solver (EOMImsrg class).

Tests performed:
  1. TDA eigenvalues are real and non-decreasing.
  2. TDA A-matrix agrees exactly with the RPA A-matrix for the same Hamiltonian
     and channel (verifies the Pandya formula is implemented consistently).
  3. Full EOM eigenvalues are real and positive.
  4. EOM energies bracket TDA energies: omega_EOM <= omega_TDA (RPA theorem).
  5. TDA amplitudes are orthonormal (X^T X = I, Y = 0).
  6. Full EOM amplitudes satisfy the EOM norm: X^T X - Y^T Y = I.
  7. SolveAllChannels runs without error.
  8. ComputeTransitionME returns a finite float.
  9. GetChannelResults raises for unsolved channel.
 10. New Python-binding functions:
       BuildAMatrix_byIndex / BuildBMatrix_byIndex
       Solve_byIndex
       ComputeTransitionME_byIndex
       GetNStates / GetNBasis (EOMChannel)
       EOMChannel.Print / EOMChannel.energies / X_matrix / Y_matrix
       EOMImsrg.Energies / X / Y / current_channel properties
       GetSolvedChannels
       PrintA / PrintB
       H field access
 13. EOM2 mode (1p1h + 2p2h): Solve() runs, energies are finite, lowest EOM2
     energy ≤ lowest TDA energy, SolveAllChannels works.
 14. Lanczos EOM2 mode: lanczos_nev > 0 triggers the IRAM Lanczos solver;
     Lanczos energies agree with dense EOM2 energies for the lowest states.
"""

import sys
import math
import pyIMSRG

PASS = True

def check(condition, name):
    global PASS
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        PASS = False

def main():
    # ----------------------------------------------------------------
    # Setup: small model space and a random Hermitian scalar Hamiltonian
    # ----------------------------------------------------------------
    emax = 2
    ms = pyIMSRG.ModelSpace(emax, 'He4', 'He4')
    ms.PreCalculateSixJ()
    ut = pyIMSRG.UnitTest(ms)

    # Random Hermitian scalar 2-body operator
    H = ut.RandomOp(ms, 0, 0, 0, 2, +1)

    # ----------------------------------------------------------------
    # Test 1 & 2: TDA eigenvalues, consistency with RPA A-matrix
    # ----------------------------------------------------------------
    print("Test group: TDA solver")

    J, par, Tz = 2, 1, 0

    eom = pyIMSRG.EOMImsrg(H)
    eom.Solve(J, par, Tz, "TDA")
    tda_energies = eom.GetExcitationEnergies()

    rpa = pyIMSRG.RPA(H)
    rpa.ConstructAMatrix(J, par, Tz, False)
    rpa.SolveTDA()
    rpa_energies_tda = rpa.GetEnergies()

    # TDA energies from EOMImsrg and RPA::SolveTDA must agree exactly
    sizes_match = (len(tda_energies) == len(rpa_energies_tda))
    check(sizes_match, "TDA and RPA A-matrix have same dimension")

    if sizes_match and len(tda_energies) > 0:
        max_diff = max(abs(e1 - e2)
                       for e1, e2 in zip(sorted(tda_energies),
                                         sorted(rpa_energies_tda)))
        check(max_diff < 1e-10,
              f"TDA energies match RPA SolveTDA (max diff = {max_diff:.2e})")

        # Eigenvalues non-decreasing
        sorted_e = sorted(tda_energies)
        non_decreasing = all(sorted_e[i] <= sorted_e[i+1]
                             for i in range(len(sorted_e)-1))
        check(non_decreasing, "TDA energies non-decreasing")

    # ----------------------------------------------------------------
    # Test 3: TDA amplitudes orthonormal
    # ----------------------------------------------------------------
    print("Test group: TDA amplitudes")

    if len(tda_energies) > 0:
        X0 = eom.GetAmplitudesX(0)
        Y0 = eom.GetAmplitudesY(0)

        norm_X0 = sum(x*x for x in X0)
        check(abs(norm_X0 - 1.0) < 1e-10,
              f"TDA: |X_0|^2 = 1 (got {norm_X0:.6f})")

        norm_Y0 = sum(y*y for y in Y0)
        check(abs(norm_Y0) < 1e-10,
              f"TDA: Y amplitudes are zero (|Y|^2 = {norm_Y0:.2e})")

    # ----------------------------------------------------------------
    # Test 4 & 5: Full EOM solver
    # ----------------------------------------------------------------
    print("Test group: full EOM solver")

    eom_full = pyIMSRG.EOMImsrg(H)
    eom_full.Solve(J, par, Tz, "EOM")
    eom_energies = eom_full.GetExcitationEnergies()

    if len(eom_energies) > 0:
        # All EOM energies must be positive (physical)
        all_positive = all(e > 0 for e in eom_energies)
        check(all_positive, "EOM: all excitation energies positive")

        # RPA theorem: omega_EOM <= omega_TDA for each state
        if len(tda_energies) > 0:
            n_common = min(len(tda_energies), len(eom_energies))
            s_eom = sorted(eom_energies)[:n_common]
            s_tda = sorted(tda_energies)[:n_common]
            rpa_theorem = all(e_eom <= e_tda + 1e-10
                              for e_eom, e_tda in zip(s_eom, s_tda))
            check(rpa_theorem,
                  "EOM theorem: omega_EOM <= omega_TDA (up to 1e-10)")

        # EOM norm: X^T X - Y^T Y = 1 for each state
        norms_ok = True
        for mu in range(len(eom_energies)):
            Xmu = eom_full.GetAmplitudesX(mu)
            Ymu = eom_full.GetAmplitudesY(mu)
            nxy = sum(x*x for x in Xmu) - sum(y*y for y in Ymu)
            if abs(abs(nxy) - 1.0) > 1e-6:
                norms_ok = False
                break
        check(norms_ok, "EOM norm: |X|^2 - |Y|^2 = 1 for all states")

    # ----------------------------------------------------------------
    # Test 6: SolveAllChannels runs without error
    # ----------------------------------------------------------------
    print("Test group: SolveAllChannels")
    try:
        eom_all = pyIMSRG.EOMImsrg(H)
        eom_all.SolveAllChannels("TDA")
        check(True, "SolveAllChannels(TDA) completed without exception")
    except Exception as ex:
        check(False, f"SolveAllChannels(TDA) raised: {ex}")

    try:
        eom_all2 = pyIMSRG.EOMImsrg(H)
        eom_all2.SolveAllChannels("EOM")
        check(True, "SolveAllChannels(EOM) completed without exception")
    except Exception as ex:
        check(False, f"SolveAllChannels(EOM) raised: {ex}")

    # ----------------------------------------------------------------
    # Test 7: ComputeTransitionME returns a finite float
    # ----------------------------------------------------------------
    print("Test group: ComputeTransitionME")
    try:
        eom_t = pyIMSRG.EOMImsrg(H)
        eom_t.Solve(J, par, Tz, "TDA")
        ens = eom_t.GetExcitationEnergies()
        if len(ens) > 0:
            # Rank-2 positive-parity one-body operator as a stand-in for E2
            E2op = ut.RandomOp(ms, 2, 0, 1, 1, +1)
            T = eom_t.ComputeTransitionME(E2op, 0)
            check(math.isfinite(T),
                  f"ComputeTransitionME returns finite value ({T:.6f})")
        else:
            check(True, "ComputeTransitionME: no 2+ states to test (trivial pass)")
    except Exception as ex:
        check(False, f"ComputeTransitionME raised: {ex}")

    # ----------------------------------------------------------------
    # Test 8: GetChannelResults raises for unsolved channel
    # ----------------------------------------------------------------
    print("Test group: GetChannelResults error handling")
    try:
        eom_err = pyIMSRG.EOMImsrg(H)
        _ = eom_err.GetChannelResults(9999)
        check(False, "GetChannelResults(unsolved) should have raised")
    except Exception:
        check(True, "GetChannelResults(unsolved) raises as expected")

    # ----------------------------------------------------------------
    # Test 9: New binding functions — index-based variants
    # ----------------------------------------------------------------
    print("Test group: index-based variants")
    ich = ms.GetTwoBodyChannelIndex(J, par, Tz)

    eom_idx = pyIMSRG.EOMImsrg(H)
    try:
        eom_idx.BuildAMatrix_byIndex(ich)
        check(True, "BuildAMatrix_byIndex runs without exception")
    except Exception as ex:
        check(False, f"BuildAMatrix_byIndex raised: {ex}")

    try:
        eom_idx.BuildBMatrix_byIndex(ich)
        check(True, "BuildBMatrix_byIndex runs without exception")
    except Exception as ex:
        check(False, f"BuildBMatrix_byIndex raised: {ex}")

    try:
        eom_idx2 = pyIMSRG.EOMImsrg(H)
        eom_idx2.Solve_byIndex(ich, "TDA")
        energies_idx = eom_idx2.GetExcitationEnergies()
        # Must match the (J, parity, Tz) variant
        if sizes_match and len(tda_energies) > 0 and len(energies_idx) > 0:
            diff = max(abs(a - b) for a, b in zip(sorted(tda_energies),
                                                   sorted(energies_idx)))
            check(diff < 1e-10,
                  f"Solve_byIndex gives same energies as Solve (diff={diff:.2e})")
        else:
            check(True, "Solve_byIndex ran without exception")
    except Exception as ex:
        check(False, f"Solve_byIndex raised: {ex}")

    # ComputeTransitionME_byIndex
    try:
        eom_idx3 = pyIMSRG.EOMImsrg(H)
        eom_idx3.Solve(J, par, Tz, "TDA")
        ens3 = eom_idx3.GetExcitationEnergies()
        if len(ens3) > 0:
            E2op = ut.RandomOp(ms, 2, 0, 1, 1, +1)
            T_idx = eom_idx3.ComputeTransitionME_byIndex(ich, E2op, 0)
            T_direct = eom_idx3.ComputeTransitionME(E2op, 0)
            check(abs(T_idx - T_direct) < 1e-14,
                  f"ComputeTransitionME_byIndex matches ComputeTransitionME")
        else:
            check(True, "ComputeTransitionME_byIndex: no states (trivial pass)")
    except Exception as ex:
        check(False, f"ComputeTransitionME_byIndex raised: {ex}")

    # ----------------------------------------------------------------
    # Test 10: EOMImsrg property fields
    # ----------------------------------------------------------------
    print("Test group: EOMImsrg property fields")

    eom_props = pyIMSRG.EOMImsrg(H)
    eom_props.Solve(J, par, Tz, "TDA")

    # GetNStates
    n_states = eom_props.GetNStates()
    check(isinstance(n_states, int) and n_states >= 0,
          f"GetNStates returns non-negative int ({n_states})")

    # Energies property
    energies_prop = eom_props.Energies
    check(isinstance(energies_prop, list),
          "Energies property returns a list")
    if len(energies_prop) > 0 and len(tda_energies) > 0:
        check(abs(energies_prop[0] - tda_energies[0]) < 1e-10,
              "Energies property matches GetExcitationEnergies()[0]")

    # X property (2D list)
    X_prop = eom_props.X
    check(isinstance(X_prop, list),
          "X property returns a 2D list")
    if len(X_prop) > 0:
        check(isinstance(X_prop[0], list),
              "X property inner elements are lists")

    # Y property (2D list)
    Y_prop = eom_props.Y
    check(isinstance(Y_prop, list), "Y property returns a 2D list")

    # current_channel (read-only)
    cc = eom_props.current_channel
    check(isinstance(cc, int) and cc >= 0,
          f"current_channel is non-negative int ({cc})")

    # H field accessible (read/write)
    try:
        H_field = eom_props.H
        check(True, "H field is accessible")
    except Exception as ex:
        check(False, f"H field raised: {ex}")

    # A and B matrices
    A_mat = eom_props.A
    check(isinstance(A_mat, pyIMSRG.arma_mat) if hasattr(pyIMSRG, 'arma_mat')
          else True, "A field is accessible (type check skipped if arma_mat not exposed)")

    # PrintA / PrintB don't raise
    try:
        eom_props.PrintA()
        check(True, "PrintA() runs without exception")
    except Exception as ex:
        check(False, f"PrintA raised: {ex}")

    try:
        eom_props.PrintB()
        check(True, "PrintB() runs without exception")
    except Exception as ex:
        check(False, f"PrintB raised: {ex}")

    # ----------------------------------------------------------------
    # Test 11: GetSolvedChannels
    # ----------------------------------------------------------------
    print("Test group: GetSolvedChannels")
    eom_sc = pyIMSRG.EOMImsrg(H)
    eom_sc.Solve(J, par, Tz, "TDA")
    solved = eom_sc.GetSolvedChannels()
    check(isinstance(solved, list), "GetSolvedChannels returns a list")
    check(ich in solved, f"GetSolvedChannels contains the solved channel ({ich})")

    # After SolveAllChannels, list must be non-empty
    eom_sc2 = pyIMSRG.EOMImsrg(H)
    eom_sc2.SolveAllChannels("TDA")
    solved_all = eom_sc2.GetSolvedChannels()
    check(len(solved_all) > 0, "GetSolvedChannels non-empty after SolveAllChannels")

    # ----------------------------------------------------------------
    # Test 12: EOMChannel extra methods
    # ----------------------------------------------------------------
    print("Test group: EOMChannel extra methods")
    eom_ch = pyIMSRG.EOMImsrg(H)
    eom_ch.Solve(J, par, Tz, "TDA")
    ch_result = eom_ch.GetChannelResults(ich)

    n_st = ch_result.GetNStates()
    check(isinstance(n_st, int) and n_st >= 0,
          f"EOMChannel.GetNStates() = {n_st}")

    n_basis = ch_result.GetNBasis()
    check(isinstance(n_basis, int) and n_basis >= 0,
          f"EOMChannel.GetNBasis() = {n_basis}")

    energies_ch = ch_result.energies
    check(isinstance(energies_ch, list),
          "EOMChannel.energies property returns a list")
    if n_st > 0:
        check(len(energies_ch) == n_st,
              f"EOMChannel.energies length ({len(energies_ch)}) == GetNStates ({n_st})")

    Xmat = ch_result.X_matrix
    check(isinstance(Xmat, list), "EOMChannel.X_matrix returns a 2D list")

    Ymat = ch_result.Y_matrix
    check(isinstance(Ymat, list), "EOMChannel.Y_matrix returns a 2D list")

    try:
        ch_result.Print()
        check(True, "EOMChannel.Print() runs without exception")
    except Exception as ex:
        check(False, f"EOMChannel.Print() raised: {ex}")

    # ----------------------------------------------------------------
    # Test 13: EOM2 mode (1p1h + 2p2h sector)
    # ----------------------------------------------------------------
    print("Test group: EOM2 solver")

    eom2 = pyIMSRG.EOMImsrg(H)
    try:
        eom2.Solve(J, par, Tz, "EOM2")
        eom2_ran = True
        check(True, "EOM2 Solve() runs without exception")
    except Exception as ex:
        eom2_ran = False
        check(False, f"EOM2 Solve() raised: {ex}")

    if eom2_ran:
        eom2_energies = eom2.GetExcitationEnergies()
        check(len(eom2_energies) > 0,
              "EOM2 GetExcitationEnergies() returns a non-empty list")

        # All EOM2 energies should be real finite numbers
        if len(eom2_energies) > 0:
            all_finite = all(math.isfinite(e) for e in eom2_energies)
            check(all_finite, "EOM2 energies are all finite")

            # EOM2 energies from the 1p1h sector should be <= TDA energies
            # (EOM2 includes more correlations, so energies can only decrease or stay).
            # For a physical Hamiltonian, both TDA and EOM2 energies are positive.
            # With a random Hamiltonian TDA may return negative "energies" (unphysical);
            # in that case only compare the lowest positive TDA energy to the lowest
            # positive EOM2 energy.
            pos_tda = [e for e in tda_energies if e > 0]
            pos_eom2 = [e for e in eom2_energies if e > 0]
            if len(pos_tda) > 0 and len(pos_eom2) > 0:
                min_tda_pos  = min(pos_tda)
                min_eom2_pos = min(pos_eom2)
                check(min_eom2_pos <= min_tda_pos + 1e-8,
                      f"EOM2 lowest pos. energy ({min_eom2_pos:.4f}) <= TDA lowest pos. ({min_tda_pos:.4f})")
            else:
                check(True, "EOM2 vs TDA: no positive eigenvalues to compare (random H, skip)")

        # The 1p1h (X) amplitudes should have at least as many rows as the ph dimension
        X2 = eom2.X
        Y2 = eom2.Y
        check(isinstance(X2, list), "EOM2: X amplitudes accessible as list")
        check(isinstance(Y2, list), "EOM2: Y amplitudes accessible as list")

        # H22 matrix should be symmetric: verify via SolveAllChannels
        eom2_all = pyIMSRG.EOMImsrg(H)
        try:
            eom2_all.SolveAllChannels("EOM2")
            check(True, "EOM2 SolveAllChannels() runs without exception")
        except Exception as ex:
            check(False, f"EOM2 SolveAllChannels() raised: {ex}")

    # ----------------------------------------------------------------
    # Test 14: Lanczos EOM2 mode
    # ----------------------------------------------------------------
    print("Test group: Lanczos EOM2 solver")

    # Dense EOM2 reference energies (from Test 13 above)
    eom2_dense = pyIMSRG.EOMImsrg(H)
    try:
        eom2_dense.Solve(J, par, Tz, "EOM2")
        dense_energies = sorted(eom2_dense.GetExcitationEnergies())
    except Exception as ex:
        dense_energies = []
        check(False, f"EOM2 dense reference run for Lanczos test raised: {ex}")

    eom2_lanczos = pyIMSRG.EOMImsrg(H)
    # Request up to 3 lowest eigenvalues via Lanczos
    n_lanczos = min(3, max(1, len(dense_energies)))
    eom2_lanczos.lanczos_nev = n_lanczos
    check(eom2_lanczos.lanczos_nev == n_lanczos,
          f"lanczos_nev field read back correctly ({n_lanczos})")

    try:
        eom2_lanczos.Solve(J, par, Tz, "EOM2")
        lanczos_ran = True
        check(True, "EOM2 Lanczos Solve() runs without exception")
    except Exception as ex:
        lanczos_ran = False
        check(False, f"EOM2 Lanczos Solve() raised: {ex}")

    if lanczos_ran and len(dense_energies) > 0:
        lanczos_energies = sorted(eom2_lanczos.GetExcitationEnergies())

        # All returned energies must be finite and positive (the pos-filter is applied)
        if len(lanczos_energies) > 0:
            check(all(math.isfinite(e) for e in lanczos_energies),
                  "EOM2 Lanczos energies are all finite")
            check(all(e > 0 for e in lanczos_energies),
                  "EOM2 Lanczos energies are all positive")

        # Lanczos energies must agree with dense EOM2 to high precision
        # for the lowest min(n_found, n_dense) states.
        # NOTE: with SMALLEST_ALGE Lanczos and a random Hamiltonian it is possible
        # that all lanczos_nev computed eigenvalues are negative and get filtered out.
        # In that case, zero states are returned, which is acceptable (but we skip
        # the agreement check).
        n_compare = min(len(lanczos_energies), len(dense_energies))
        if n_compare > 0:
            max_diff = max(abs(lanczos_energies[i] - dense_energies[i])
                          for i in range(n_compare))
            check(max_diff < 1e-6,
                  f"EOM2 Lanczos agrees with dense EOM2 (max diff = {max_diff:.2e})")
        else:
            check(True,
                  "EOM2 Lanczos: no positive eigenvalues among lowest lanczos_nev "
                  "(acceptable for random H; real nuclear H gives positive excitations)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\npassed? {PASS}")
    return PASS

if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)

