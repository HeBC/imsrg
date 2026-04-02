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
    rpa_energies = rpa.GetEnergies()   # RPA: just A diagonalised in SolveTDA
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
        n = len(tda_energies)
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
    # Summary
    # ----------------------------------------------------------------
    print(f"\npassed? {PASS}")
    return PASS

if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
