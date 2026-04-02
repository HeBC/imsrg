#!/usr/bin/env python3
"""
eom_imsrg_example.py
====================
Example driver for the EOM-IMSRG excited-state solver.

Workflow:
  1. Build a toy model space (emax=2, ¹⁶O-like reference).
  2. Create a random Hermitian Hamiltonian.
  3. Run a schematic IMSRG-like evolution (here we skip the real IMSRG flow
     and just use the HF-normal-ordered Hamiltonian as H_evolved, to keep
     the example self-contained without large interaction files).
  4. Solve the EOM-IMSRG in TDA and full EOM mode for J=2⁺ and J=3⁻ channels.
  5. Print the lowest few excitation energies.
  6. Compute a schematic E2 transition matrix element.

Usage:
  python3 eom_imsrg_example.py
"""

import sys
import pyIMSRG

def main():
    # ------------------------------------------------------------------
    # 1. Model space: emax=2, core=He4, valence+qspace = He4 continuum
    # ------------------------------------------------------------------
    emax = 2
    reference = 'He4'
    ms = pyIMSRG.ModelSpace(emax, reference, reference)
    ms.PreCalculateSixJ()

    print("=== EOM-IMSRG Example ===")
    print(f"  Model space: emax={emax}, reference={reference}")
    print(f"  Number of orbits: {ms.GetNumberOrbits()}")

    # ------------------------------------------------------------------
    # 2. Build a random Hermitian scalar 2-body operator as a stand-in
    #    for an IMSRG-evolved Hamiltonian.
    # ------------------------------------------------------------------
    ut = pyIMSRG.UnitTest(ms)
    # jrank=0, tz=0, parity=0, particle_rank=2, hermitian=+1
    H = ut.RandomOp(ms, 0, 0, 0, 2, +1)
    print("  Using a random Hermitian 2-body operator as H_evolved.")

    # ------------------------------------------------------------------
    # 3. Create the EOM-IMSRG solver
    # ------------------------------------------------------------------
    eom = pyIMSRG.EOMImsrg(H)

    # ------------------------------------------------------------------
    # 4a. Solve TDA in several (J, parity, Tz) channels
    # ------------------------------------------------------------------
    channels_to_solve = [
        (0, 1, 0, "0+"),
        (1, -1, 0, "1-"),
        (2, 1, 0, "2+"),
        (3, -1, 0, "3-"),
    ]

    print("\n--- TDA (Tamm-Dancoff Approximation) excitation energies ---")
    for J, par, Tz, label in channels_to_solve:
        try:
            eom.Solve(J, par, Tz, "TDA")
            energies = eom.GetExcitationEnergies()
            if len(energies) == 0:
                print(f"  {label}: no 1p-1h pairs in this channel")
                continue
            lowest = energies[:min(3, len(energies))]
            print(f"  {label}: lowest {len(lowest)} energies = "
                  + ", ".join(f"{e:.4f}" for e in lowest) + " MeV")
        except Exception as ex:
            print(f"  {label}: skipped ({ex})")

    # ------------------------------------------------------------------
    # 4b. Solve full EOM (with B matrix) for 2+ channel
    # ------------------------------------------------------------------
    print("\n--- Full EOM excitation energies (J=2+) ---")
    try:
        eom.Solve(2, 1, 0, "EOM")
        energies = eom.GetExcitationEnergies()
        if len(energies) > 0:
            lowest = energies[:min(3, len(energies))]
            print("  J=2+: " + ", ".join(f"{e:.4f}" for e in lowest) + " MeV")
        else:
            print("  J=2+: no 1p-1h pairs")
    except Exception as ex:
        print(f"  J=2+: skipped ({ex})")

    # ------------------------------------------------------------------
    # 5. Compute a schematic transition matrix element for the lowest 2+ state.
    #    We use another random rank-2 operator as a stand-in for E2.
    # ------------------------------------------------------------------
    print("\n--- Transition matrix element for lowest 2+ state ---")
    try:
        eom.Solve(2, 1, 0, "TDA")
        energies = eom.GetExcitationEnergies()
        if len(energies) > 0:
            # jrank=2, tz=0, parity=1 (positive), particle_rank=1, hermitian=+1
            E2op = ut.RandomOp(ms, 2, 0, 1, 1, +1)
            T = eom.ComputeTransitionME(E2op, 0)
            print(f"  <0 || E2 || 2+_1> = {T:.6f}  (random op, schematic)")
        else:
            print("  No 2+ states found.")
    except Exception as ex:
        print(f"  Transition ME: skipped ({ex})")

    # ------------------------------------------------------------------
    # 6. Solve all channels at once; inspect via GetSolvedChannels
    # ------------------------------------------------------------------
    print("\n--- SolveAllChannels (TDA) ---")
    eom2 = pyIMSRG.EOMImsrg(H)
    eom2.SolveAllChannels("TDA")
    solved_channels = eom2.GetSolvedChannels()
    print(f"  SolveAllChannels completed. Solved {len(solved_channels)} channels.")

    # ------------------------------------------------------------------
    # 7. Demonstrate index-based API and EOMChannel introspection
    # ------------------------------------------------------------------
    print("\n--- Index-based API ---")
    ich = ms.GetTwoBodyChannelIndex(2, 1, 0)   # J=2+
    eom3 = pyIMSRG.EOMImsrg(H)
    eom3.Solve_byIndex(ich, "TDA")
    ch_result = eom3.GetChannelResults(ich)
    n_states = ch_result.GetNStates()
    n_basis  = ch_result.GetNBasis()
    print(f"  J=2+ channel (ich={ich}): {n_states} states, {n_basis} 1p-1h basis pairs")
    if n_states > 0:
        print(f"  Lowest energy (from EOMChannel): {ch_result.energies[0]:.4f} MeV")
        ch_result.Print()

    # ------------------------------------------------------------------
    # 8. Demonstrate EOMImsrg property accessors
    # ------------------------------------------------------------------
    print("\n--- Property accessors ---")
    eom4 = pyIMSRG.EOMImsrg(H)
    eom4.Solve(2, 1, 0, "TDA")
    print(f"  current_channel = {eom4.current_channel}")
    print(f"  GetNStates()    = {eom4.GetNStates()}")
    ens = eom4.Energies
    if ens:
        print(f"  Energies[0]     = {ens[0]:.4f} MeV  (via .Energies property)")
    X_rows = len(eom4.X)
    X_cols = len(eom4.X[0]) if X_rows > 0 else 0
    print(f"  X matrix shape  = {X_rows} x {X_cols}")
    eom4.PrintA()

    print("\nDone.")

if __name__ == '__main__':
    main()

