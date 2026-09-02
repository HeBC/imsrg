# IM-CC three-space downfolding

This branch implements an experimental IMSRG(2) downfolding for a subsequent
coupled-cluster calculation in a reduced orbital space.

## Space definition

The parent single-particle space is partitioned as

- `C`: occupied reference orbits;
- `A`: unoccupied retained orbits with `2*n+l <= imcc_emax`;
- `X`: excluded orbits with `2*n+l > imcc_emax`;
- `P = C + A`: the orbital space passed to the reduced CC solver.

Internally, `C`, `A`, and `X` use the existing `core`, `valence`, and `qspace`
containers. `ModelSpace::SetIMCCPartition()` rebuilds every cached two-body ket
classification after changing the partition.

The target block condition is

```text
Q_X Hbar P = 0,    P Hbar Q_X = 0,
```

where `Q_X` contains configurations with excluded-space content. At IMSRG(2),
the implemented off-diagonal mask contains:

- one-body elements with one index in `P` and one in `X`;
- two-body elements connecting a pair wholly in `P` (`cc`, `vc`, or `vv`) to
  a pair containing at least one `X` orbit (`qc`, `qv`, or `qq`).

The mask intentionally does not include `C-A`, `P-P`, or `X-X` matrix
elements. Low-energy correlations inside `P` therefore remain available to CC.

For diagnostic calculations, the mask can be split into two disjoint blocks:

- `xc`: one-body `X-C` elements and the part of the two-body P-X mask
  containing at least one `C` index;
- `xa`: one-body `X-A` elements and the core-free, pure-particle two-body
  block, namely an `XA` or `XX` pair (`qv` or `qq` in the legacy cache
  names) coupled to an `AA` pair (`vv`);
- `both`: the union of `xc` and `xa`, and the default.

Only `both` represents the complete implemented `P-X` decoupling. The partial
options are intended to diagnose the separate effects of the two blocks; a
reduced-space CC Hamiltonian obtained from a partial flow is not closed under
coupling to `X`. A block omitted from the generator is not frozen: it still
evolves through `[eta,H]` and can be regenerated. If `xc` is evolved first in
a staged calculation, use `both` for the final stage if the final Hamiltonian
must keep both blocks suppressed.

## Command-line use

Enable the mode by supplying a retained cutoff smaller than the parent IMSRG
cutoff:

```bash
imsrg++ \
  2bme=INTERACTION_FILE fmt2=me2j \
  reference=O16 valence_space=O16 \
  emax=12 emax_imsrg=12 e3max=24 \
  imcc_emax=8 imcc_generator=atan imcc_decoupling=both \
  method=magnus
```

Setting `imcc_emax` automatically selects a one-step generator named
`imcc-<imcc_generator>`. Available functional forms are `atan` (recommended
starting point), `white`, `imaginary-time`, and `wegner`.

Use `imcc_decoupling=xc` to test the `H_XC` block alone or
`imcc_decoupling=xa` to test `H_XA` alone. The default is `both`.

The existing constant denominator shift is also available. To apply it to all
one- and two-body IM-CC denominators, both parameters are required:

```bash
denominator_delta=2.0 denominator_delta_orbit=all
```

Setting `denominator_delta` without `denominator_delta_orbit=all` does not
apply a global shift. This is a literal additive shift, `Delta -> Delta+delta`,
and can therefore move or reverse small denominators; it should be varied as a
diagnostic rather than treated as an intruder-free prescription.

The run prints the initial and final values of

```text
||Hod_1b||, ||Hod_2b||, ||Hod||, and ||Hod||/||H||.
```

These are direct diagnostics of the requested `P-X` decoupling. The parent
Hamiltonian remains allocated during the flow. Existing valence-space writers
see `C` as the core and `A` as the retained non-core space. For a CC interface,
`Generator::GetIMCCPBlock()` returns `P Hbar P` in the parent allocation with
every matrix element carrying an `X` index set to zero. The exporter should
write only the `core + valence` orbit indices, and CC particle amplitudes must
be restricted to `A`.

## Python use

```python
ms.SetIMCCPartition(emax_cut)

generator = pyIMSRG.Generator()
generator.SetType("imcc-atan")
generator.SetIMCCDecoupling("both")  # or "xc", "xa"
generator.Update(H, eta)

one_body, two_body, total = generator.GetIMCCOffDiagonalNorms(H)
Hod = generator.GetHod_IMCC(H)
H_for_cc = generator.GetIMCCPBlock(H)
```

For an `IMSRGSolver`, the equivalent Python setup is

```python
solver.SetGenerator("imcc-atan")
solver.SetIMCCDecoupling("both")
solver.SetDenominatorDelta(2.0)
solver.SetDenominatorDeltaOrbit("all")
```

## Present scope and validation

The command-line mode is deliberately restricted to IMSRG(2)/NO2B. It does
not claim a three-body `P-X` decoupling, and induced residual three- and
higher-body interactions remain a truncation error.

For physics validation, compare at each cutoff:

1. full-space CC or exact diagonalization where possible;
2. reduced-space CC with the bare truncated Hamiltonian;
3. reduced-space CC with the IM-CC downfolded Hamiltonian.

Also scan the cutoff, parent `emax`, oscillator frequency, generator, and
reference. A usable calculation should strongly suppress the reported
off-diagonal norm without large generator norms, denominator pathologies, or
non-smooth cutoff dependence.
