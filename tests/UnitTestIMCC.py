#!/usr/bin/env python3

import math

import pyIMSRG


def require(condition, message):
    if not condition:
        raise AssertionError(message)


ms = pyIMSRG.ModelSpace(4, "He4", "He4")
ms.SetIMCCPartition(2)

c = ms.GetOrbitIndex_fromString("p0s1")
a = ms.GetOrbitIndex_fromString("p1s1")
x = ms.GetOrbitIndex_fromString("p2s1")

require(ms.GetOrbit(c).cvq == 0, "occupied p0s1 must be in C")
require(ms.GetOrbit(a).cvq == 1, "unoccupied p1s1 below the cutoff must be in A")
require(ms.GetOrbit(x).cvq == 2, "p2s1 above the cutoff must be in X")

H = pyIMSRG.Operator(ms, 0, 0, 0, 2)
H.SetHermitian()
for i, energy in ((c, 0.0), (a, 10.0), (x, 20.0)):
    H.SetOneBody(i, i, energy)

# Internal P-P and target P-X one-body couplings.
H.SetOneBody(c, a, 3.0)
H.SetOneBody(a, c, 3.0)
H.SetOneBody(c, x, 6.0)
H.SetOneBody(x, c, 6.0)
H.SetOneBody(a, x, 4.0)
H.SetOneBody(x, a, 4.0)

ch = ms.GetTwoBodyChannelIndex(0, 0, -1)
tbc = ms.GetTwoBodyChannel(ch)
icc = tbc.GetLocalIndex(c, c)
ica = tbc.GetLocalIndex(c, a)
icx = tbc.GetLocalIndex(c, x)
iaa = tbc.GetLocalIndex(a, a)
iax = tbc.GetLocalIndex(a, x)

require(ica in tbc.GetKetIndex_vc(), "C-A pair cache was not rebuilt")
require(icx in tbc.GetKetIndex_qc(), "C-X pair cache was not rebuilt")
require(iaa in tbc.GetKetIndex_vv(), "A-A pair cache was not rebuilt")
require(iax in tbc.GetKetIndex_qv(), "A-X pair cache was not rebuilt")

# Internal P-P and target P-X two-body couplings.
H.TwoBody.SetTBME_chij(ch, ch, ica, icc, 7.0)
H.TwoBody.SetTBME_chij(ch, ch, icc, ica, 7.0)
H.TwoBody.SetTBME_chij(ch, ch, iax, icc, 6.0)
H.TwoBody.SetTBME_chij(ch, ch, icc, iax, 6.0)
H.TwoBody.SetTBME_chij(ch, ch, iax, ica, 5.0)
H.TwoBody.SetTBME_chij(ch, ch, ica, iax, 5.0)
H.TwoBody.SetTBME_chij(ch, ch, icx, iaa, 4.25)
H.TwoBody.SetTBME_chij(ch, ch, iaa, icx, 4.25)
H.TwoBody.SetTBME_chij(ch, ch, iax, iaa, 4.5)
H.TwoBody.SetTBME_chij(ch, ch, iaa, iax, 4.5)

generator = pyIMSRG.Generator()
generator.SetType("imcc-atan")
require(generator.GetIMCCDecoupling() == "both", "default IM-CC block must be both")
Hod = generator.GetHod(H)

require(abs(Hod.GetOneBody(c, a)) < 1.0e-12, "C-A one-body coupling entered Hod")
require(abs(Hod.GetOneBody(c, x) - 6.0) < 1.0e-12, "X-C one-body coupling missing from Hod")
require(abs(Hod.GetOneBody(a, x) - 4.0) < 1.0e-12, "P-X one-body coupling missing from Hod")
require(abs(Hod.GetTwoBody(ch, ch, ica, icc)) < 1.0e-12, "P-P two-body coupling entered Hod")
require(abs(Hod.GetTwoBody(ch, ch, iax, icc) - 6.0) < 1.0e-12,
        "H_XC two-body coupling missing from Hod")
require(abs(Hod.GetTwoBody(ch, ch, iax, ica) - 5.0) < 1.0e-12, "P-X two-body coupling missing from Hod")
require(abs(Hod.GetTwoBody(ch, ch, icx, iaa) - 4.25) < 1.0e-12,
        "H_XC C-X-to-A-A coupling missing from Hod")
require(abs(Hod.GetTwoBody(ch, ch, iax, iaa) - 4.5) < 1.0e-12,
        "pure-particle H_XA coupling missing from Hod")

generator.SetIMCCDecoupling("xc")
Hod_xc = generator.GetHod(H)
require(abs(Hod_xc.GetOneBody(c, x) - 6.0) < 1.0e-12,
        "xc mode omitted X-C one-body coupling")
require(abs(Hod_xc.GetOneBody(a, x)) < 1.0e-12,
        "xc mode included X-A one-body coupling")
require(abs(Hod_xc.GetTwoBody(ch, ch, iax, icc) - 6.0) < 1.0e-12,
        "xc mode omitted X-containing-to-C-C coupling")
require(abs(Hod_xc.GetTwoBody(ch, ch, iax, ica) - 5.0) < 1.0e-12,
        "xc mode omitted X-A-to-C-A coupling")
require(abs(Hod_xc.GetTwoBody(ch, ch, icx, iaa) - 4.25) < 1.0e-12,
        "xc mode omitted X-C-to-A-A coupling")
require(abs(Hod_xc.GetTwoBody(ch, ch, iax, iaa)) < 1.0e-12,
        "xc mode included core-free X-A-to-A-A coupling")

generator.SetIMCCDecoupling("xa")
Hod_xa = generator.GetHod(H)
require(abs(Hod_xa.GetOneBody(c, x)) < 1.0e-12,
        "xa mode included X-C one-body coupling")
require(abs(Hod_xa.GetOneBody(a, x) - 4.0) < 1.0e-12,
        "xa mode omitted X-A one-body coupling")
require(abs(Hod_xa.GetTwoBody(ch, ch, iax, icc)) < 1.0e-12,
        "xa mode included X-containing-to-CC coupling")
require(abs(Hod_xa.GetTwoBody(ch, ch, iax, ica)) < 1.0e-12,
        "xa mode included a two-body term containing C")
require(abs(Hod_xa.GetTwoBody(ch, ch, icx, iaa)) < 1.0e-12,
        "xa mode included a C-X pair")
require(abs(Hod_xa.GetTwoBody(ch, ch, iax, iaa) - 4.5) < 1.0e-12,
        "xa mode omitted the pure-particle X-A-to-A-A coupling")

eta_xc = pyIMSRG.Operator(ms, 0, 0, 0, 2)
eta_xc.SetAntiHermitian()
generator.SetIMCCDecoupling("xc")
generator.Update(H, eta_xc)
require(abs(eta_xc.GetOneBody(c, x)) > 1.0e-12 and
        abs(eta_xc.GetOneBody(a, x)) < 1.0e-12,
        "xc generator used the wrong one-body block")
require(abs(eta_xc.GetTwoBody(ch, ch, iax, icc)) > 1.0e-12 and
        abs(eta_xc.GetTwoBody(ch, ch, iax, ica)) > 1.0e-12 and
        abs(eta_xc.GetTwoBody(ch, ch, icx, iaa)) > 1.0e-12 and
        abs(eta_xc.GetTwoBody(ch, ch, iax, iaa)) < 1.0e-12,
        "xc generator used the wrong two-body block")

eta_xa = pyIMSRG.Operator(ms, 0, 0, 0, 2)
eta_xa.SetAntiHermitian()
generator.SetIMCCDecoupling("xa")
generator.Update(H, eta_xa)
require(abs(eta_xa.GetOneBody(c, x)) < 1.0e-12 and
        abs(eta_xa.GetOneBody(a, x)) > 1.0e-12,
        "xa generator used the wrong one-body block")
require(abs(eta_xa.GetTwoBody(ch, ch, iax, icc)) < 1.0e-12 and
        abs(eta_xa.GetTwoBody(ch, ch, iax, ica)) < 1.0e-12 and
        abs(eta_xa.GetTwoBody(ch, ch, icx, iaa)) < 1.0e-12 and
        abs(eta_xa.GetTwoBody(ch, ch, iax, iaa)) > 1.0e-12,
        "xa generator used the wrong two-body block")

try:
    generator.SetIMCCDecoupling("invalid")
    require(False, "invalid IM-CC block was accepted")
except ValueError:
    pass

generator.SetIMCCDecoupling("both")

HP = generator.GetIMCCPBlock(H)
require(abs(HP.GetOneBody(c, a) - 3.0) < 1.0e-12, "P-block lost a C-A one-body coupling")
require(abs(HP.GetOneBody(a, x)) < 1.0e-12, "P-block retained an X one-body index")
require(abs(HP.GetTwoBody(ch, ch, ica, icc) - 7.0) < 1.0e-12,
        "P-block lost a P-P two-body coupling")
require(abs(HP.GetTwoBody(ch, ch, iax, ica)) < 1.0e-12,
        "P-block retained an X two-body index")
require(abs(HP.GetTwoBody(ch, ch, iax, iaa)) < 1.0e-12,
        "P-block retained a pure-particle X two-body index")

eta = pyIMSRG.Operator(ms, 0, 0, 0, 2)
eta.SetAntiHermitian()
generator.Update(H, eta)

require(abs(eta.GetOneBody(c, a)) < 1.0e-12, "generator changed an internal P-P one-body element")
require(abs(eta.GetOneBody(c, x)) > 1.0e-12, "generator omitted an X-C one-body element")
require(abs(eta.GetOneBody(a, x)) > 1.0e-12, "generator omitted a P-X one-body element")
require(abs(eta.GetOneBody(a, x) + eta.GetOneBody(x, a)) < 1.0e-12,
        "one-body eta is not anti-Hermitian")
require(abs(eta.GetTwoBody(ch, ch, ica, icc)) < 1.0e-12,
        "generator changed an internal P-P two-body element")
require(abs(eta.GetTwoBody(ch, ch, iax, icc)) > 1.0e-12,
        "generator omitted an H_XC two-body element")
require(abs(eta.GetTwoBody(ch, ch, iax, ica)) > 1.0e-12,
        "generator omitted a P-X two-body element")
require(abs(eta.GetTwoBody(ch, ch, icx, iaa)) > 1.0e-12,
        "generator omitted an H_XC C-X-to-A-A element")
require(abs(eta.GetTwoBody(ch, ch, iax, iaa)) > 1.0e-12,
        "generator omitted a pure-particle H_XA element")
require(abs(eta.GetTwoBody(ch, ch, iax, ica) + eta.GetTwoBody(ch, ch, ica, iax)) < 1.0e-12,
        "two-body eta is not anti-Hermitian")

norms = generator.GetIMCCOffDiagonalNorms(H)
require(len(norms) == 3 and all(math.isfinite(value) for value in norms),
        "invalid IM-CC residual norms")

print("UnitTestIMCC passed")
