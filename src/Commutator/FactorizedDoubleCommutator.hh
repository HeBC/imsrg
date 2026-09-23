///////////////////////////////////////////////////////////////////////////////////
//    FactorizedDoubleCommutator.hh, part of  imsrg++
//    Copyright (C) 2023 Bingcheng He and Ragnar Stroberg
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


#ifndef FactorizedDoubleCommutator_hh
#define FactorizedDoubleCommutator_hh 1

#include "Operator.hh"

namespace Commutator
{

 namespace FactorizedDoubleCommutator
 {
    extern bool use_goose_tank_1b;
    extern bool use_goose_tank_2b;

    extern bool use_1b_intermediates;
    extern bool use_2b_intermediates;

    extern bool use_goose_tank_only_1b;
    extern bool use_goose_tank_only_2b;
    extern bool use_TypeII_1b;
    extern bool use_TypeIII_1b;
    extern bool use_TypeII_2b;
    extern bool use_TypeIII_2b;

    extern bool use_GT_TypeI_2b;
    extern bool use_GT_TypeIV_2b;


    void SetUse_GooseTank_1b(bool tf);
    void SetUse_GooseTank_2b(bool tf);
    void SetUse_1b_Intermediates(bool tf);
    void SetUse_2b_Intermediates(bool tf);

    void SetUse_GooseTank_only_1b(bool tf);
    void SetUse_GooseTank_only_2b(bool tf);
    void SetUse_TypeII_1b(bool tf);
    void SetUse_TypeIII_1b(bool tf);
    void SetUse_TypeII_2b(bool tf);
    void SetUse_TypeIII_2b(bool tf);

    void SetUse_GT_TypeI_2b(bool tf);
    void SetUse_GT_TypeIV_2b(bool tf);


//    extern bool SlowVersion;
//    void UseSlowVersion(bool tf);
    // factorize double commutator [Eta, [Eta, Gamma]]
    void comm223_231(const Operator &Eta, const Operator &Gamma, Operator &Z);
    void comm223_232(const Operator &Eta, const Operator &Gamma, Operator &Z);

    // Ordered [[Gamma_2, EtaInner_2]_3, EtaOuter_2]_{1,2}.
    // Gamma and Z are Hermitian scalars; both Etas are anti-Hermitian scalars.
    // Only the indicated output sector is accumulated. No three-body storage.
    void comm223_231(const Operator &EtaOuter, const Operator &EtaInner, const Operator &Gamma, Operator &Z);
    void comm223_232(const Operator &EtaOuter, const Operator &EtaInner, const Operator &Gamma, Operator &Z);

    // Factorized kernels: non-reduced, scalar inputs; use the public overloads
    // above for validation, representation conversion, and output aliasing.
    void comm223_231_chi1b_distinct(const Operator &EtaOuter, const Operator &EtaInner, const Operator &Gamma, Operator &Z);
    void comm223_231_chi2b_distinct(const Operator &EtaOuter, const Operator &EtaInner, const Operator &Gamma, Operator &Z);
    void comm223_232_chi1b_distinct(const Operator &EtaOuter, const Operator &EtaInner, const Operator &Gamma, Operator &Z);
    void comm223_232_chi2b_distinct(const Operator &EtaOuter, const Operator &EtaInner, const Operator &Gamma, Operator &Z);

    void comm223_231_chi1b(const Operator &Eta, const Operator &Gamma, Operator &Z);
    void comm223_231_chi2b(const Operator &Eta, const Operator &Gamma, Operator &Z);
    void comm223_232_chi1b(const Operator &Eta, const Operator &Gamma, Operator &Z);
    void comm223_232_chi2b(const Operator &Eta, const Operator &Gamma, Operator &Z);


    // Internal direct-reference contraction; implemented in this module's .cc.
    // Optimized production kernels do not call this helper.
    namespace detail
    {
      double Contract223(const Operator &X, const Operator &Y,
                         int Jab, int Jde, int twoJ,
                         size_t a, size_t b, size_t c,
                         size_t d, size_t e, size_t f);
    }

 } //namespace FactorizedDoubleCommutator
} //namespace Commutator

#endif
