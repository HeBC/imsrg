
#include "TensorCommutators.hh"
#include "Commutator.hh"
#include "AngMom.hh"
#include "PhysicalConstants.hh"

namespace Commutator
{


  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  ////////////   BEGIN SCALAR-TENSOR COMMUTATORS      //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  //*****************************************************************************************
  //
  //        |____. Y          |___.X
  //        |        _        |
  //  X .___|            Y.___|              [X1,Y1](1)  =  XY - YX
  //        |                 |
  //
  // This is no different from the scalar-scalar version
  void comm111st(const Operator &X, const Operator &Y, Operator &Z)
  {
    double tstart = omp_get_wtime();
    Z.OneBody += X.OneBody * Y.OneBody - Y.OneBody * X.OneBody;
    X.profiler.timer["comm111st"] += omp_get_wtime() - tstart;
  }

  //*****************************************************************************************
  //                                       |
  //      i |              i |             |
  //        |    ___.Y       |__X__        |
  //        |___(_)    _     |   (_)__.    |  [X2,Y1](1)  =  1/(2j_i+1) sum_ab(n_a-n_b)y_ab
  //      j | X            j |        Y    |        * sum_J (2J+1) x_biaj^(J)
  //                                       |
  //---------------------------------------*        = 1/(2j+1) sum_a n_a sum_J (2J+1)
  //                                                  * sum_b y_ab x_biaj - yba x_aibj
  //
  // X is scalar one-body, Y is tensor two-body
  // There must be a better way to do this looping.
  //
  // void Operator::comm121st( Operator& Y, Operator& Z)
  void comm121st(const Operator &X, const Operator &Y, Operator &Z)
  {

    double tstart = omp_get_wtime();
    //   int norbits = Z.modelspace->GetNumberOrbits();
    int norbits = Z.modelspace->all_orbits.size();
    int Lambda = Z.GetJRank();
    int hZ = Z.IsHermitian() ? +1 : -1;
    Z.modelspace->PreCalculateSixJ();
    std::vector<index_t> allorb_vec(Z.modelspace->all_orbits.begin(), Z.modelspace->all_orbits.end());
    //   #pragma omp parallel for schedule(dynamic,1) if (not Z.modelspace->tensor_transform_first_pass.at(Z.GetJRank()*2+Z.GetParity()) )
    #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
    for (int indexi = 0; indexi < norbits; ++indexi)
    //   for (int i=0;i<norbits;++i)
    {
      //      auto i = Z.modelspace->all_orbits[indexi];
      auto i = allorb_vec[indexi];
      Orbit &oi = Z.modelspace->GetOrbit(i);
      double ji = 0.5 * oi.j2;
      //      for (auto j : Z.OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) )
      for (auto j : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
      {
        Orbit &oj = Z.modelspace->GetOrbit(j);
        double jj = 0.5 * oj.j2;
        if (j < i)
          continue; // only calculate upper triangle
                    //          double& Zij = Z.OneBody(i,j);
        double zij = 0;
        int phase_ij = AngMom::phase((oi.j2 - oj.j2) / 2);
        for (auto a : Z.modelspace->holes) // C++11 syntax
        {
          Orbit &oa = Z.modelspace->GetOrbit(a);
          double ja = 0.5 * oa.j2;
          //             for (auto b : X.OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) )
          for (auto b : X.GetOneBodyChannel(oa.l, oa.j2, oa.tz2))
          {
            Orbit &ob = Z.modelspace->GetOrbit(b);
            //                  if ( not ( i==0 and j==4 and a==0 and b==10) ) continue;
            double nanb = oa.occ * (1 - ob.occ);
            int J1min = std::abs(ji - ja);
            int J1max = ji + ja;
            for (int J1 = J1min; J1 <= J1max; ++J1)
            {
              int phasefactor = Z.modelspace->phase(jj + ja + J1 + Lambda);
              int J2min = std::max(std::abs(Lambda - J1), std::abs(int(ja - jj)));
              int J2max = std::min(Lambda + J1, int(ja + jj));
              for (int J2 = J2min; J2 <= J2max; ++J2)
              {
                if (!(J2 >= std::abs(ja - jj) and J2 <= ja + jj))
                  continue;
                double prefactor = nanb * phasefactor * sqrt((2 * J1 + 1) * (2 * J2 + 1)) * Z.modelspace->GetSixJ(J1, J2, Lambda, jj, ji, ja);
                zij += prefactor * (X.OneBody(a, b) * Y.TwoBody.GetTBME_J(J1, J2, b, i, a, j) - X.OneBody(b, a) * Y.TwoBody.GetTBME_J(J1, J2, a, i, b, j));
              }
            }
          }
          // Now, X is scalar two-body and Y is tensor one-body
          //             for (auto& b : Y.OneBodyChannels.at({oa.l,oa.j2,oa.tz2}) )
          for (auto &b : Y.GetOneBodyChannel(oa.l, oa.j2, oa.tz2))
          {
            //                continue;

            Orbit &ob = Z.modelspace->GetOrbit(b);
            double jb = 0.5 * ob.j2;
            if (std::abs(ob.occ - 1) < ModelSpace::OCC_CUT)
              continue;
            double nanb = oa.occ * (1 - ob.occ);
            int J1min = std::max(std::abs(ji - jb), std::abs(jj - ja));
            int J1max = std::min(ji + jb, jj + ja);
            double Xbar_ijab = 0;
            for (int J1 = J1min; J1 <= J1max; ++J1)
            {
              //                  Xtmp -= Z.modelspace->phase(ji+jb+J1) * (2*J1+1) * Z.modelspace->GetSixJ(ja,jb,Lambda,ji,jj,J1) * X.TwoBody.GetTBME_J(J1,J1,b,i,a,j);
              Xbar_ijab -= (2 * J1 + 1) * Z.modelspace->GetSixJ(ji, jj, Lambda, ja, jb, J1) * X.TwoBody.GetTBME_J(J1, J1, i, b, a, j);
            }

            zij -= nanb * Y.OneBody(a, b) * Xbar_ijab;
            //                Xtmp = 0;
            double Xbar_ijba = 0;
            J1min = std::max(std::abs(ji - ja), std::abs(jj - jb));
            J1max = std::min(ji + ja, jj + jb);
            for (int J1 = J1min; J1 <= J1max; ++J1)
            {
              //                  Xtmp += Z.modelspace->phase(ji+jb+J1) * (2*J1+1) * Z.modelspace->GetSixJ(jb,ja,Lambda,ji,jj,J1) * X.TwoBody.GetTBME_J(J1,J1,a,i,b,j) ;
              Xbar_ijba -= (2 * J1 + 1) * Z.modelspace->GetSixJ(ji, jj, Lambda, jb, ja, J1) * X.TwoBody.GetTBME_J(J1, J1, i, a, b, j);
            }

            zij += nanb * Y.OneBody(b, a) * Xbar_ijba;
          }

        } // for a
        Z.OneBody(i, j) += zij;
        if (i != j)
          Z.OneBody(j, i) += hZ * phase_ij * zij; // we're dealing with reduced matrix elements, which get a phase under hermitian conjugation
      } // for j
    } // for i

    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  }

  //*****************************************************************************************
  //
  //    |     |               |      |           [X2,Y1](2) = sum_a ( Y_ia X_ajkl + Y_ja X_iakl - Y_ak X_ijal - Y_al X_ijka )
  //    |     |___.Y          |__X___|
  //    |     |         _     |      |
  //    |_____|               |      |_____.Y
  //    |  X  |               |      |
  //
  // -- AGREES WITH NATHAN'S RESULTS
  // Right now, this is the slowest one...
  // Agrees with previous code in the scalar-scalar limit
  // void Operator::comm122st( Operator& Y, Operator& Z )
  // void Operator::comm122st( const Operator& X, const Operator& Y )
  void comm122st(const Operator &X, const Operator &Y, Operator &Z)
  {
    double tstart = omp_get_wtime();
    int Lambda = Z.rank_J;
    int hZ = Z.IsHermitian() ? +1 : -1;

    std::vector<int> bra_channels;
    std::vector<int> ket_channels;

    for (auto &itmat : Z.TwoBody.MatEl)
    {
      bra_channels.push_back(itmat.first[0]);
      ket_channels.push_back(itmat.first[1]);
    }
    int nmat = bra_channels.size();
    //   #pragma omp parallel for schedule(dynamic,1) if (not Z.modelspace->tensor_transform_first_pass[Z.GetJRank()*2+Z.GetParity()])
    #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
    for (int ii = 0; ii < nmat; ++ii)
    {
      int ch_bra = bra_channels[ii];
      int ch_ket = ket_channels[ii];

      TwoBodyChannel &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      int nbras = tbc_bra.GetNumberKets();
      int nkets = tbc_ket.GetNumberKets();
      double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1));
      arma::mat &Z2 = Z.TwoBody.GetMatrix(ch_bra, ch_ket);

      for (int ibra = 0; ibra < nbras; ++ibra)
      {
        Ket &bra = tbc_bra.GetKet(ibra);
        int i = bra.p;
        int j = bra.q;
        Orbit &oi = Z.modelspace->GetOrbit(i);
        Orbit &oj = Z.modelspace->GetOrbit(j);
        double ji = oi.j2 / 2.0;
        double jj = oj.j2 / 2.0;
        for (int iket = 0; iket < nkets; ++iket)
        {
          Ket &ket = tbc_ket.GetKet(iket);
          int k = ket.p;
          int l = ket.q;
          Orbit &ok = Z.modelspace->GetOrbit(k);
          Orbit &ol = Z.modelspace->GetOrbit(l);
          double jk = ok.j2 / 2.0;
          double jl = ol.j2 / 2.0;

          double cijkl = 0;
          double c1 = 0;
          double c2 = 0;
          double c3 = 0;
          double c4 = 0;
          if ((ch_bra==ch_ket)  and (iket>ibra)) continue;


          //            for ( int a : X.OneBodyChannels.at({oi.l,oi.j2,oi.tz2}) )
          for (int a : X.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
          {
            //            c1 += X.OneBody(i, a) * Y.TwoBody.GetTBME(ch_bra, ch_ket, a, j, k, l);
            c1 += X.OneBody(i, a) * Y.TwoBody.GetTBME_J(J1, J2, a, j, k, l);
          }
          if (i == j)
          {
            c2 = c1; // there should be a phase here, but if the ket exists, it'd better be +1.
          }
          else
          {
            //              for ( int a : X.OneBodyChannels.at({oj.l,oj.j2,oj.tz2}) )
            for (int a : X.GetOneBodyChannel(oj.l, oj.j2, oj.tz2))
            {
              //              c2 += X.OneBody(j, a) * Y.TwoBody.GetTBME(ch_bra, ch_ket, i, a, k, l);
              c2 += X.OneBody(j, a) * Y.TwoBody.GetTBME_J(J1, J2, i, a, k, l);
            }
          }
          //            for ( int a : X.OneBodyChannels.at({ok.l,ok.j2,ok.tz2}) )
          for (int a : X.GetOneBodyChannel(ok.l, ok.j2, ok.tz2))
          {
            //            c3 += X.OneBody(a, k) * Y.TwoBody.GetTBME(ch_bra, ch_ket, i, j, a, l);
            c3 += X.OneBody(a, k) * Y.TwoBody.GetTBME_J(J1, J2, i, j, a, l);
          }
          if (k == l)
          {
            c4 = c3;
          }
          else
          {
            //              for ( int a : X.OneBodyChannels.at({ol.l,ol.j2,ol.tz2}) )
            for (int a : X.GetOneBodyChannel(ol.l, ol.j2, ol.tz2))
            {
              //              c4 += X.OneBody(a, l) * Y.TwoBody.GetTBME(ch_bra, ch_ket, i, j, k, a);
              c4 += X.OneBody(a, l) * Y.TwoBody.GetTBME_J(J1, J2, i, j, k, a);
            }
          }

          cijkl = c1 + c2 - c3 - c4;

          c1 = 0;
          c2 = 0;
          c3 = 0;
          c4 = 0;
          int phase1 = Z.modelspace->phase(ji + jj + J2 + Lambda);
          int phase2 = Z.modelspace->phase(J1 - J2 + Lambda);
          int phase3 = Z.modelspace->phase(J1 - J2 + Lambda);
          int phase4 = Z.modelspace->phase(jk + jl - J1 + Lambda);

          for (int a : Y.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
          {
            double ja = Z.modelspace->GetOrbit(a).j2 * 0.5;
            if (not AngMom::Triangle(J2, ja, jj))
              continue;
            //            c1 -= Z.modelspace->GetSixJ(J2, J1, Lambda, ji, ja, jj) * Y.OneBody(i, a) * X.TwoBody.GetTBME(ch_ket, ch_ket, a, j, k, l);
            c1 -= Z.modelspace->GetSixJ(J2, J1, Lambda, ji, ja, jj) * Y.OneBody(i, a) * X.TwoBody.GetTBME_J(J2, J2, a, j, k, l);
          }

          if (false and i == j)
          {
            c2 = -c1;
          }
          else
          {
            for (int a : Y.GetOneBodyChannel(oj.l, oj.j2, oj.tz2))
            {
              double ja = Z.modelspace->GetOrbit(a).j2 * 0.5;
              if (not AngMom::Triangle(J2, ja, ji))
                continue;
              //              c2 += Z.modelspace->GetSixJ(J2, J1, Lambda, jj, ja, ji) * Y.OneBody(j, a) * X.TwoBody.GetTBME(ch_ket, ch_ket, a, i, k, l);
              c2 += Z.modelspace->GetSixJ(J2, J1, Lambda, jj, ja, ji) * Y.OneBody(j, a) * X.TwoBody.GetTBME_J(J2, J2, a, i, k, l);
            }
          }
          for (int a : Y.GetOneBodyChannel(ok.l, ok.j2, ok.tz2))
          {
            double ja = Z.modelspace->GetOrbit(a).j2 * 0.5;
            if (not AngMom::Triangle(J1, ja, jl))
              continue;
            //            c3 -= Z.modelspace->GetSixJ(J1, J2, Lambda, jk, ja, jl) * Y.OneBody(a, k) * X.TwoBody.GetTBME(ch_bra, ch_bra, i, j, l, a);
            c3 -= Z.modelspace->GetSixJ(J1, J2, Lambda, jk, ja, jl) * Y.OneBody(a, k) * X.TwoBody.GetTBME_J(J1, J1, i, j, l, a);
          }
          if (k == l)
          {
            c4 = -c3;
          }
          else
          {
            for (int a : Y.GetOneBodyChannel(ol.l, ol.j2, ol.tz2))
            {
              double ja = Z.modelspace->GetOrbit(a).j2 * 0.5;
              if (not AngMom::Triangle(J1, ja, jk))
                continue;
              //              c4 += Z.modelspace->GetSixJ(J1, J2, Lambda, jl, ja, jk) * Y.OneBody(a, l) * X.TwoBody.GetTBME(ch_bra, ch_bra, i, j, k, a);
              c4 += Z.modelspace->GetSixJ(J1, J2, Lambda, jl, ja, jk) * Y.OneBody(a, l) * X.TwoBody.GetTBME_J(J1, J1, i, j, k, a);
            }
          }
          cijkl += hatfactor * (phase1 * c1 + phase2 * c2 + phase3 * c3 + phase4 * c4);

          double norm = bra.delta_pq() == ket.delta_pq() ? 1 + bra.delta_pq() : PhysConst::SQRT2;
          Z2(ibra, iket) += cijkl / norm;
          if ((ch_bra == ch_ket) and (iket < ibra))
            Z2(iket, ibra) += hZ * Z.modelspace->phase(J1 - J2) * cijkl / norm;
          
        }
      }
    }
    // if (X.GetParity()==1 and Y.GetParity()==1)
    // {
    //   Z.PrintTwoBody();
    // }
    X.profiler.timer["comm122st"] += omp_get_wtime() - tstart;
  }

  // Since comm222_pp_hh and comm211 both require the construction of
  // the intermediate matrices Mpp and Mhh, we can combine them and
  // only calculate the intermediates once.
  // X is a scalar, Y is a tensor
  // void Operator::comm222_pp_hh_221st( Operator& Y, Operator& Z )
  // void Operator::comm222_pp_hh_221st( const Operator& X, const Operator& Y )
  void comm222_pp_hh_221st(const Operator &X, const Operator &Y, Operator &Z)
  {

    double tstart = omp_get_wtime();
    int Lambda = Z.GetJRank();
    int hX = X.IsHermitian() ? +1 : -1;
    int hY = Y.IsHermitian() ? +1 : -1;
    int hZ = Z.IsHermitian() ? +1 : -1;

    TwoBodyME Mpp = Z.TwoBody;
    TwoBodyME Mhh = Z.TwoBody;
    // (not used)   TwoBodyME Mff = Z.TwoBody;

    std::vector<int> vch_bra;
    std::vector<int> vch_ket;
    std::vector<const arma::mat *> vmtx;
    //   for ( auto& itmat : Y.TwoBody.MatEl )
    for (auto &itmat : Z.TwoBody.MatEl)
    {
      vch_bra.push_back(itmat.first[0]);
      vch_ket.push_back(itmat.first[1]);
      vmtx.push_back(&(itmat.second));
    }
    if (X.GetTRank() != 0)
    {
      std::cout << "Uh Oh. " << __func__ << "  can't handle an isospin-changing scalar operator (not yet implemented). Dying." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    size_t nchan = vch_bra.size();
    //   #pragma omp parallel for schedule(dynamic,1)
    for (size_t i = 0; i < nchan; ++i)
    {
      int ch_bra = vch_bra[i];
      int ch_ket = vch_ket[i];

      TwoBodyChannel &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      size_t ch_XY = Z.modelspace->GetTwoBodyChannelIndex(tbc_bra.J, (tbc_bra.parity + X.parity) % 2, tbc_bra.Tz);
      TwoBodyChannel &tbc_XY = Z.modelspace->GetTwoBodyChannel(ch_XY);
      size_t ch_YX = Z.modelspace->GetTwoBodyChannelIndex(tbc_ket.J, (tbc_ket.parity + X.parity) % 2, tbc_ket.Tz);
      TwoBodyChannel &tbc_YX = Z.modelspace->GetTwoBodyChannel(ch_YX);

      //    auto& LHS1 = X.TwoBody.GetMatrix(ch_bra,ch_bra);
      //    auto& LHS2 = X.TwoBody.GetMatrix(ch_ket,ch_ket);

      // Should be Xdir * Ydir - Yexc * Xexc

      auto &Xdir = ch_bra <= ch_XY ? X.TwoBody.GetMatrix(ch_bra, ch_XY)
                                   : X.TwoBody.GetMatrix(ch_XY, ch_bra).t() * hX;
      auto &Xexc = ch_YX <= ch_ket ? X.TwoBody.GetMatrix(ch_YX, ch_ket)
                                   : X.TwoBody.GetMatrix(ch_ket, ch_YX).t() * hX;

      auto &Ydir = ch_XY <= ch_ket ? Y.TwoBody.GetMatrix(ch_XY, ch_ket)
                                   : Y.TwoBody.GetMatrix(ch_ket, ch_XY).t() * hY;
      auto &Yexc = ch_bra <= ch_YX ? Y.TwoBody.GetMatrix(ch_bra, ch_YX)
                                   : Y.TwoBody.GetMatrix(ch_YX, ch_bra).t() * hY;

      //    auto& RHS  =  *vmtx[i];

      arma::mat &Matrixpp = Mpp.GetMatrix(ch_bra, ch_ket);
      arma::mat &Matrixhh = Mhh.GetMatrix(ch_bra, ch_ket);

      const arma::uvec &bras_pp = tbc_XY.GetKetIndex_pp();
      const arma::uvec &bras_hh = tbc_XY.GetKetIndex_hh();
      const arma::uvec &bras_ph = tbc_XY.GetKetIndex_ph();
      const arma::uvec &kets_pp = tbc_YX.GetKetIndex_pp();
      const arma::uvec &kets_hh = tbc_YX.GetKetIndex_hh();
      const arma::uvec &kets_ph = tbc_YX.GetKetIndex_ph();

      // the complicated-looking construct after the % signs just multiply the matrix elements by the proper occupation numbers (nanb, etc.)
      // TODO: Does this whole stacking thing actually improve performance, or just obfuscate the code?
      //      -- Regardless of performance, simpler expressions run into issues with zero-dimensional index vectors. Work for another time.

      arma::mat MLeft = join_horiz(Xdir.cols(bras_hh), -Yexc.cols(kets_hh));
      arma::mat MRight = join_vert(Ydir.rows(bras_hh) % tbc_XY.Ket_occ_hh.cols(arma::uvec(Ydir.n_cols, arma::fill::zeros)),
                                   Xexc.rows(kets_hh) % tbc_YX.Ket_occ_hh.cols(arma::uvec(Xexc.n_cols, arma::fill::zeros)));
      //    arma::mat MRight = join_vert( RHS.rows(bras_hh)  % tbc_bra.Ket_occ_hh.cols( arma::uvec(RHS.n_cols,arma::fill::zeros ) ),
      //                                 LHS2.rows(kets_hh)  % tbc_ket.Ket_occ_hh.cols( arma::uvec(LHS2.n_cols,arma::fill::zeros) ));

      Matrixhh = MLeft * MRight;

      MLeft = join_horiz(Xdir.cols(join_vert(bras_pp, join_vert(bras_hh, bras_ph))), -Yexc.cols(join_vert(kets_pp, join_vert(kets_hh, kets_ph))));

      MRight = join_vert(join_vert(Ydir.rows(bras_pp),
                                   join_vert(Ydir.rows(bras_hh) % tbc_XY.Ket_unocc_hh.cols(arma::uvec(Ydir.n_cols, arma::fill::zeros)),
                                             Ydir.rows(bras_ph) % tbc_XY.Ket_unocc_ph.cols(arma::uvec(Ydir.n_cols, arma::fill::zeros)))),
                         join_vert(Xexc.rows(kets_pp),
                                   join_vert(Xexc.rows(kets_hh) % tbc_YX.Ket_unocc_hh.cols(arma::uvec(Xexc.n_cols, arma::fill::zeros)),
                                             Xexc.rows(kets_ph) % tbc_YX.Ket_unocc_ph.cols(arma::uvec(Xexc.n_cols, arma::fill::zeros)))));

      Matrixpp = MLeft * MRight;

      if (Z.GetParticleRank() > 1)
      {
        Z.TwoBody.GetMatrix(ch_bra, ch_ket) += Matrixpp - Matrixhh;
      }

    } // for itmat

    // The one body part takes some additional work

    //   int norbits = Z.modelspace->GetNumberOrbits();
    int norbits = Z.modelspace->all_orbits.size();
    std::vector<index_t> allorb_vec(Z.modelspace->all_orbits.begin(), Z.modelspace->all_orbits.end());
    //   #pragma omp parallel for schedule(dynamic,1) if (not Z.modelspace->tensor_transform_first_pass[Z.GetJRank()*2+Z.GetParity()])
    #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
    for (int indexi = 0; indexi < norbits; ++indexi)
    //   for (int i=0;i<norbits;++i)
    {
      //      auto i = Z.modelspace->all_orbits[indexi];
      auto i = allorb_vec[indexi];
      Orbit &oi = Z.modelspace->GetOrbit(i);
      double ji = oi.j2 / 2.0;
      //      for (auto j : Z.OneBodyChannels.at({oi.l, oi.j2, oi.tz2}) )
      for (auto j : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
      {
        if (j < i)
          continue;
        Orbit &oj = Z.modelspace->GetOrbit(j);
        double jj = oj.j2 / 2.0;
        int phase_ij = AngMom::phase((oi.j2 - oj.j2) / 2);
        double cijJ = 0;
        // Sum c over holes and include the nbar_a * nbar_b terms
        for (auto &c : Z.modelspace->holes)
        {
          Orbit &oc = Z.modelspace->GetOrbit(c);
          double jc = oc.j2 / 2.0;
          int j1min = std::abs(jc - ji);
          int j1max = jc + ji;
          for (int J1 = j1min; J1 <= j1max; ++J1)
          {
            int j2min = std::max(int(std::abs(jc - jj)), std::abs(Lambda - J1));
            int j2max = std::min(int(jc + jj), J1 + Lambda);
            for (int J2 = j2min; J2 <= j2max; ++J2)
            {
              double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1));
              double sixj = Z.modelspace->GetSixJ(J1, J2, Lambda, jj, ji, jc);
              cijJ += hatfactor * sixj * Z.modelspace->phase(jj + jc + J1 + Lambda) * (oc.occ * Mpp.GetTBME_J(J1, J2, c, i, c, j) + (1 - oc.occ) * Mhh.GetTBME_J(J1, J2, c, i, c, j));
            }
          }
          // Sum c over particles and include the n_a * n_b terms
        }
        for (auto &c : Z.modelspace->particles)
        {
          Orbit &oc = Z.modelspace->GetOrbit(c);
          double jc = oc.j2 / 2.0;
          int j1min = std::abs(jc - ji);
          int j1max = jc + ji;
          for (int J1 = j1min; J1 <= j1max; ++J1)
          {
            int j2min = std::max(int(std::abs(jc - jj)), std::abs(Lambda - J1));
            int j2max = std::min(int(jc + jj), J1 + Lambda);
            for (int J2 = j2min; J2 <= j2max; ++J2)
            {
              double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1));
              double sixj = Z.modelspace->GetSixJ(J1, J2, Lambda, jj, ji, jc);
              cijJ += hatfactor * sixj * Z.modelspace->phase(jj + jc + J1 + Lambda) * Mhh.GetTBME_J(J1, J2, c, i, c, j);  // The phase here is different than in Parzuchowski et al. It appears that the published phase is wrong.
            }
          }
        }
        //         #pragma omp critical
        Z.OneBody(i, j) += cijJ;
        if (i != j)
          Z.OneBody(j, i) += hZ * phase_ij * cijJ;
      } // for j
    }   // for i
    X.profiler.timer["comm222_pp_hh_221st"] += omp_get_wtime() - tstart;
  }


  void comm221st(const Operator &X, const Operator &Y, Operator &Z)
  {
     auto Z2save = Z.TwoBody;
     comm222_pp_hh_221st(X,Y,Z);
     Z.TwoBody = Z2save;
  }
  void comm222_pp_hhst(const Operator &X, const Operator &Y, Operator &Z)
  {
     auto Z1save = Z.OneBody;
     comm222_pp_hh_221st(X,Y,Z);
     Z.OneBody = Z1save;
  }

  //**************************************************************************
  //
  //  X^J_ij`kl` = - sum_J' { i j J } (2J'+1) X^J'_ilkj
  //                        { k l J'}
  // TENSOR VARIETY
  /// The scalar Pandya transformation is defined as
  /// \f[
  ///  \bar{X}^{J}_{i\bar{j}k\bar{l}} = - \sum_{J'} (2J'+1)
  ///  \left\{ \begin{array}{lll}
  ///  j_i  &  j_j  &  J \\
///  j_k  &  j_l  &  J' \\
///  \end{array} \right\}
  ///  X^{J}_{ilkj}
  /// \f]
  /// where the overbar indicates time-reversed orbits.
  /// This function is designed for use with comm222_phss() and so it takes in
  /// two arrays of matrices, one for hp terms and one for ph terms.
  void DoTensorPandyaTransformation(const Operator &Z, std::map<std::array<index_t, 2>, arma::mat> &TwoBody_CC_ph)
  {
    int Lambda = Z.rank_J;
    // loop over cross-coupled channels
    index_t nch = Z.modelspace->SortedTwoBodyChannels_CC.size();

    // Allocate map for matrices -- this needs to be serial.
    for (index_t ch_bra_cc : Z.modelspace->SortedTwoBodyChannels_CC)
    {
      TwoBodyChannel_CC &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
      arma::uvec bras_ph = arma::join_cols(tbc_bra_cc.GetKetIndex_hh(), tbc_bra_cc.GetKetIndex_ph());
      index_t nph_bras = bras_ph.n_rows;
      for (index_t ch_ket_cc : Z.modelspace->SortedTwoBodyChannels_CC)
      {
        TwoBodyChannel_CC &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
        index_t nKets_cc = tbc_ket_cc.GetNumberKets();

        TwoBody_CC_ph[{ch_bra_cc, ch_ket_cc}] = arma::mat(2 * nph_bras, nKets_cc, arma::fill::zeros);
      }
    }

    //   #pragma omp parallel for schedule(dynamic,1) if (not Z.modelspace->tensor_transform_first_pass[Z.GetJRank()*2+Z.GetParity()])
    #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
    for (index_t ich = 0; ich < nch; ++ich)
    {
      index_t ch_bra_cc = Z.modelspace->SortedTwoBodyChannels_CC[ich];
      TwoBodyChannel_CC &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
      int Jbra_cc = tbc_bra_cc.J;
      arma::uvec bras_ph = arma::join_cols(tbc_bra_cc.GetKetIndex_hh(), tbc_bra_cc.GetKetIndex_ph());
      //      arma::uvec& bras_ph = tbc_bra_cc.GetKetIndex_ph();
      index_t nph_bras = bras_ph.size();

      for (index_t ch_ket_cc : Z.modelspace->SortedTwoBodyChannels_CC)
      {
        TwoBodyChannel_CC &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
        int Jket_cc = tbc_ket_cc.J;
        if ((Jbra_cc + Jket_cc < Z.GetJRank()) or std::abs(Jbra_cc - Jket_cc) > Z.GetJRank())
          continue;
        if ((tbc_bra_cc.parity + tbc_ket_cc.parity + Z.GetParity()) % 2 > 0)
          continue;

        index_t nKets_cc = tbc_ket_cc.GetNumberKets();

        //        arma::mat& MatCC_hp = TwoBody_CC_hp[{ch_bra_cc,ch_ket_cc}];
        arma::mat &MatCC_ph = TwoBody_CC_ph[{ch_bra_cc, ch_ket_cc}];
        // loop over ph bras <ad| in this channel
        for (index_t ibra = 0; ibra < nph_bras; ++ibra)
        {
          Ket &bra_cc = tbc_bra_cc.GetKet(bras_ph[ibra]);
          index_t a = bra_cc.p;
          index_t b = bra_cc.q;
          Orbit &oa = Z.modelspace->GetOrbit(a);
          Orbit &ob = Z.modelspace->GetOrbit(b);
          double ja = oa.j2 * 0.5;
          double jb = ob.j2 * 0.5;

          // loop over kets |bc> in this channel
          index_t iket_max = nKets_cc;
          for (index_t iket_cc = 0; iket_cc < iket_max; ++iket_cc)
          {
            Ket &ket_cc = tbc_ket_cc.GetKet(iket_cc % nKets_cc);
            index_t c = iket_cc < nKets_cc ? ket_cc.p : ket_cc.q;
            index_t d = iket_cc < nKets_cc ? ket_cc.q : ket_cc.p;
            Orbit &oc = Z.modelspace->GetOrbit(c);
            Orbit &od = Z.modelspace->GetOrbit(d);
            double jc = oc.j2 * 0.5;
            double jd = od.j2 * 0.5;

            int j1min = std::abs(ja - jd);
            int j1max = ja + jd;
            double sm = 0;
            for (int J1 = j1min; J1 <= j1max; ++J1)
            {
              int j2min = std::max(int(std::abs(jc - jb)), std::abs(J1 - Lambda));
              int j2max = std::min(int(jc + jb), J1 + Lambda);
              for (int J2 = j2min; J2 <= j2max; ++J2)
              {
                //                  double ninej = Z.modelspace->GetNineJ(ja,jd,J1,jb,jc,J2,Jbra_cc,Jket_cc,Lambda);
                double ninej = 0;
                if (Lambda == 0)
                {
                  ninej = AngMom::phase(jb + jd + J1 + Jbra_cc) * Z.modelspace->GetSixJ(ja, jb, Jbra_cc, jc, jd, J1) / sqrt((2 * J2 + 1) * (2 * Jbra_cc + 1));
                }
                else
                {
                  ninej = Z.modelspace->GetNineJ(ja, jd, J1, jb, jc, J2, Jbra_cc, Jket_cc, Lambda);
                }

                if (std::abs(ninej) < 1e-10)
                  continue;
                double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1) * (2 * Jbra_cc + 1) * (2 * Jket_cc + 1));
                double tbme = Z.TwoBody.GetTBME_J(J1, J2, a, d, c, b);
                sm -= hatfactor * Z.modelspace->phase(jb + jd + Jket_cc + J2) * ninej * tbme;
              }
            }
            MatCC_ph(ibra, iket_cc) = sm;

            // Exchange (a <-> b) to account for the (n_a - n_b) term
            // Get Tz,parity and range of J for <bd || ca > coupling
            j1min = std::abs(jb - jd);
            j1max = jb + jd;
            sm = 0;
            for (int J1 = j1min; J1 <= j1max; ++J1)
            {
              int j2min = std::max(int(std::abs(jc - ja)), std::abs(J1 - Lambda));
              int j2max = std::min(int(jc + ja), J1 + Lambda);
              for (int J2 = j2min; J2 <= j2max; ++J2)
              {
                //                  double ninej = Z.modelspace->GetNineJ(jb,jd,J1,ja,jc,J2,Jbra_cc,Jket_cc,Lambda);
                double ninej = 0;
                if (Lambda == 0)
                {
                  ninej = AngMom::phase(ja + jd + J1 + Jbra_cc) * Z.modelspace->GetSixJ(jb, ja, Jbra_cc, jc, jd, J1) / sqrt((2 * J2 + 1) * (2 * Jbra_cc + 1));
                }
                else
                {
                  ninej = Z.modelspace->GetNineJ(jb, jd, J1, ja, jc, J2, Jbra_cc, Jket_cc, Lambda);
                }

                if (std::abs(ninej) < 1e-10)
                  continue;
                double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1) * (2 * Jbra_cc + 1) * (2 * Jket_cc + 1));
                double tbme = Z.TwoBody.GetTBME_J(J1, J2, b, d, c, a);
                sm -= hatfactor * Z.modelspace->phase(ja + jd + Jket_cc + J2) * ninej * tbme;
              }
            }
            MatCC_ph(ibra + nph_bras, iket_cc) = sm;
          }
        }
      }
    }
  }

  // This happens inside an OMP loop, and so everything here needs to be thread safe
  //
  // void Operator::DoTensorPandyaTransformation_SingleChannel( arma::mat& TwoBody_CC_ph, int ch_bra_cc, int ch_ket_cc) const
  // void Operator::DoTensorPandyaTransformation_SingleChannel( arma::mat& MatCC_ph, int ch_bra_cc, int ch_ket_cc) const
  void DoTensorPandyaTransformation_SingleChannel(const Operator &Z, arma::mat &MatCC_ph, int ch_bra_cc, int ch_ket_cc)
  {
    int Lambda = Z.rank_J;
    TwoBodyChannel_CC &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
    arma::uvec bras_ph = arma::join_cols(tbc_bra_cc.GetKetIndex_hh(), tbc_bra_cc.GetKetIndex_ph());
    int nph_bras = bras_ph.n_rows;

    TwoBodyChannel_CC &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
    int nKets_cc = tbc_ket_cc.GetNumberKets();
    // The Pandya-transformed (formerly cross-coupled) particle-hole type matrix elements
    // (this is the output of this method)
    MatCC_ph = arma::mat(2 * nph_bras, nKets_cc, arma::fill::zeros);

    int Jbra_cc = tbc_bra_cc.J;
    int Jket_cc = tbc_ket_cc.J;
    if ((Jbra_cc + Jket_cc < Z.GetJRank()) or std::abs(Jbra_cc - Jket_cc) > Z.GetJRank())
      return;
    if ((tbc_bra_cc.parity + tbc_ket_cc.parity + Z.GetParity()) % 2 > 0)
    {
      return;
    }


    // loop over ph bras <ad| in this channel
    for (int ibra = 0; ibra < nph_bras; ++ibra)
    {
      Ket &bra_cc = tbc_bra_cc.GetKet(bras_ph[ibra]);
      int a = bra_cc.p;
      int b = bra_cc.q;
      Orbit &oa = Z.modelspace->GetOrbit(a);
      Orbit &ob = Z.modelspace->GetOrbit(b);
      double ja = oa.j2 * 0.5;
      double jb = ob.j2 * 0.5;

      // loop over kets |bc> in this channel
      int iket_max = nKets_cc;
      for (int iket_cc = 0; iket_cc < iket_max; ++iket_cc)
      {
        Ket &ket_cc = tbc_ket_cc.GetKet(iket_cc % nKets_cc);
        int c = iket_cc < nKets_cc ? ket_cc.p : ket_cc.q;
        int d = iket_cc < nKets_cc ? ket_cc.q : ket_cc.p;
        Orbit &oc = Z.modelspace->GetOrbit(c);
        Orbit &od = Z.modelspace->GetOrbit(d);
        double jc = oc.j2 * 0.5;
        double jd = od.j2 * 0.5;

        int j1min = std::abs(ja - jd);
        int j1max = ja + jd;
        double sm = 0;
        for (int J1 = j1min; J1 <= j1max; ++J1)
        {
          int j2min = std::max(int(std::abs(jc - jb)), std::abs(J1 - Lambda));
          int j2max = std::min(int(jc + jb), J1 + Lambda);
          for (int J2 = j2min; J2 <= j2max; ++J2)
          {
            //             double ninej = Z.modelspace->GetNineJ(ja,jd,J1,jb,jc,J2,Jbra_cc,Jket_cc,Lambda);
            double ninej = 0;
            if (Lambda == 0)
            {
              ninej = AngMom::phase(jb + jd + J1 + Jbra_cc) * Z.modelspace->GetSixJ(ja, jb, Jbra_cc, jc, jd, J1) / sqrt((2 * J2 + 1) * (2 * Jbra_cc + 1));
            }
            else
            {
              ninej = Z.modelspace->GetNineJ(ja, jd, J1, jb, jc, J2, Jbra_cc, Jket_cc, Lambda);
            }
            if (std::abs(ninej) < 1e-10)
              continue;
            double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1) * (2 * Jbra_cc + 1) * (2 * Jket_cc + 1));
            double tbme = Z.TwoBody.GetTBME_J(J1, J2, a, d, c, b);
            sm -= hatfactor * Z.modelspace->phase(jb + jd + Jket_cc + J2) * ninej * tbme;
          }
        }

        MatCC_ph(ibra, iket_cc) = sm;

        // Exchange (a <-> b) to account for the (n_a - n_b) term
        if (a == b)
        {
          MatCC_ph(ibra + nph_bras, iket_cc) = sm;
        }
        else
        {

          // Get Tz,parity and range of J for <bd || ca > coupling
          j1min = std::abs(jb - jd);
          j1max = jb + jd;
          sm = 0;
          for (int J1 = j1min; J1 <= j1max; ++J1)
          {
            int j2min = std::max(int(std::abs(jc - ja)), std::abs(J1 - Lambda));
            int j2max = std::min(int(jc + ja), J1 + Lambda);
            for (int J2 = j2min; J2 <= j2max; ++J2)
            {
              //               double ninej = Z.modelspace->GetNineJ(jb,jd,J1,ja,jc,J2,Jbra_cc,Jket_cc,Lambda);
              double ninej = 0;
              if (Lambda == 0)
              {
                ninej = AngMom::phase(ja + jd + J1 + Jbra_cc) * Z.modelspace->GetSixJ(jb, ja, Jbra_cc, jc, jd, J1) / sqrt((2 * J2 + 1) * (2 * Jbra_cc + 1));
              }
              else
              {
                ninej = Z.modelspace->GetNineJ(jb, jd, J1, ja, jc, J2, Jbra_cc, Jket_cc, Lambda);
              }

              if (std::abs(ninej) < 1e-10)
                continue;
              double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1) * (2 * Jbra_cc + 1) * (2 * Jket_cc + 1));
              double tbme = Z.TwoBody.GetTBME_J(J1, J2, b, d, c, a);
              sm -= hatfactor * Z.modelspace->phase(ja + jd + Jket_cc + J2) * ninej * tbme;
            }
          }
          MatCC_ph(ibra + nph_bras, iket_cc) = sm;
        }
      }
    }
  }

  void AddInverseTensorPandyaTransformation(Operator &Z, const std::map<std::array<index_t, 2>, arma::mat> &Zbar)
  {
    // Do the inverse Pandya transform
    int Lambda = Z.rank_J;
    //   std::vector<std::map<std::array<int,2>,arma::mat>::iterator> iteratorlist;
    std::vector<std::map<std::array<size_t, 2>, arma::mat>::iterator> iteratorlist;
    //   for (std::map<std::array<int,2>,arma::mat>::iterator iter= Z.TwoBody.MatEl.begin(); iter!= Z.TwoBody.MatEl.end(); ++iter) iteratorlist.push_back(iter);
    for (auto iter = Z.TwoBody.MatEl.begin(); iter != Z.TwoBody.MatEl.end(); ++iter)
      iteratorlist.push_back(iter);
    int niter = iteratorlist.size();
    int hZ = Z.IsHermitian() ? 1 : -1;
    

    // Only go parallel if we've previously calculated the SixJs/NineJs. Otherwise, it's not thread safe.
    //   #pragma omp parallel for schedule(dynamic,1) if (not Z.modelspace->tensor_transform_first_pass[Z.GetJRank()*2+Z.GetParity()])
    #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
    for (int i = 0; i < niter; ++i)
    {
      const auto iter = iteratorlist[i];
      int ch_bra = iter->first[0];
      int ch_ket = iter->first[1];
      arma::mat &Zijkl = iter->second;
      const TwoBodyChannel &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      const TwoBodyChannel &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      index_t nBras = tbc_bra.GetNumberKets();
      index_t nKets = tbc_ket.GetNumberKets();

      for (index_t ibra = 0; ibra < nBras; ++ibra)
      {
        const Ket &bra = tbc_bra.GetKet(ibra);
        int i = bra.p;
        int j = bra.q;
        const Orbit &oi = Z.modelspace->GetOrbit(i);
        const Orbit &oj = Z.modelspace->GetOrbit(j);
        double ji = oi.j2 / 2.;
        double jj = oj.j2 / 2.;
        index_t ketmin = ch_bra == ch_ket ? ibra : 0;
        for (index_t iket = ketmin; iket < nKets; ++iket)
        {
          const Ket &ket = tbc_ket.GetKet(iket);
          int k = ket.p;
          int l = ket.q;
          const Orbit &ok = Z.modelspace->GetOrbit(k);
          const Orbit &ol = Z.modelspace->GetOrbit(l);
          double jk = ok.j2 / 2.;
          double jl = ol.j2 / 2.;

          double commij = 0;
          double commji = 0;

          // Transform Z_ilkj
          int parity_bra_cc = (oi.l + ol.l) % 2;
          int parity_ket_cc = (ok.l + oj.l) % 2;
          //            int Tz_bra_cc = std::abs(oi.tz2+ol.tz2)/2;
          //            int Tz_ket_cc = std::abs(ok.tz2+oj.tz2)/2;
          int Tz_bra_cc = std::abs(oi.tz2 - ol.tz2) / 2;
          int Tz_ket_cc = std::abs(ok.tz2 - oj.tz2) / 2;
          int j3min = std::abs(int(ji - jl));
          int j3max = ji + jl;
          for (int J3 = j3min; J3 <= j3max; ++J3)
          {
            index_t ch_bra_cc = Z.modelspace->GetTwoBodyChannelIndex(J3, parity_bra_cc, Tz_bra_cc);
            const TwoBodyChannel_CC &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
            index_t nbras = tbc_bra_cc.GetNumberKets();
            index_t indx_il = tbc_bra_cc.GetLocalIndex(std::min(i, l), std::max(i, l));
            int j4min = std::max(std::abs(int(jk - jj)), std::abs(J3 - Lambda));
            int j4max = std::min(int(jk + jj), J3 + Lambda);
            for (int J4 = j4min; J4 <= j4max; ++J4)
            {

              index_t ch_ket_cc = Z.modelspace->GetTwoBodyChannelIndex(J4, parity_ket_cc, Tz_ket_cc);
              const TwoBodyChannel_CC &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
              index_t nkets = tbc_ket_cc.GetNumberKets();
              index_t indx_kj = tbc_ket_cc.GetLocalIndex(std::min(j, k), std::max(j, k));

              //                  double ninej = Z.modelspace->GetNineJ(ji,jl,J3,jj,jk,J4,J1,J2,Lambda);
              double ninej = 0;
              double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1) * (2 * J3 + 1) * (2 * J4 + 1));
              double tbme = 0;

              if (Lambda == 0)
              {
                ninej = AngMom::phase(jj + jl + J1 + J3) * Z.modelspace->GetSixJ(ji, jj, J1, jk, jl, J3) / sqrt((2 * J2 + 1) * (2 * J4 + 1));
              }
              else
              {
                ninej = Z.modelspace->GetNineJ(ji, jl, J3, 
                                               jj, jk, J4, 
                                               J1, J2, Lambda);
              }
              if (std::abs(ninej) < 1e-10)
                continue;
              index_t ch_lo = std::min(ch_bra_cc, ch_ket_cc);
              index_t ch_hi = std::max(ch_bra_cc, ch_ket_cc);
              auto zbar_iter = Zbar.find({ch_lo, ch_hi});
              if (zbar_iter == Zbar.end())
                continue;
              const auto &Zmat = zbar_iter->second;

              if (ch_bra_cc <= ch_ket_cc)
              {
                if (i <= l)
                  tbme = Zmat(indx_il, indx_kj + (k > j ? nkets : 0));
                else
                  tbme = Zmat(indx_il, indx_kj + (k > j ? 0 : nkets)) * hZ * Z.modelspace->phase(Lambda + J3 + J4 + int(ji + jj + jk + jl));
              }
              else
              {
                if (k <= j)
                  tbme = Zmat(indx_kj, indx_il + (i > l ? nbras : 0)) * hZ * Z.modelspace->phase(J3 - J4); // Z_ilkj = Z_kjil * (phase)
                else
                  tbme = Zmat(indx_kj, indx_il + (i > l ? 0 : nbras)) * Z.modelspace->phase(Lambda + int(ji + jj + jk + jl)); // Z_ilkj = Z_kjil * (phase)
              }

              commij += hatfactor * Z.modelspace->phase(jj + jl + J2 + J4) * ninej * tbme;
            }// for J4
          }// for J3

          if (i == j)
          {
            commji = commij;
          }
          else
          {
            // Transform Z_jlki

            parity_bra_cc = (oj.l + ol.l) % 2;
            parity_ket_cc = (ok.l + oi.l) % 2;
            Tz_bra_cc = std::abs(oj.tz2 - ol.tz2) / 2;
            Tz_ket_cc = std::abs(ok.tz2 - oi.tz2) / 2;
            j3min = std::abs(int(jj - jl));
            j3max = jj + jl;

            for (int J3 = j3min; J3 <= j3max; ++J3)
            {

              int ch_bra_cc = Z.modelspace->GetTwoBodyChannelIndex(J3, parity_bra_cc, Tz_bra_cc);
              const TwoBodyChannel_CC &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
              int nbras = tbc_bra_cc.GetNumberKets();
              int indx_jl = tbc_bra_cc.GetLocalIndex(std::min(j, l), std::max(j, l));
              int j4min = std::max(std::abs(int(jk - ji)), std::abs(J3 - Lambda));
              int j4max = std::min(int(jk + ji), J3 + Lambda);
              for (int J4 = j4min; J4 <= j4max; ++J4)
              {

                int ch_ket_cc = Z.modelspace->GetTwoBodyChannelIndex(J4, parity_ket_cc, Tz_ket_cc);
                const TwoBodyChannel_CC &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
                int nkets = tbc_ket_cc.GetNumberKets();
                int indx_ki = tbc_ket_cc.GetLocalIndex(std::min(k, i), std::max(k, i));

                //                    double ninej = Z.modelspace->GetNineJ(jj,jl,J3,ji,jk,J4,J1,J2,Lambda);

                double ninej = 0;
                double hatfactor = sqrt((2 * J1 + 1) * (2 * J2 + 1) * (2 * J3 + 1) * (2 * J4 + 1));
                double tbme = 0;

                if (Lambda == 0)
                {
                  ninej = AngMom::phase(ji + jl + J1 + J3) * Z.modelspace->GetSixJ(jj, ji, J1, jk, jl, J3) / sqrt((2 * J2 + 1) * (2 * J4 + 1));
                }
                else
                {
                  ninej = Z.modelspace->GetNineJ(jj, jl, J3, 
                                                 ji, jk, J4, 
                                                 J1, J2, Lambda);
                }

                if (std::abs(ninej) < 1e-10)
                  continue;

                index_t ch_lo = std::min(ch_bra_cc, ch_ket_cc);
                index_t ch_hi = std::max(ch_bra_cc, ch_ket_cc);
                auto zbar_iter = Zbar.find({ch_lo, ch_hi});
                if (zbar_iter == Zbar.end())
                  continue;
                const auto &Zmat = zbar_iter->second;

                if (ch_bra_cc <= ch_ket_cc)
                {
                  if (j <= l)
                    tbme = Zmat(indx_jl, indx_ki + (k > i ? nkets : 0));
                  else
                    tbme = Zmat(indx_jl, indx_ki + (k > i ? 0 : nkets)) * hZ * Z.modelspace->phase(Lambda + J3 + J4 + int(ji + jj + jk + jl));
                }
                else
                {
                  if (k <= i)
                    tbme = Zmat(indx_ki, indx_jl + (j > l ? nbras : 0)) * hZ * Z.modelspace->phase(J3 - J4); // Z_ilkj = Z_kjil * (phase)
                  else
                    tbme = Zmat(indx_ki, indx_jl + (j > l ? 0 : nbras)) * Z.modelspace->phase(Lambda+ int(ji + jj + jk + jl)); // Z_ilkj = Z_kjil * (phase)
                }

                commji += hatfactor * Z.modelspace->phase(ji + jl + J2 + J4) * ninej * tbme;
              }// for J4
            }// for J3
          }

          double norm = bra.delta_pq() == ket.delta_pq() ? 1 + bra.delta_pq() : PhysConst::SQRT2;
          Zijkl(ibra, iket) += (commij - Z.modelspace->phase(ji + jj - J1) * commji) / norm;

          if (ch_bra == ch_ket)
            Zijkl(iket, ibra) = hZ * Zijkl(ibra, iket);
        }// for iket
      }// for ibra
    }// for i in niter (loop over ch_bra,ch_ket)
  }


  ///*************************************
  /// convenience function
  /// called by comm222_phst
  ///*************************************
  std::deque<arma::mat> InitializePandya(Operator &Z, size_t nch, std::string orientation = "normal", int X_parity = 0)
  {
    std::deque<arma::mat> X(nch);
    int n_nonzero = Z.modelspace->SortedTwoBodyChannels_CC.size();
    for (int ich = 0; ich < n_nonzero; ++ich)
    {
      int ch_cc = Z.modelspace->SortedTwoBodyChannels_CC[ich];
      TwoBodyChannel_CC &tbc_cc_ket = Z.modelspace->GetTwoBodyChannel_CC(ch_cc);
      // If the operator X violates parity, we need to construct the matrix for the opposite parity channel as bra
      int parity = (tbc_cc_ket.parity + X_parity)%2;
      int ch_cc_bra = Z.modelspace->GetTwoBodyChannelIndex(tbc_cc_ket.J, parity, tbc_cc_ket.Tz);
      TwoBodyChannel_CC &tbc_cc_bra = Z.modelspace->GetTwoBodyChannel_CC(ch_cc_bra);
      int nKets_cc = tbc_cc_ket.GetNumberKets();
      arma::uvec bras_ph = arma::join_cols(tbc_cc_bra.GetKetIndex_hh(), tbc_cc_bra.GetKetIndex_ph());
      int nph_bras = bras_ph.n_rows;  
      if (nph_bras == 0 and nKets_cc == 0)
        continue;
      if (orientation == "normal")
        X[ch_cc] = arma::mat(2 * nph_bras, nKets_cc, arma::fill::zeros);
      else if (orientation == "transpose")
        X[ch_cc] = arma::mat(nKets_cc, 2 * nph_bras, arma::fill::zeros);
    }
    return X;
  }

  //*****************************************************************************************
  //
  //  THIS IS THE BIG UGLY ONE.
  //
  //   |          |      |          |
  //   |     __Y__|      |     __X__|
  //   |    /\    |      |    /\    |
  //   |   (  )   |  _   |   (  )   |
  //   |____\/    |      |____\/    |
  //   |  X       |      |  Y       |
  //
  //
  // -- This appears to agree with Nathan's results
  //
  /// Calculates the part of \f$ [X_{(2)},\mathbb{Y}^{\Lambda}_{(2)}]_{ijkl} \f$ which involves ph intermediate states, here indicated by \f$ \mathbb{Z}^{J_1J_2\Lambda}_{ijkl} \f$
  /// \f[
  /// \mathbb{Z}^{J_1J_2\Lambda}_{ijkl} = \sum_{abJ_3J_4}(n_a-n_b) \hat{J_1}\hat{J_2}\hat{J_3}\hat{J_4}
  /// \left[
  ///  \left\{ \begin{array}{lll}
  ///  j_i  &  j_l  &  J_3 \\
///  j_j  &  j_k  &  J_4 \\
///  J_1  &  J_2  &  \Lambda \\
///  \end{array} \right\}
  /// \left( \bar{X}^{J3}_{i\bar{l}a\bar{b}}\bar{\mathbb{Y}}^{J_3J_4\Lambda}_{a\bar{b}k\bar{j}} -
  ///   \bar{\mathbb{Y}}^{J_3J_4\Lambda}_{i\bar{l}a\bar{b}}\bar{X}^{J_4}_{a\bar{b}k\bar{j}} \right)
  ///  -(-1)^{j_i+j_j-J_1}
  ///  \left\{ \begin{array}{lll}
  ///  j_j  &  j_l  &  J_3 \\
///  j_i  &  j_k  &  J_4 \\
///  J_1  &  J_2  &  \Lambda \\
///  \end{array} \right\}
  /// \left( \bar{X}^{J_3}_{i\bar{l}a\bar{b}}\bar{\mathbb{Y}}^{J_3J_4\Lambda}_{a\bar{b}k\bar{j}} -
  ///   \bar{\mathbb{Y}}^{J_3J_4\Lambda}_{i\bar{l}a\bar{b}}\bar{X}^{J_4}_{a\bar{b}k\bar{j}} \right)
  /// \right]
  /// \f]
  /// This is implemented by defining an intermediate matrix
  /// \f[
  /// \bar{\mathbb{Z}}^{J_3J_4\Lambda}_{i\bar{l}k\bar{j}} \equiv \sum_{ab}(n_a\bar{n}_b)
  /// \left[ \left( \bar{X}^{J3}_{i\bar{l}a\bar{b}}\bar{\mathbb{Y}}^{J_3J_4\Lambda}_{a\bar{b}k\bar{j}} -
  ///   \bar{\mathbb{Y}}^{J_3J_4\Lambda}_{i\bar{l}a\bar{b}}\bar{X}^{J_4}_{a\bar{b}k\bar{j}} \right)
  /// -\left( \bar{X}^{J_3}_{i\bar{l}b\bar{a}}\bar{\mathbb{Y}}^{J_3J_4\Lambda}_{b\bar{a}k\bar{j}} -
  ///    \bar{\mathbb{Y}}^{J_3J_4\Lambda}_{i\bar{l}b\bar{a}}\bar{X}^{J_4}_{b\bar{a}k\bar{j}} \right)\right]
  /// \f]
  /// The Pandya-transformed matrix elements are obtained with DoTensorPandyaTransformation().
  /// The matrices \f$ \bar{X}^{J_3}_{i\bar{l}a\bar{b}}\bar{\mathbb{Y}}^{J_3J_4\Lambda}_{a\bar{b}k\bar{j}} \f$
  /// and \f$ \bar{\mathbb{Y}}^{J_4J_3\Lambda}_{i\bar{l}a\bar{b}}\bar{X}^{J_3}_{a\bar{b}k\bar{j}} \f$
  /// are related by a Hermitian conjugation, which saves two matrix multiplications, provided we
  /// take into account the phase \f$ (-1)^{J_3-J_4} \f$ from conjugating the spherical tensor.
  /// The commutator is then given by
  /// \f[
  /// \mathbb{Z}^{J_1J_2\Lambda}_{ijkl} = \sum_{J_3J_4} \hat{J_1}\hat{J_2}\hat{J_3}\hat{J_4}
  /// \left[
  ///  \left\{ \begin{array}{lll}
  ///  j_i  &  j_l  &  J_3 \\
///  j_j  &  j_k  &  J_4 \\
///  J_1  &  J_2  &  \Lambda \\
///  \end{array} \right\}
  ///  \bar{\mathbb{Z}}^{J_3J_4\Lambda}_{i\bar{l}k\bar{j}}
  ///  -(-1)^{j_i+j_j-J}
  ///  \left\{ \begin{array}{lll}
  ///  j_j  &  j_l  &  J_3 \\
///  j_i  &  j_k  &  J_4 \\
///  J_1  &  J_2  &  \Lambda \\
///  \end{array} \right\}
  ///  \bar{\mathbb{Z}}^{J_3J_4\Lambda}_{j\bar{l}k\bar{i}}
  ///  \right]
  ///  \f]
  ///
  // void Operator::comm222_phst( Operator& Y, Operator& Z )
  // void Operator::comm222_phst( const Operator& X, const Operator& Y )
  void comm222_phst(const Operator &X, const Operator &Y, Operator &Z)
  {
    int hX = X.IsHermitian() ? 1 : -1;
    int hY = Y.IsHermitian() ? 1 : -1;
    int Lambda = Y.GetJRank();

    double t_start = omp_get_wtime();
    Z.modelspace->PreCalculateSixJ(); // if we already did it, this does nothing.
    Z.modelspace->PreCalculateNineJ(); // if we already did it, this does nothing.
    bool PVPV = false; //Flag to handle the case where both X and Y violate parity
    if (X.GetParity() == 1 and Y.GetParity() == 1)
      PVPV = true;
    // We reuse Xt_bar multiple times, so it makes sense to calculate them once and store them in a deque.
    std::deque<arma::mat> Xt_bar_ph; // We re-use the scalar part multiple times, so there's a significant speed gain for saving it
    Xt_bar_ph = InitializePandya(Z, Z.nChannels, "transpose", X.GetParity());
    DoPandyaTransformation(X, Xt_bar_ph, "transpose");
    X.profiler.timer["_DoTensorPandyaTransformationX"] += omp_get_wtime() - t_start;
    // Construct the intermediate matrix Z_bar
    // First, we initialize the map Z_bar with empty matrices
    // to avoid problems in the parallel loop -- (do we even want a parallel loop here?)
    std::map<std::array<index_t, 2>, arma::mat> Z_bar;

    double t_internal = omp_get_wtime();
    // TODO: I suspect that using pandya_lookup isn't all that beneficial. Check this, and if it's not, we can clean up ModelSpace a bit.
    const auto &pandya_lookup = Z.modelspace->GetPandyaLookup(Z.GetJRank(), Z.GetTRank(), Z.GetParity());
    X.profiler.timer["_PandyaLookup"] += omp_get_wtime() - t_internal;
    t_internal = omp_get_wtime();

    // If X violates parity, we need the intermediate states where the parity is flipped.
    // (Note that I changed DoPandyaTransformation to look for the parity of the operator
    // so that part is already handled, just need to call the right channels for the Pandya
    // transform of Y later on and the phase matrices.)
    // We need them both cases when only X violates parity, i.e.
    // <J+|X|J-><J-|Y|J'->
    // and
    // <J+|Y|J'+><J'+|X|J'->
    // or when both X and Y violate parity, i.e.
    // <J+|X|J-><J-|Y|J'+>
    // and
    // <J+|Y|J'-><J'-|X|J'+>.
    // This means that wee need to sort 2 extra lists of channels.
    // Otherwise, we the codes stays the same as before except
    // that certain channels are changed for the intermediate ones accordingly.
    // A. Belley
    std::vector<index_t> zbras;
    std::vector<index_t> inter_x_states;
    std::vector<index_t> inter_y_states;
    std::vector<index_t> zkets;
    for (auto ich_bra : pandya_lookup)
    {
      auto tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ich_bra);
      int n_rows = tbc_bra_cc.GetNumberKets();
      if (n_rows < 1)
        continue;
      for (auto ich_ket : pandya_lookup)
      {
        if (ich_bra > ich_ket)
          continue;
        auto tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ich_ket);
        int n_cols = 2 * tbc_ket_cc.GetNumberKets();
        if (n_cols < 1)
          continue;
        if ((tbc_bra_cc.parity + tbc_ket_cc.parity + Z.parity) % 2 > 0)
          continue;
        if ((tbc_bra_cc.J + tbc_ket_cc.J < Z.GetJRank()) or (std::abs(tbc_bra_cc.J - tbc_ket_cc.J) > Z.GetJRank()))
          continue;
        // Important to remember. For CC channels, Tz is the magnitude of the difference of the isospin | tz1 - tz2|
        // For RankT=0, we can have <pn|pn>, <pp|nn>, <pp|pp>, <nn|nn>, (Tz_bra,Tz_ket) =>  (1,1) , (0,0)
        // For RankT=1, we can have <pn|pp>, <pn|nn>  (Tz_bra,Tz_ket) => (0,1) , (1,0)
        // For RankT=2, we can have <pn|pn>   (Tz_bra,Tz_ket) => (1,1)
        if (not((tbc_bra_cc.Tz + tbc_ket_cc.Tz == Z.GetTRank()) or (std::abs(tbc_bra_cc.Tz - tbc_ket_cc.Tz) == Z.GetTRank())))
          continue;
    
        
        int i_interx = Z.modelspace->GetTwoBodyChannelIndex(tbc_bra_cc.J, (tbc_bra_cc.parity + X.GetParity()) % 2, tbc_bra_cc.Tz);
        int i_intery = Z.modelspace->GetTwoBodyChannelIndex(tbc_ket_cc.J, (tbc_ket_cc.parity + X.GetParity()) % 2, tbc_ket_cc.Tz); 

        zbras.push_back(ich_bra);
        inter_y_states.push_back(i_intery);
        inter_x_states.push_back(i_interx);
        zkets.push_back(ich_ket);
        Z_bar[{ich_bra, ich_ket}] = arma::mat(n_rows, n_cols);
        //         Z_bar[{ich_bra,ich_ket}] = arma::mat();
      }
    }
    int counter = zbras.size();

    X.profiler.timer["_Allocate Z_bar_tensor"] += omp_get_wtime() - t_internal;

    t_internal = omp_get_wtime();
    // BEGIN OLD WAY
    if (Z.GetJRank() > 0)
    {
          //  std::cout << "  in  " << __func__ << "  doing it the old way. Counter = " << counter << std::endl;

      #ifndef OPENBLAS_NOUSEOMP
                //  #pragma omp parallel for schedule(dynamic,1) if (not Z.modelspace->tensor_transform_first_pass[Z.GetJRank()*2+Z.GetParity()])
      #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
      #endif

      for (int i = 0; i < counter; ++i)
      {
        index_t ch_bra_cc = zbras[i];
        index_t ch_ix_cc = inter_x_states[i];
        index_t ch_iy_cc = inter_y_states[i];
        index_t ch_ket_cc = zkets[i];

        const auto &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
        const auto &tbc_ix_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ix_cc);
        const auto &tbc_iy_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_iy_cc);
        const auto &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);
        int Jbra = tbc_bra_cc.J;
        int Jket = tbc_ket_cc.J;

        arma::mat YJ1J2;
        arma::mat YJ2J1;
        const auto &XJ1 = Xt_bar_ph[ch_bra_cc];
        const auto &XJ2 = Xt_bar_ph[ch_ket_cc];

        arma::uvec kets_ph = arma::join_cols(tbc_iy_cc.GetKetIndex_hh(), tbc_iy_cc.GetKetIndex_ph());
        arma::uvec bras_ph = arma::join_cols(tbc_ix_cc.GetKetIndex_hh(), tbc_ix_cc.GetKetIndex_ph());
        DoTensorPandyaTransformation_SingleChannel(Y, YJ1J2, ch_ix_cc, ch_ket_cc);
        if (ch_bra_cc == ch_ket_cc and ch_ix_cc == ch_iy_cc)
        {
          YJ2J1 = YJ1J2;
        }
        else
        {
          DoTensorPandyaTransformation_SingleChannel(Y, YJ2J1, ch_iy_cc, ch_bra_cc);
        }
        int flipphaseY = hY * Z.modelspace->phase(Jbra - Jket);
        // construct a matrix of phases (-1)^{k+j+p+h} used below to generate X_phkj for k>j
        arma::mat PhaseMatXJ2(tbc_ket_cc.GetNumberKets(), kets_ph.size(), arma::fill::ones);
        arma::mat PhaseMatYJ1J2(bras_ph.size(), tbc_ket_cc.GetNumberKets(), arma::fill::ones);
        for (index_t iket = 0; iket < (index_t)tbc_ket_cc.GetNumberKets(); iket++)
        {
          const Ket &ket = tbc_ket_cc.GetKet(iket);
          if (Z.modelspace->phase((ket.op->j2 + ket.oq->j2) / 2) < 0)
          {
            PhaseMatXJ2.row(iket) *= -1;
            PhaseMatYJ1J2.col(iket) *= -1;
          }
        }
        for (index_t iph = 0; iph < kets_ph.size(); iph++)
        {
          const Ket &ket_ph = tbc_iy_cc.GetKet(kets_ph[iph]);
          if (Z.modelspace->phase((ket_ph.op->j2 + ket_ph.oq->j2) / 2) < 0)
            PhaseMatXJ2.col(iph) *= -1;
        }
        for (index_t iph = 0; iph < bras_ph.size(); iph++)
        {
          const Ket &bra_ph = tbc_ix_cc.GetKet(bras_ph[iph]);
          if (Z.modelspace->phase((bra_ph.op->j2 + bra_ph.oq->j2) / 2) < 0)
            PhaseMatYJ1J2.row(iph) *= -1;
        }
        PhaseMatXJ2 *= -1 * hX;
        PhaseMatYJ1J2 *= hY * Z.modelspace->phase(Jbra + Jket + Lambda);
        

        //                J2                       J1         J2                       J2          J2
        //             k<=j     k>=j                hp  -ph    hp   ph                 k<=j       k<=j
        //      J1   [       |       ]       J1   [           |          ]         [hp        |ph        ]
        //     i<=j  [  Zbar | Zbar  ]  =   i<=j  [   Xbar    | -Ybar    ]   * J1  [   Ybar   |   Ybar'  ]
        //           [       |       ]            [           |          ]         [ph        |hp        ]      where Ybar'_phkj = Ybar_hpkj * (-1)^{p+h+k+j}*(-1)^{Lambda + J1 + J2}*hY
        //                                                                         [----------|----------]       and
        //                                                                     J2  [hp        |ph        ]            Xbar'_phkj = Xbar_hpkj * (-1)^{p+h+k+j}*hX
        //                                                                         [   Xbar   |   Xbar'  ]
        //                                                                         [-ph       |-hp       ]
        //
        //
        int halfncx2 = XJ2.n_cols / 2;
        int halfnry12 = YJ1J2.n_rows / 2;
       
        arma::mat Mleft = join_horiz(XJ1, -flipphaseY * YJ2J1.t());
        arma::mat Mright = join_vert(join_horiz(YJ1J2, join_vert(YJ1J2.tail_rows(halfnry12) % PhaseMatYJ1J2,
                                                                 YJ1J2.head_rows(halfnry12) % PhaseMatYJ1J2)),
                                     hX * join_vert(XJ2, join_horiz(XJ2.tail_cols(halfncx2) % PhaseMatXJ2,
                                                                    XJ2.head_cols(halfncx2) % PhaseMatXJ2))
                                              .t());

        auto &Zmat = Z_bar.at({ch_bra_cc, ch_ket_cc});

        Zmat = Mleft * Mright;
        //         std::cout << "   ... line " << __LINE__ << "  " << ch_bra_cc << " " << ch_ket_cc << "  |Z| = " << arma::norm(Zmat,"fro") << std::endl;
      }
    }
    else // faster, more memory hungry way
    {
          //  std::cout << "  in  " << __func__ << "  doing it the new way" << std::endl;
      std::deque<arma::mat> YJ1J2_list(counter);
      std::deque<arma::mat> YJ2J1_list(counter);

      #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
      for (int i = 0; i < counter; ++i)
      {
        index_t ch_bra_cc = zbras[i];
        index_t ch_ix_cc = inter_x_states[i];
        index_t ch_iy_cc = inter_y_states[i];
        index_t ch_ket_cc = zkets[i];
        
        arma::mat &YJ1J2 = YJ1J2_list[i];
        arma::mat &YJ2J1 = YJ2J1_list[i];


        DoTensorPandyaTransformation_SingleChannel(Y, YJ1J2, ch_ix_cc, ch_ket_cc);
        if (ch_bra_cc != ch_ket_cc or ch_ix_cc != ch_iy_cc)
        {
          DoTensorPandyaTransformation_SingleChannel(Y, YJ2J1, ch_iy_cc, ch_bra_cc);
        }
        
      }

      X.profiler.timer["_DoTensorPandyaTransformationY"] += omp_get_wtime() - t_internal;

      t_internal = omp_get_wtime();

      #ifndef OPENBLAS_NOUSEOMP
      #pragma omp parallel for schedule(dynamic, 1) if (not single_thread)
      #endif
      for (int i = 0; i < counter; ++i)
      {

        index_t ch_bra_cc = zbras[i];
        index_t ch_ix_cc = inter_x_states[i];
        index_t ch_iy_cc = inter_y_states[i];
        index_t ch_ket_cc = zkets[i];

        const auto &tbc_bra_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_bra_cc);
        const auto &tbc_ix_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ix_cc);
        const auto &tbc_iy_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_iy_cc);
        const auto &tbc_ket_cc = Z.modelspace->GetTwoBodyChannel_CC(ch_ket_cc);

    
        int Jbra = tbc_bra_cc.J;
        int Jket = tbc_ket_cc.J;
        
        arma::mat &YJ1J2 = YJ1J2_list[i];
        arma::mat &YJ2J1 = (ch_ket_cc == ch_bra_cc and ch_ix_cc == ch_iy_cc) ? YJ1J2_list[i] : YJ2J1_list[i];
        

        const auto &XJ1 = Xt_bar_ph[ch_bra_cc];
        const auto &XJ2 = Xt_bar_ph[ch_ket_cc];


        arma::uvec kets_ph = arma::join_cols(tbc_iy_cc.GetKetIndex_hh(), tbc_iy_cc.GetKetIndex_ph());
        arma::uvec bras_ph = arma::join_cols(tbc_ix_cc.GetKetIndex_hh(), tbc_ix_cc.GetKetIndex_ph());

        
        int size_XJ2 = kets_ph.size();
        int size_YJ1J2 = bras_ph.size();

        
        // construct a matrix of phases (-1)^{k+j+p+h} used below to generate X_phkj for k>j
        arma::mat PhaseMatXJ2(tbc_ket_cc.GetNumberKets(), size_XJ2, arma::fill::ones);
        arma::mat PhaseMatYJ1J2(size_YJ1J2, tbc_ket_cc.GetNumberKets(), arma::fill::ones);
        int flipphaseY = hY * Z.modelspace->phase(Jbra - Jket);

        for (index_t iket = 0; iket < (index_t)tbc_ket_cc.GetNumberKets(); iket++)
        {
          const Ket &ket = tbc_ket_cc.GetKet(iket);
          if (Z.modelspace->phase((ket.op->j2 + ket.oq->j2) / 2) < 0)
          {
            PhaseMatXJ2.row(iket) *= -1;
            PhaseMatYJ1J2.col(iket) *= -1;
          }
        }
        for (index_t iph = 0; iph < size_XJ2; iph++)
        {
          const Ket &ket_ph = tbc_iy_cc.GetKet(kets_ph[iph]);
          if (Z.modelspace->phase((ket_ph.op->j2 + ket_ph.oq->j2) / 2) < 0)
            PhaseMatXJ2.col(iph) *= -1;
        }
        for (index_t iph = 0; iph < size_YJ1J2; iph++)
        {
          const Ket &bra_ph = tbc_ix_cc.GetKet(bras_ph[iph]);
          if (Z.modelspace->phase((bra_ph.op->j2 + bra_ph.oq->j2) / 2) < 0)
            PhaseMatYJ1J2.row(iph) *= -1;
        }
        PhaseMatXJ2 *= -1*hX;
        PhaseMatYJ1J2 *= flipphaseY;
        //                J2                       J1         J2                       J2          J2
        //             k<=j     k>=j                hp  -ph    hp   ph                 k<=j       k<=j
        //      J1   [       |       ]       J1   [           |          ]         [hp        |ph        ]
        //     i<=j  [  Zbar | Zbar  ]  =   i<=j  [   Xbar    | -Ybar    ]   * J1  [   Ybar   |   Ybar'  ]
        //           [       |       ]            [           |          ]         [ph        |hp        ]      where Ybar'_phkj = Ybar_hpkj * (-1)^{p+h+k+j}*(-1)^{J1-J2}*hY
        //                                                                         [----------|----------]       and
        //                                                                     J2  [hp        |ph        ]            Xbar'_phkj = Xbar_hpkj * (-1)^{p+h+k+j}*hX
        //                                                                         [   Xbar   |   Xbar'  ]
        //                                                                         [-ph       |-hp       ]
        //
        //
        int halfncx2 = XJ2.n_cols / 2;
        int halfnry12 = YJ1J2.n_rows / 2;

        arma::mat Mleft = join_horiz(XJ1, -flipphaseY * YJ2J1.t());
        // std::cout<<YJ1J2<<std::endl;
        // std::cout<<XJ2<<std::endl;
        arma::mat Mright = join_vert(join_horiz(YJ1J2, join_vert(YJ1J2.tail_rows(halfnry12) % PhaseMatYJ1J2,
                                                                 YJ1J2.head_rows(halfnry12) % PhaseMatYJ1J2)),
                                     hX * join_vert(XJ2, join_horiz(XJ2.tail_cols(halfncx2) % PhaseMatXJ2,
                                                                    XJ2.head_cols(halfncx2) % PhaseMatXJ2))
                                              .t());

        Z_bar.at({ch_bra_cc, ch_ket_cc}) = Mleft * Mright;
      }// for i looping over channels

    } // else J=0
    X.profiler.timer["_Build Z_bar_tensor"] += omp_get_wtime() - t_internal;



    t_internal = omp_get_wtime();
    AddInverseTensorPandyaTransformation(Z, Z_bar);
    X.profiler.timer[__func__] += omp_get_wtime() - t_start;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// scalar-tensor commutators with 3-body
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Zij^\lamda = 1/12 sum_abcde sum_J1J2J (na nb n¯c n¯d n¯e + ¯na n¯b nc nd ne) (2J+1)/(2ji+1)
  ///                             * (-)^(j + lamda + J1 + j0)
  ///                             * (X^J1,j0,J2,j0;0_abicde Y^J2,j0,J1,j1;lamda_cdeabj
  ///                             -  Y^J1,j0,J2,j1;lamda_abicde  X^J2,j1,J1,j0;0_cdeabj )
  ///
  ///
  void comm331st(const Operator &X, const Operator &Y, Operator &Z)
  {
    Z.modelspace->PreCalculateSixJ();
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z1 = Z.OneBody;

    double tstart = omp_get_wtime();
    int Lambda = Z.GetJRank();
    int hZ = Z.IsHermitian() ? +1 : -1;

    size_t norb = Z.modelspace->GetNumberOrbits();
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < norb; i++)
    {
      Orbit &oi = Z.modelspace->GetOrbit(i);
      for (size_t j : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
      {
        if (j < i)
          continue;

        Orbit &oj = Z.modelspace->GetOrbit(j);
        if (std::abs(oj.j2 - oi.j2) > Lambda * 2 or (oi.j2 + oj.j2) < Lambda * 2)
          continue;

        double zij = 0;
        for (size_t a : Z.modelspace->all_orbits)
        {
          Orbit &oa = Z.modelspace->GetOrbit(a);
          for (size_t b : Z.modelspace->all_orbits)
          {
            Orbit &ob = Z.modelspace->GetOrbit(b);

            for (size_t c : Z.modelspace->all_orbits)
            {
              Orbit &oc = Z.modelspace->GetOrbit(c);
              for (size_t d : Z.modelspace->all_orbits)
              {
                Orbit &od = Z.modelspace->GetOrbit(d);

                for (size_t e : Z.modelspace->all_orbits)
                {
                  Orbit &oe = Z.modelspace->GetOrbit(e);

                  // Tensor operator may change parity. So orbit i and j may not have the same parity
                  if (
                      ((oa.l + ob.l + oi.l + oc.l + od.l + oe.l + X.GetParity()) % 2 != 0) and ((oa.l + ob.l + oj.l + oc.l + od.l + oe.l + Y.GetParity()) % 2 != 0) and
                      ((oa.l + ob.l + oi.l + oc.l + od.l + oe.l + Y.GetParity()) % 2 != 0) and ((oa.l + ob.l + oj.l + oc.l + od.l + oe.l + X.GetParity()) % 2 != 0))
                    continue;

                  if (
                      (std::abs(oa.tz2 + ob.tz2 + oi.tz2 - oc.tz2 - od.tz2 - oe.tz2) != 2 * X.GetTRank()) and (std::abs(oa.tz2 + ob.tz2 + oj.tz2 - oc.tz2 - od.tz2 - oe.tz2) != 2 * Y.GetTRank()) and
                      (std::abs(oa.tz2 + ob.tz2 + oi.tz2 - oc.tz2 - od.tz2 - oe.tz2) != 2 * Y.GetTRank()) and (std::abs(oa.tz2 + ob.tz2 + oj.tz2 - oc.tz2 - od.tz2 - oe.tz2) != 2 * X.GetTRank()))
                    continue;

                  double occupation_factor = oa.occ * ob.occ * (1 - oc.occ) * (1 - od.occ) * (1 - oe.occ) // fixed mistake found by Matthias Heinz Oct 2022
                                             + (1 - oa.occ) * (1 - ob.occ) * oc.occ * od.occ * oe.occ;
                  // occupation_factor = 1.;
                  if (std::abs(occupation_factor) < 1e-7)
                    continue;

                  int J1min = std::abs(oa.j2 - ob.j2) / 2;
                  int J1max = (oa.j2 + ob.j2) / 2;

                  int J2min = std::abs(oc.j2 - od.j2) / 2;
                  int J2max = (oc.j2 + od.j2) / 2;

                  for (int J1 = J1min; J1 <= J1max; J1++)
                  {
                    if (a == b and J1 % 2 > 0)
                      continue;

                    for (int J2 = J2min; J2 <= J2max; J2++)
                    {
                      if (c == d and J2 % 2 > 0)
                        continue;

                      int j0min = std::max(std::abs(2 * J1 - oi.j2), std::abs(2 * J2 - oe.j2));
                      int j0max = std::min(2 * J1 + oi.j2, 2 * J2 + oe.j2);

                      int j1min = std::abs(2 * J1 - oj.j2);
                      int j1max = 2 * J1 + oj.j2;

                      for (int j0 = j0min; j0 <= j0max; j0 += 2)
                      {
                        for (int j1 = j1min; j1 <= j1max; j1 += 2)
                        {
                          if (std::abs(j0 - j1) > Lambda * 2 or (j0 + j1) < Lambda * 2)
                            continue;
                          //double sixj1 = AngMom::SixJ(oj.j2 / 2., oi.j2 / 2., Lambda, j0 / 2., j1 / 2., J1);
                          double sixj1 = AngMom::SixJ(oj.j2 / 2., oi.j2 / 2., Lambda, j0 / 2., j1 / 2., J1);

                          if (std::abs(sixj1) < 1.e-6)
                            continue;
                          double xabicde = X3.GetME_pn(J1, J2, j0, a, b, i, c, d, e);     // scalar
                          double ycdeabj = Y3.GetME_pn(J2, j0, J1, j1, c, d, e, a, b, j); // tensor

                          int phase = AngMom::phase((oj.j2 + j0) / 2 + J1 + Lambda);
                          zij += 1. / 12 * phase * occupation_factor * sqrt(j0 + 1) * sqrt(j1 + 1) * sixj1 * (xabicde * ycdeabj); //
                        } // j1
                      } // j0

                      j1min = std::max(std::abs(2 * J1 - oj.j2), std::abs(2 * J2 - oe.j2));
                      j1max = std::min(2 * J1 + oj.j2, 2 * J2 + oe.j2);

                      j0min = std::abs(2 * J1 - oi.j2);
                      j0max = 2 * J1 + oi.j2;

                      for (int j0 = j0min; j0 <= j0max; j0 += 2)
                      {
                        for (int j1 = j1min; j1 <= j1max; j1 += 2)
                        {
                          if (std::abs(j0 - j1) > Lambda * 2 or (j0 + j1) < Lambda * 2)
                            continue;

                          //double sixj1 = AngMom::SixJ(oj.j2 / 2., oi.j2 / 2., Lambda, j0 / 2., j1 / 2., J1);
                          double sixj1 = AngMom::SixJ(oj.j2 / 2., oi.j2 / 2., Lambda, j0 / 2., j1 / 2., J1);
                          if (std::abs(sixj1) < 1.e-6)
                            continue;

                          double yabicde = Y3.GetME_pn(J1, j0, J2, j1, a, b, i, c, d, e); // tensor
                          double xcdeabj = X3.GetME_pn(J2, J1, j1, c, d, e, a, b, j);     // sclar

                          int phase = AngMom::phase((oj.j2 + j0) / 2 + J1 + Lambda);
                          zij -= 1. / 12 * phase * occupation_factor * sqrt(j0 + 1) * sqrt(j1 + 1) * sixj1 * (yabicde * xcdeabj); //
                        } // j1
                      } // j0

                    } // J2
                  } // J1
                } // e
              } // d
            } // c
          } // b
        } // a
        Z1(i, j) += zij;
        if ( i != j)
          Z1(j, i) += AngMom::phase((oi.j2 - oj.j2)/2 ) * hZ * zij;
      } // j
    } // i

    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm331st

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:   Z^(J1j1,J2j2)Lamda_ijklmn = sum_{a, J3} PJ1j1(ij/k) PJ1j2(lm/n) sqrt( (2J1+1) (2j1+1)) (2j2+1) (2J3+1)) 
  ///                                           (  (-)^j2+jn+Lamda+ J3 + 1
  ///                                           { jk ja J3 } { J3 J2 Lamda }  XJ1_ijna YJ3J2_Lamda_kalm
  ///                                           { jn j1 J1 } { j2 j1 jn    }
  ///                                            - (-)^j2+jk+Lamda+ J1
  ///                                           { jn ja J3 } { J3 J1 Lamda }  YJ1J3_Lamda_ijna XJ2_kalm
  ///                                           { jk j2 J2 } { j1 j2 jk    }
  ///
  void comm223st(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X2 = X.TwoBody;
    auto &Y2 = Y.TwoBody;
    auto &Z3 = Z.ThreeBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();
    // Permutations of indices which are needed to produce antisymmetrized matrix elements  P(ij/k) |ijk> = |ijk> - |kji> - |ikj>
    const std::array<ThreeBodyStorage::Permutation, 3> index_perms = {ThreeBodyStorage::ABC, ThreeBodyStorage::CBA, ThreeBodyStorage::ACB};

    double tstart = omp_get_wtime();

    std::vector<std::array<size_t, 3>> bra_ket_channels;
    for (auto &it : Z.ThreeBody.Get_ch_start())
    {
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(it.first.ch_bra);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      for (size_t ibra = 0; ibra < nbras3; ibra++)
      {
        bra_ket_channels.push_back({it.first.ch_bra, it.first.ch_ket, static_cast<size_t>(ibra)}); // (ch_bra, ch_ket,ibra)
      }
    }
    size_t n_bra_ket_ch = bra_ket_channels.size();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t ibra_ket = 0; ibra_ket < n_bra_ket_ch; ibra_ket++)
    {
      size_t ch3bra = bra_ket_channels[ibra_ket][0];
      size_t ch3ket = bra_ket_channels[ibra_ket][1];
      size_t ibra = bra_ket_channels[ibra_ket][2];
      auto &Tbc_bra = Z.modelspace->GetThreeBodyChannel(ch3bra);
      auto &Tbc_ket = Z.modelspace->GetThreeBodyChannel(ch3ket);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      size_t nkets3 = Tbc_ket.GetNumberKets();


      // int twoJ = Tbc_bra.twoJ; // Scalar commutator so J is the same in bra and ket channel
      int twoj1 = Tbc_bra.twoJ; 
      int twoj2 = Tbc_ket.twoJ; 

      auto &bra = Tbc_bra.GetKet(ibra);
      size_t i = bra.p;
      size_t j = bra.q;
      size_t k = bra.r;
      Orbit &oi = Z.modelspace->GetOrbit(i);
      Orbit &oj = Z.modelspace->GetOrbit(j);
      Orbit &ok = Z.modelspace->GetOrbit(k);
      double ji = 0.5 * oi.j2;
      double jj = 0.5 * oj.j2;
      double jk = 0.5 * ok.j2;

      int J1 = bra.Jpq;

      size_t iket_max = nkets3 - 1;
      if (ch3bra == ch3ket)
        iket_max = ibra;
      for (size_t iket = 0; iket <= iket_max; iket++)
      {
        auto &ket = Tbc_ket.GetKet(iket);
        size_t l = ket.p;
        size_t m = ket.q;
        size_t n = ket.r;
        Orbit &ol = Z.modelspace->GetOrbit(l);
        Orbit &om = Z.modelspace->GetOrbit(m);
        Orbit &on = Z.modelspace->GetOrbit(n);
        double jl = 0.5 * ol.j2;
        double jm = 0.5 * om.j2;
        double jn = 0.5 * on.j2;
        int J2 = ket.Jpq;

        double zijklmn = 0;
        /// BEGIN THE SLOW BIT...

        // Now we need to loop over the permutations in ijk and then lmn
        for (auto perm_ijk : index_perms)
        {
          size_t I1, I2, I3;
          Z3.Permute(perm_ijk, i, j, k, I1, I2, I3);
          Orbit &o1 = Z.modelspace->GetOrbit(I1);
          Orbit &o2 = Z.modelspace->GetOrbit(I2);
          Orbit &o3 = Z.modelspace->GetOrbit(I3);

          int J1p_min = J1;
          int J1p_max = J1;
          if (perm_ijk != ThreeBodyStorage::ABC)
          {
            J1p_min = std::max(std::abs(o1.j2 - o2.j2), std::abs(twoj1 - o3.j2)) / 2;
            J1p_max = std::min(o1.j2 + o2.j2, twoj1 + o3.j2) / 2;
          }

          int parity_12 = (o1.l + o2.l) % 2;
          int Tz_12 = (o1.tz2 + o2.tz2) / 2;

          double j3 = 0.5 * o3.j2;

          for (int J1p = J1p_min; J1p <= J1p_max; J1p++)
          {

            double rec_ijk = Z3.RecouplingCoefficient(perm_ijk, ji, jj, jk, J1p, J1, twoj1);
            rec_ijk *= Z3.PermutationPhase(perm_ijk); // do we get a fermionic minus sign?

            for (auto perm_lmn : index_perms)
            {
              size_t I4, I5, I6;
              Z3.Permute(perm_lmn, l, m, n, I4, I5, I6);
              Orbit &o4 = Z.modelspace->GetOrbit(I4);
              Orbit &o5 = Z.modelspace->GetOrbit(I5);
              Orbit &o6 = Z.modelspace->GetOrbit(I6);

              double j6 = 0.5 * o6.j2;

              int J2p_min = J2;
              int J2p_max = J2;
              if (perm_lmn != ThreeBodyStorage::ABC)
              {
                J2p_min = std::max(std::abs(o4.j2 - o5.j2), std::abs(twoj2 - o6.j2)) / 2;
                J2p_max = std::min(o4.j2 + o5.j2, twoj2 + o6.j2) / 2;
              }

              for (size_t a : Z.modelspace->all_orbits)
              {
                Orbit &oa = Z.modelspace->GetOrbit(a);
                int j2a = oa.j2;

                int dTz_126a = (o1.tz2 + o2.tz2 - o6.tz2 - oa.tz2) / 2;
                int dTz_3a45 = (o3.tz2 + oa.tz2 - o4.tz2 - o5.tz2) / 2;
                if (std::abs(dTz_126a) != X.GetTRank() and std::abs(dTz_126a) != Y.GetTRank())
                  continue;
                if (std::abs(dTz_3a45) != X.GetTRank() and std::abs(dTz_3a45) != Y.GetTRank())
                  continue;

                for (int J2p = J2p_min; J2p <= J2p_max; J2p++)
                {
                  double rec_lmn = Z3.RecouplingCoefficient(perm_lmn, jl, jm, jn, J2p, J2, twoj2);
                  rec_lmn *= Z3.PermutationPhase(perm_lmn); // do we get a fermionic minus sign?

                  // direct term
                  int J3_min = std::abs(o3.j2 - j2a) / 2;
                  int J3_max = ( o3.j2 + j2a ) / 2;
                  for (int J3 = J3_min ; J3 <= J3_max; J3++)
                  {
                    if ( J3 +  J2p < Lambda or  std::abs(J3 - J2p) > Lambda )
                      continue;
                    double sixj_1, sixj_2;
                    sixj_1 = AngMom::SixJ(o3.j2 / 2., j2a / 2.,      J3,
                                          o6.j2 / 2., twoj1 / 2.,    J1p);
                    sixj_2 = AngMom::SixJ(J3,         J2p,           Lambda,
                                          twoj2 / 2., twoj1 / 2.,    o6.j2 / 2.);  
                    int phase = AngMom::phase((o6.j2 + twoj2) / 2 + J3 + Lambda);
                    double facotrs = sqrt((2 * J1p + 1) * (2 * J3 + 1) * (twoj1 + 1) * (twoj2 + 1)); 
                    double x_126a = X2.GetTBME_J(J1p, J1p, I1, I2, I6, a);
                    double y_3a45 = Y2.GetTBME_J(J3, J2p, I3, a, I4, I5);
                    zijklmn += rec_ijk * rec_lmn * sixj_1 * sixj_2 * phase * facotrs * (x_126a * y_3a45); 
                  } // J3

                  // exchange term
                  J3_min = std::abs(o6.j2 - j2a) / 2;
                  J3_max = ( o6.j2 + j2a ) / 2;
                  for (int J3 = J3_min ; J3 <= J3_max; J3++)
                  {
                    if ( J3 +  J1p < Lambda or  std::abs(J3 - J1p) > Lambda )
                      continue;
                    double sixj_1, sixj_2;
                    sixj_1 = AngMom::SixJ(o6.j2 / 2., j2a / 2.,      J3,
                                          o3.j2 / 2., twoj2 / 2.,    J2p);
                    sixj_2 = AngMom::SixJ(J3,         J1p,           Lambda,
                                          twoj1 / 2., twoj2 / 2.,    o3.j2 / 2.);

                    int phase = AngMom::phase((o3.j2 + twoj2) / 2 + J1p + Lambda);
                    double facotrs = sqrt((2 * J2p + 1) * (2 * J3 + 1) * (twoj1 + 1) * (twoj2 + 1)); 

                    double y_126a = Y2.GetTBME_J(J1p, J3, I1, I2, I6, a);
                    double x_3a45 = X2.GetTBME_J(J2p, J2p, I3, a, I4, I5);
                    zijklmn -= rec_ijk * rec_lmn * sixj_1 * sixj_2 * phase * facotrs * ( y_126a * x_3a45 );
                  } // J3
                } // for J2p
              } // for orbit a 
            } // for perm_lmn
          } // for J1p
        } // for perm_ijk

        Z3.AddToME_pn_ch(ch3bra, ch3ket, ibra, iket, zijklmn); // this needs to be modified for beta decay
      } // for iket
    } // for ch3

    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm233st

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Zij^Lamda = 1/4 sum_abcd sum_J1j1j2 na nb (1-nc) (1-nd) sqrt((2j1+1)(2j2+1)) 
  ///                            (-)^(j_j + Lamda + J1 + j1) { jj  ji  Lamda }     
  ///                                                        { j1  j2  J1    } 
  ///                            ( X^J1_abcd Y^(J1j1, J1j2)Lamda_cdiabj - Y^(J1j1, J1j2)Lamda_abicdj XJ1_cdab )
  ///
  ///                            + 1/4 sum_abcd sum_J1J2j1 na nb (1-nc) (1-nd) (2j1+1) 
  ///                            [(-)^(j_i + J1 + j1) { Lamda  J2  J1  }  X(J1j1,J2j1)0_abicdj Y(J1,J2)Lamda_cdab   
  ///                                                 { j1     ji  jj  } 
  ///                            -(-)^(j_i + J2 + j1) { Lamda  J2  J1  }  Y(J1,J2)Lamda_abcd X(J2j1,J1j1)0_cdiabj ]
  ///                                                 { j1     jj  ji  } 
  ///
  void comm231st(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X2 = X.TwoBody;
    auto &Y2 = Y.TwoBody;
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z1 = Z.OneBody;
    int Lambda = Z.GetJRank();
    int hZ = Z.IsHermitian() ? +1 : -1;
    Z.modelspace->PreCalculateSixJ();
    size_t norb = Z.modelspace->GetNumberOrbits();

    double tstart = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic,1)
    for (size_t i = 0; i < norb; i++)
    {
      Orbit &oi = Z.modelspace->GetOrbit(i);
      for (size_t j : Z.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
      {
        Orbit &oj = Z.modelspace->GetOrbit(j);
        double zij = 0;
        if (j < i)
          continue;

        for (size_t a : Z.modelspace->all_orbits)
        {
          Orbit &oa = Z.modelspace->GetOrbit(a);
          for (size_t b : Z.modelspace->all_orbits)
          {
            Orbit &ob = Z.modelspace->GetOrbit(b);
            for (size_t c : Z.modelspace->all_orbits)
            {
              Orbit &oc = Z.modelspace->GetOrbit(c);
              for (size_t d : Z.modelspace->all_orbits)
              {
                Orbit &od = Z.modelspace->GetOrbit(d);

                double occ_factor =  oa.occ * ob.occ * (1 - oc.occ) * (1 - od.occ);
                if (std::abs(occ_factor) < 1e-8)
                  continue;

                int J1min = std::max(std::abs(oa.j2 - ob.j2), std::abs(oc.j2 - od.j2)) / 2;
                int J1max = std::min(oa.j2 + ob.j2, oc.j2 + od.j2) / 2;
                for (int J1 = J1min; J1 <= J1max; J1++)
                {
                  if (a == b and J1 % 2 > 0)
                    continue;
                  if (c == d and J1 % 2 > 0)
                    continue;

                  double xabcd = X2.GetTBME_J(J1, J1, a, b, c, d);
                  double xcdab = X2.GetTBME_J(J1, J1, c, d, a, b);

                  int twoj1min = std::abs(2 * J1 - oi.j2);
                  int twoj1max = 2 * J1 + oi.j2;
                  int twoj2min = std::abs(2 * J1 - oj.j2);
                  int twoj2max = 2 * J1 + oj.j2;
                  for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                  {
                    for (int twoj2 = twoj2min; twoj2 <= twoj2max; twoj2 += 2)
                    {
                      double sixj = AngMom::SixJ(oi.j2 / 2., oj.j2 / 2.,  Lambda,
                                                 twoj2 / 2., twoj1 / 2.,  J1);
                      int phase = AngMom::phase((oj.j2 + twoj1) / 2 + J1 + Lambda);
                      double facotrs = sqrt((twoj1 + 1) * (twoj2 + 1)); 
                      double yabicdj = Y3.GetME_pn(J1, twoj1, J1, twoj2, a, b, i, c, d, j);
                      double ycdiabj = Y3.GetME_pn(J1, twoj1, J1, twoj2, c, d, i, a, b, j);
                      zij += 1. / 4 * occ_factor * phase * facotrs * sixj * (xabcd * ycdiabj- yabicdj * xcdab);
                    } // j2
                  } // j1
                } // J1

                J1min = std::abs(oa.j2 - ob.j2) / 2;
                J1max = (oa.j2 + ob.j2) / 2;
                int J2min = std::abs(oc.j2 - od.j2) / 2;
                int J2max = (oc.j2 + od.j2) / 2;
                for (int J1 = J1min; J1 <= J1max; J1++)
                {
                  if (a == b and J1 % 2 > 0)
                    continue;
                  for (int J2 = J2min; J2 <= J2max; J2++)
                  {
                    if (c == d and J2 % 2 > 0)
                      continue;
                    double yabcd = Y2.GetTBME_J(J1, J2, a, b, c, d);
                    double ycdab = Y2.GetTBME_J(J2, J1, c, d, a, b);
                    
                    int twoj1min = std::max(std::abs(oi.j2 - 2 * J1), std::abs(oj.j2 - 2 * J2)) ;
                    int twoj1max = std::min(oi.j2 + 2 * J1, oj.j2 + 2 * J2) ;
                    for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                    {
                      double sixj = AngMom::SixJ(Lambda,     J2,          J1,
                                                 twoj1 / 2., oi.j2 / 2.,  oj.j2 / 2.);
                      int phase = AngMom::phase((oi.j2 + twoj1) / 2 + J1);
                      double facotrs = (twoj1 + 1); 
                      double xabicdj = X3.GetME_pn(J1, J2, twoj1, a, b, i, c, d, j);
                      zij += 1. / 4 * occ_factor * phase * sixj * facotrs * (xabicdj * ycdab);
                    } // j1

                    twoj1min = std::max(std::abs(oi.j2 - 2 * J2), std::abs(oj.j2 - 2 * J1)) ;
                    twoj1max = std::min(oi.j2 + 2 * J2, oj.j2 + 2 * J1) ;
                    for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                    {
                      double sixj = AngMom::SixJ(Lambda,     J2,          J1,
                                                 twoj1 / 2., oj.j2 / 2.,  oi.j2 / 2.);
                      int phase = AngMom::phase((oi.j2 + twoj1) / 2 + J2);
                      double facotrs = (twoj1 + 1); 
                      double xcdiabj = X3.GetME_pn(J2, J1, twoj1, c, d, i, a, b, j);
                      zij -= 1. / 4 * occ_factor * phase * sixj * facotrs * (yabcd * xcdiabj);
                    } // j1

                  } // J2
                } // J1
              } // d
            } // c
          } // b
        } // a
        Z1(i, j) += zij;

        if ( i != j)
          Z1(j, i) += AngMom::phase((oi.j2 - oj.j2)/2 ) * hZ * zij;

      } // j
    } // i
  
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm231st

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1, J2)Lamda_ijkl = -1/2 sum_abc sum_{J3,j1,j2} ( nanb(1-nc) + (1-na)(1-nb)nc ) 
  ///                                           (1-PJ1_ij) sqrt{ (2J1+1)(2J3+1)(2j1+1)(2j2+1) } 
  ///                                            (-)^(J1 + Lamda + jc + j2)  { ji jj J1 } { Lamda  J1  J2 } 
  ///                                                                        { jc j1 J3 } { jc     j2  j1 }
  ///                                                       X^J2_cjab Y^(J3j1,J2j2)Lamda_abiklc
  ///                                       -1/2 sum_abc sum_{J3,J4,j1,j2} ( nanb(1-nc) + (1-na)(1-nb)nc ) 
  ///                                           (1-PJ1_ij) sqrt{ (2J1+1)(2J3+1) }    (2j1+1)(2j2+1)
  ///                                            (-)^(J1 + ji+ jc + j3) { Lamda J4 J3 } { jc j1 J2 } { Lamda  J1  J2 } 
  ///                                                                   { jc    jj j2 } { ji j2 J4 } { ji     j2  jj }
  ///                                                        Y^(J3,J4)Lamda_cjab X^(J4j1,J2j1)0_abiklc
  ///                                        1/2 sum_abc sum_{J3,j1,j2} ( nanb(1-nc) + (1-na)(1-nb)nc ) 
  ///                                           (1-PJ2_kl) sqrt{ (2J2+1)(2J3+1)(2j1+1)(2j2+1) } 
  ///                                            (-)^(J1 + Lamda + jc + j2)  { J2 Lamda J1 } { jk  jl  J2 } 
  ///                                                                        { j1 jc    j2 } { jc  j2  J3 }
  ///                                                        Y^(J1j1,J3j2)Lamda_ijcabk XJ3_abcl
  ///                                        1/2 sum_abc sum_{J3,J4j1,j2} ( nanb(1-nc) + (1-na)(1-nb)nc ) 
  ///                                           (1-PJ2_kl) sqrt{ (2J2+1)(2J4+1)}    (2j1+1)(2j2+1) 
  ///                                            (-)^(J1 + jk + jc + J3)  { jc j1 J1 } { Lamda  J3  J4 } { J2 J1 Lamda } 
  ///                                                                     { jk j2 j3 } { jc     jl  j2 } { j2 jl jk    }
  ///                                                        X^(J1j1,J3j1)_ijcabk Y^(J3,J4)Lamda_abcl
  ///
  void comm232st(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X2 = X.TwoBody;
    auto &X3 = X.ThreeBody;
    auto &Y2 = Y.TwoBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z2 = Z.TwoBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();
    int nch = Z.modelspace->GetNumberTwoBodyChannels();
    double tstart = omp_get_wtime();
    std::vector<std::array<size_t, 2>> channels;
    for (auto &iter : Z.TwoBody.MatEl)
      channels.push_back(iter.first);
    size_t nchans = channels.size();
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t ich = 0; ich < nchans; ich++)
    {
      size_t ch_bra = channels[ich][0];
      size_t ch_ket = channels[ich][1];
      auto &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      auto &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      int nbras = tbc_bra.GetNumberKets();
      int nkets = tbc_ket.GetNumberKets();
      for (int ibra = 0; ibra < nbras; ibra++)
      {
        Ket &bra = tbc_bra.GetKet(ibra);
        int i = bra.p;
        int j = bra.q;
        Orbit &oi = Z.modelspace->GetOrbit(i);
        Orbit &oj = Z.modelspace->GetOrbit(j);
        double ji = 0.5 * oi.j2;
        double jj = 0.5 * oj.j2;
        int ket_min = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ket_min; iket < nkets; iket++)
        {
          double zijkl = 0;
          Ket &ket = tbc_ket.GetKet(iket);
          int k = ket.p;
          int l = ket.q;
          Orbit &ok = Z.modelspace->GetOrbit(k);
          Orbit &ol = Z.modelspace->GetOrbit(l);
          double jk = 0.5 * ok.j2;
          double jl = 0.5 * ol.j2;
          for (auto c : Z.modelspace->all_orbits)
          {
            Orbit &oc = Z.modelspace->GetOrbit(c);
            double jc = 0.5 * oc.j2;

            for (int ch_ab = 0; ch_ab < nch; ch_ab++)
            {
              auto &tbc_ab = X.modelspace->GetTwoBodyChannel(ch_ab);
              int Jab = tbc_ab.J;
              size_t nkets_ab = tbc_ab.GetNumberKets();
              for (size_t iket_ab = 0; iket_ab < nkets_ab; iket_ab++)
              {
                Ket &ket_ab = tbc_ab.GetKet(iket_ab);
                int a = ket_ab.p;
                int b = ket_ab.q;
                Orbit &oa = Z.modelspace->GetOrbit(a);
                Orbit &ob = Z.modelspace->GetOrbit(b);
                double occfactor = oa.occ * ob.occ * (1 - oc.occ) + (1 - oa.occ) * (1 - ob.occ) * oc.occ;
                if (std::abs(occfactor) < 1e-6)
                  continue;
                if (a == b)
                  occfactor *= 0.5; // we sum a<=b, and drop the 1/2, but we still need the 1/2 for a==b

                // P_ij
                bool xcjab_good = ((oj.l + oc.l + tbc_ab.parity) % 2 == X.parity) and (std::abs(oj.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * X.rank_T);
                // Xcjab term  // J3 = Jab
                if ((xcjab_good) and (std::abs(oj.j2 - oc.j2) <= 2 * Jab) and (oj.j2 + oc.j2 >= 2 * Jab))
                {
                  int twoj1_min = std::abs(oi.j2 - 2 * Jab);
                  int twoj1_max = oi.j2 + 2 * Jab;
                  int twoj2_min = std::abs(oc.j2 - 2 * J2);
                  int twoj2_max = oc.j2 + 2 * J2;
                  double xcjab = X2.GetTBME_J(Jab, c, j, a, b);

                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int phasefactor = Z.modelspace->phase((oc.j2 + twoj2) / 2 + J1 + Lambda);
                      double hatfactor = sqrt((2 * Jab + 1.) * (2 * J1 + 1) * (twoj1 + 1) * ( twoj2 + 1));
                      double yabiklc = Y3.GetME_pn(Jab, twoj1, J2, twoj2, a, b, i, k, l, c);
                      double sixj = AngMom::SixJ(oi.j2 / 2.,     oj.j2 / 2.,      J1,
                                                 oc.j2 / 2.,     twoj1 / 2.,      Jab);
                      sixj *=       AngMom::SixJ(Lambda,         J1,              J2,
                                                 oc.j2 / 2.,     twoj2 / 2.,      twoj1 / 2.);
                      zijkl -= occfactor * hatfactor * phasefactor * sixj * xcjab * yabiklc;
                    } // j2
                  } // j1
                }

                bool xciab_good = ((oi.l + oc.l + tbc_ab.parity) % 2 == X.parity) and (std::abs(oi.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * X.rank_T);
                // xciab   // J3 = Jab
                if ((xciab_good) and (std::abs(oi.j2 - oc.j2) <= 2 * Jab) and (oi.j2 + oc.j2 >= 2 * Jab))
                {
                  int twoj1_min = std::abs(oj.j2 - 2 * Jab);
                  int twoj1_max = oj.j2 + 2 * Jab;
                  int twoj2_min = std::abs(oc.j2 - 2 * J2);
                  int twoj2_max = oc.j2 + 2 * J2;
                  double xciab = X2.GetTBME_J(Jab, c, i, a, b);
                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int phasefactor = Z.modelspace->phase((oi.j2 + oj.j2 + oc.j2 + twoj2) / 2  + Lambda);
                      double hatfactor = sqrt((2 * Jab + 1.) * (2 * J1 + 1) * (twoj1 + 1) * ( twoj2 + 1));
                      double yabjklc = Y3.GetME_pn(Jab, twoj1, J2, twoj2, a, b, j, k, l, c);
                      double sixj = AngMom::SixJ(oj.j2 / 2.,     oi.j2 / 2.,      J1,
                                                 oc.j2 / 2.,     twoj1 / 2.,      Jab);
                      sixj *=       AngMom::SixJ(Lambda,         J1,              J2,
                                                 oc.j2 / 2.,     twoj2 / 2.,      twoj1 / 2.);
                      zijkl += occfactor * hatfactor * phasefactor * sixj * xciab * yabjklc;
                    } // j2
                  } // j1
                }

                bool ycjab_good = ((oj.l + oc.l + tbc_ab.parity) % 2 == Y.parity) and (std::abs(oj.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * Y.rank_T);
                // ycjab   // J4 = Jab
                if ((ycjab_good) )
                {
                  int twoj1_min = std::max({std::abs(oi.j2 - 2 * Jab), std::abs(oc.j2 - 2 * J2)});
                  int twoj1_max = std::min({oi.j2 + 2 * Jab, oc.j2 + 2 * J2});

                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    double xabiklc = X3.GetME_pn(Jab, J2, twoj1, a, b, i, k, l, c);
                    int twoj2_min = std::max({std::abs(oc.j2 - 2 * Jab), std::abs(oi.j2 - 2 * J2), std::abs(oj.j2 - 2 * Lambda)});
                    int twoj2_max = std::min({oc.j2 + 2 * Jab, oi.j2 + 2 * J2, oj.j2 + 2 * Lambda});
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int J3min = std::abs(oj.j2 - oc.j2) / 2;
                      int J3max = (oj.j2 + oc.j2) / 2;
                      for (int J3 = J3min; J3 <= J3max; J3++ )
                      {
                        int phasefactor = Z.modelspace->phase((oi.j2 + oc.j2) / 2 + J1 + J3);
                        double hatfactor = sqrt((2 * J3 + 1.) * (2 * J1 + 1) ) * (twoj1 + 1) * ( twoj2 + 1);
                        double ycjab = Y2.GetTBME_J(J3, Jab, c, j, a, b);
              
                        double sixj = AngMom::SixJ(Lambda,         Jab,            J3,
                                                  oc.j2 / 2.,     oj.j2 / 2.,      twoj2 / 2.);
                        sixj *=       AngMom::SixJ(oc.j2 / 2.,    twoj1 / 2.,      J2,
                                                   oi.j2 / 2.,    twoj2 / 2.,      Jab);
                        sixj *=       AngMom::SixJ(J1,            Lambda,          J2,
                                                   twoj2 / 2.,    oi.j2 / 2.,      oj.j2 / 2.);
                        zijkl -= occfactor * hatfactor * phasefactor * sixj * ycjab * xabiklc;
                      } // J3
                    } // j2
                  } // j1
                }

                bool yciab_good = ((oi.l + oc.l + tbc_ab.parity) % 2 == Y.parity) and (std::abs(oi.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * Y.rank_T);
                // yciab   // J4 = Jab  (ci)^J3
                if ((yciab_good) )
                {
                  int twoj1_min = std::max({std::abs(oj.j2 - 2 * Jab), std::abs(oc.j2 - 2 * J2)});
                  int twoj1_max = std::min({oj.j2 + 2 * Jab, oc.j2 + 2 * J2});
                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    double xabjklc = X3.GetME_pn(Jab, J2, twoj1, a, b, j, k, l, c);
                    int twoj2_min = std::max({std::abs(oc.j2 - 2 * Jab), std::abs(oj.j2 - 2 * J2), std::abs(oi.j2 - 2 * Lambda)});
                    int twoj2_max = std::min({oc.j2 + 2 * Jab, oj.j2 + 2 * J2, oi.j2 + 2 * Lambda});
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int J3min = std::abs(oi.j2 - oc.j2) / 2;
                      int J3max = (oi.j2 + oc.j2) / 2;
                      for (int J3 = J3min; J3 <= J3max; J3++ )
                      {
                        if (i == c and J3 % 2 == 1 )
                          continue;
                      
                        int phasefactor = Z.modelspace->phase((oi.j2 + oc.j2) / 2 + J3);
                        double hatfactor = sqrt((2 * J3 + 1.) * (2 * J1 + 1) ) * (twoj1 + 1) * ( twoj2 + 1);
                        double yciab = Y2.GetTBME_J(J3, Jab, c, i, a, b);
              
                        double sixj = AngMom::SixJ(Lambda,        Jab,             J3,
                                                  oc.j2 / 2.,     oi.j2 / 2.,      twoj2 / 2.);

                               sixj *=AngMom::SixJ(oc.j2 / 2.,    twoj1 / 2.,      J2,
                                                   oj.j2 / 2.,    twoj2 / 2.,      Jab);

                               sixj *=AngMom::SixJ(J1,            Lambda,          J2,
                                                   twoj2 / 2.,    oj.j2 / 2.,      oi.j2 / 2.);
                        zijkl -= occfactor * hatfactor * phasefactor * sixj * yciab * xabjklc;
                      } // J3
                    } // j2
                  } // j1
                }

                // P_kl
                bool xabcl_good = ((ol.l + oc.l + tbc_ab.parity) % 2 == X.parity) and (std::abs(ol.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * X.rank_T);
                // Xabcl term  // J3 = Jab
                if ((xabcl_good) and (std::abs(ol.j2 - oc.j2) <= 2 * Jab) and (ol.j2 + oc.j2 >= 2 * Jab))
                {
                  int twoj1_min = std::abs(oc.j2 - 2 * J1);
                  int twoj1_max = oc.j2 + 2 * J1;
                  int twoj2_min = std::abs(ok.j2 - 2 * Jab);
                  int twoj2_max = ok.j2 + 2 * Jab;
                  double xabcl = X2.GetTBME_J(Jab, a, b, c, l);
                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int phasefactor = Z.modelspace->phase((oc.j2 + twoj2) / 2 + J1 + Lambda);
                      double hatfactor = sqrt((2 * Jab + 1.) * (2 * J2 + 1) * (twoj1 + 1) * ( twoj2 + 1));
                      double yijcabk = Y3.GetME_pn(J1, twoj1, Jab, twoj2, i, j, c, a, b, k);

                      double sixj = AngMom::SixJ(J2,             Lambda,          J1,
                                                 twoj1 / 2.,     oc.j2 / 2.,      twoj2 / 2.);
                      sixj *=       AngMom::SixJ(ok.j2 / 2.,     ol.j2 / 2.,      J2,
                                                 oc.j2 / 2.,     twoj2 / 2.,      Jab);
                      zijkl += occfactor * hatfactor * phasefactor * sixj * xabcl * yijcabk;
                    } // j2
                  } // j1
                }

                bool xabck_good = ((ok.l + oc.l + tbc_ab.parity) % 2 == X.parity) and (std::abs(ok.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * X.rank_T);
                // Xabck term  // J3 = Jab
                if ((xabck_good) and (std::abs(ok.j2 - oc.j2) <= 2 * Jab) and (ok.j2 + oc.j2 >= 2 * Jab))
                {
                  int twoj1_min = std::abs(oc.j2 - 2 * J1);
                  int twoj1_max = oc.j2 + 2 * J1;
                  int twoj2_min = std::abs(ol.j2 - 2 * Jab);
                  int twoj2_max = ol.j2 + 2 * Jab;
                  double xabck = X2.GetTBME_J(Jab, a, b, c, k);
                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int phasefactor = Z.modelspace->phase((oc.j2 + twoj2 + ok.j2 + ol.j2) / 2 + J1 + J2 + Lambda);
                      double hatfactor = sqrt((2 * Jab + 1.) * (2 * J2 + 1) * (twoj1 + 1) * ( twoj2 + 1));
                      double yijcabkl = Y3.GetME_pn(J1, twoj1, Jab, twoj2, i, j, c, a, b, l);

                      double sixj = AngMom::SixJ(J2,             Lambda,          J1,
                                                 twoj1 / 2.,     oc.j2 / 2.,      twoj2 / 2.);
                      sixj *=       AngMom::SixJ(ol.j2 / 2.,     ok.j2 / 2.,      J2,
                                                 oc.j2 / 2.,     twoj2 / 2.,      Jab);
                      zijkl -= occfactor * hatfactor * phasefactor * sixj * xabck * yijcabkl;
                    } // j2
                  } // j1
                }

                bool yabcl_good = ((ol.l + oc.l + tbc_ab.parity) % 2 == Y.parity) and (std::abs(ol.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * Y.rank_T);
                // yabcl   // J3 = Jab   (cl)^J4
                if ((yabcl_good) )
                {
                  int twoj1_min = std::max(std::abs(oc.j2 - 2 * J1), std::abs(ok.j2 - 2 * Jab));
                  int twoj1_max = std::min(oc.j2 + 2 * J1, ok.j2 + 2 * Jab);
                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    double xijcabk = X3.GetME_pn(J1, Jab, twoj1, i, j, c, a, b, k);
                    int twoj2_min = std::max({std::abs(oc.j2 - 2 * Jab), std::abs(ok.j2 - 2 * J1), std::abs(ol.j2 - 2 * Lambda)});
                    int twoj2_max = std::min({oc.j2 + 2 * Jab, ok.j2 + 2 * J1, ol.j2 + 2 * Lambda});
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int J4min = std::abs(ol.j2 - oc.j2) / 2;
                      int J4max = (ol.j2 + oc.j2) / 2;
                      for (int J4 = J4min; J4 <= J4max; J4 += 1)
                      {
                        int phasefactor = Z.modelspace->phase((ok.j2 + oc.j2) / 2 + J1 + Jab);
                        double hatfactor = sqrt((2 * J2 + 1.) * (2 * J4 + 1) ) * (twoj1 + 1) * ( twoj2 + 1);
                        double yabcl = Y2.GetTBME_J(Jab, J4, a, b, c, l);
                        double sixj = AngMom::SixJ(oc.j2 / 2.,    twoj1 / 2.,      J1,
                                                   ok.j2 / 2.,    twoj2 / 2.,      Jab);
                        sixj *=       AngMom::SixJ(Lambda,        Jab,             J4,
                                                   oc.j2 / 2.,    ol.j2 / 2.,      twoj2 / 2.);
                        sixj *=       AngMom::SixJ(J2,            J1,              Lambda,
                                                   twoj2 / 2.,    ol.j2 / 2.,      ok.j2 / 2.);
                        zijkl += occfactor * hatfactor * phasefactor * sixj * yabcl * xijcabk;
                      } // J3
                    } // j2
                  } // j1
                }

                bool yabck_good = ((ok.l + oc.l + tbc_ab.parity) % 2 == Y.parity) and (std::abs(ok.tz2 + oc.tz2 - 2 * tbc_ab.Tz) == 2 * Y.rank_T);
                // yabck   // J3 = Jab   (ck)^J4
                if ((yabck_good) )
                {
                  int twoj1_min = std::max(std::abs(oc.j2 - 2 * J1), std::abs(ol.j2 - 2 * Jab));
                  int twoj1_max = std::min(oc.j2 + 2 * J1, ol.j2 + 2 * Jab);
                  for (int twoj1 = twoj1_min; twoj1 <= twoj1_max; twoj1 += 2)
                  {
                    double xijcabl = X3.GetME_pn(J1, Jab, twoj1, i, j, c, a, b, l);
                    int twoj2_min = std::max({std::abs(oc.j2 - 2 * Jab), std::abs(ol.j2 - 2 * J1), std::abs(ok.j2 - 2 * Lambda)});
                    int twoj2_max = std::min({oc.j2 + 2 * Jab, ol.j2 + 2 * J1, ok.j2 + 2 * Lambda});
                    for (int twoj2 = twoj2_min; twoj2 <= twoj2_max; twoj2 += 2)
                    {
                      int J4min = std::abs(ok.j2 - oc.j2) / 2;
                      int J4max = (ok.j2 + oc.j2) / 2;
                      for (int J4 = J4min; J4 <= J4max; J4 += 1)
                      {
                        int phasefactor = Z.modelspace->phase((ok.j2 + oc.j2) / 2 + J1 + J2 + Jab);
                        double hatfactor = sqrt((2 * J2 + 1.) * (2 * J4 + 1) ) * (twoj1 + 1) * ( twoj2 + 1);
                        double yabck = Y2.GetTBME_J(Jab, J4, a, b, c, k);
                        double sixj = AngMom::SixJ(oc.j2 / 2.,    twoj1 / 2.,      J1,
                                                   ol.j2 / 2.,    twoj2 / 2.,      Jab);
                        sixj *=       AngMom::SixJ(Lambda,        Jab,             J4,
                                                   oc.j2 / 2.,    ok.j2 / 2.,      twoj2 / 2.);
                        sixj *=       AngMom::SixJ(J2,            J1,              Lambda,
                                                   twoj2 / 2.,    ok.j2 / 2.,      ol.j2 / 2.);
                        zijkl += occfactor * hatfactor * phasefactor * sixj * yabck * xijcabl;
                      } // J3
                    } // j2
                  } // j1
                }
              } // for iket_ab
            } // for ch2 J3
          } // for c

          // normalize the tbme
          zijkl *= -1.0 / sqrt((1 + bra.delta_pq()) * (1 + ket.delta_pq()));
          Z2.AddToTBME(ch_bra, ch_ket, ibra, iket, zijkl);
        } // for iket
      } // for ibra
    } // for ch
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm232st

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1j1, J2j2)Lamda_ijklmn =   PJ3(ij/k) sum_a  ( X_ka Y^(J1j1, J2j2)Lamda_ijalmn 
  ///                                                      -(-)^(J1+Lambda+j1+ja)sqrt(2 j1 + 1) sqrt(2 j2 + 1) 
  ///                                                           { jk ja Lamda} Y^Lamda_ka X^(J1j2, J2j2)0_ijalmn)
  ///                                                           { j2 j1 J1   }
  ///                                                                                 
  ///                                             - PJ3(lm/n) sum_a ( Y^(J1j1, J2j2)Lamda_ijklma X_an 
  ///                                                       -(-)^(J2+Lambda+j1+jn) sqrt(2 j1 + 1) sqrt(2 j2 + 1) 
  ///                                                           { jn ja Lamda} X^(J1j1,J2j1)0_ijklma Y^Lamda_an )
  ///                                                           { j1 j2 J2   }
  ///
  ///
  void comm133st(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &X1 = X.OneBody;
    auto &Y1 = Y.OneBody;
    auto &Z3 = Z.ThreeBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();
    double tstart = omp_get_wtime();
    std::vector<std::array<size_t, 3>> bra_ket_channels;
    for (auto &it : Z.ThreeBody.Get_ch_start())
    {
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(it.first.ch_bra);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      for (size_t ibra = 0; ibra < nbras3; ibra++)
      {
        bra_ket_channels.push_back({it.first.ch_bra, it.first.ch_ket, static_cast<size_t>(ibra)}); // (ch_bra, ch_ket,ibra)
      }
    }
    size_t n_bra_ket_ch = bra_ket_channels.size();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t ich = 0; ich < n_bra_ket_ch; ich++)
    {
      size_t ch_bra = bra_ket_channels[ich][0];
      size_t ch_ket = bra_ket_channels[ich][1];
      size_t ibra = bra_ket_channels[ich][2];
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(ch_bra);
      ThreeBodyChannel &Tbc_ket = Z.modelspace->GetThreeBodyChannel(ch_ket);

      int twoj1 = Tbc_bra.twoJ;
      int twoj2 = Tbc_ket.twoJ;
      // std::cout<< ch_bra << "   " << ch_ket << "   "<< twoj1 << "   "<< twoj2 <<"   "<< Tbc_bra.parity <<"   "<< Tbc_ket.parity<<"   "<< Tbc_bra.twoTz <<"   "<< Tbc_ket.twoTz << std::endl;

      size_t nbras = Tbc_bra.GetNumberKets();
      size_t nkets = Tbc_ket.GetNumberKets();
      Ket3 &bra = Tbc_bra.GetKet(ibra);
      size_t i = bra.p;
      size_t j = bra.q;
      size_t k = bra.r;
      Orbit &oi = Z.modelspace->GetOrbit(i);
      Orbit &oj = Z.modelspace->GetOrbit(j);
      Orbit &ok = Z.modelspace->GetOrbit(k);
      int Jij = bra.Jpq;
      size_t ket_min = (ch_bra == ch_ket) ? ibra : 0;
      for (size_t iket = ket_min; iket < nkets; iket++)
      {
        Ket3 &ket = Tbc_ket.GetKet(iket);
        size_t l = ket.p;
        size_t m = ket.q;
        size_t n = ket.r;
        Orbit &ol = Z.modelspace->GetOrbit(l);
        Orbit &om = Z.modelspace->GetOrbit(m);
        Orbit &on = Z.modelspace->GetOrbit(n);
        int Jlm = ket.Jpq;

        double zsum = 0;
        // First, connect on the bra side
        for (auto a : X.GetOneBodyChannel(ok.l, ok.j2, ok.tz2))
        {
          // eq.(9.1)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != ok.j2)
            continue;
          zsum += X1(k, a) * Y3.GetME_pn(Jij, twoj1, Jlm, twoj2, i, j, a, l, m, n);
        }
        for (auto a : Y.GetOneBodyChannel(ok.l, ok.j2, ok.tz2))
        {          
          // eq.(9.2)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != ok.j2)
            continue;

          int phasefactor = Z.modelspace->phase((oa.j2 + twoj1) / 2 + Jij + Lambda);
          double sixj = AngMom::SixJ(ok.j2 / 2.,     oa.j2 / 2.,      Lambda,
                                     twoj2 / 2.,     twoj1 / 2.,      Jij);
          double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1) );
          zsum -= phasefactor * hatfactor * sixj * Y1(k, a) * X3.GetME_pn(Jij, Jlm, twoj2, i, j, a, l, m, n);
        }
        for (auto a : X.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
        {
          // eq.(9.3)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != oi.j2)
            continue;
          int J3min = std::abs(ok.j2 - oj.j2) / 2;
          int J3max = (ok.j2 + oj.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            double hatfactor = sqrt( ( 2 * Jij + 1) * ( 2 * J3 + 1) );
            double sixj = AngMom::SixJ(oi.j2 / 2.,     oj.j2 / 2.,      Jij,
                                       ok.j2 / 2.,     twoj1 / 2.,      J3);
            zsum += sixj * hatfactor * X1(i, a) * Y3.GetME_pn(J3, twoj1, Jlm, twoj2, k, j, a, l, m, n);
          }
        }
        
        // for( auto a : X.modelspace->all_orbits )
        for (auto a : Y.GetOneBodyChannel(oi.l, oi.j2, oi.tz2))
        {
          // eq.(9.4)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != oi.j2)
            continue;

          int J3min = std::abs(ok.j2 - oj.j2) / 2;
          int J3max = (ok.j2 + oj.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            int phasefactor = Z.modelspace->phase((oa.j2 + twoj1) / 2 + J3 + Lambda);
            double hatfactor = sqrt( ( 2 * Jij + 1) * ( 2 * J3 + 1) * (twoj1 + 1) * ( twoj2 + 1) );
            double sixj = AngMom::SixJ(twoj2 / 2.,     oa.j2 / 2.,      J3,
                                       oi.j2 / 2.,     twoj1 / 2.,      Lambda);
            sixj       *= AngMom::SixJ(oi.j2 / 2.,     oj.j2 / 2.,      Jij,
                                       ok.j2 / 2.,     twoj1 / 2.,      J3);
            zsum -= phasefactor * hatfactor * sixj * Y1(i, a) * X3.GetME_pn(J3, Jlm, twoj2, k, j, a, l, m, n);
          }
        }
        for (auto a : X.GetOneBodyChannel(oj.l, oj.j2, oj.tz2))
        {
          // eq.(9.5)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != oj.j2)
            continue;
          int J3min = std::abs(ok.j2 - oi.j2) / 2;
          int J3max = (ok.j2 + oi.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            int phasefactor = Z.modelspace->phase((oj.j2 + ok.j2) / 2 + Jij + J3);
            double hatfactor = sqrt( ( 2 * Jij + 1) * ( 2 * J3 + 1) );

            double sixj = AngMom::SixJ(oj.j2 / 2.,     oi.j2 / 2.,      Jij,
                                       ok.j2 / 2.,     twoj1 / 2.,      J3);
            
            zsum -= phasefactor * hatfactor * sixj * X1(j, a) * Y3.GetME_pn(J3, twoj1, Jlm, twoj2, i, k, a, l, m, n);
          }
        }
        for (auto a : Y.GetOneBodyChannel(oj.l, oj.j2, oj.tz2))
        {
          // eq.(9.6)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != oj.j2)
            continue;

          int J3min = std::abs(ok.j2 - oi.j2) / 2;
          int J3max = (ok.j2 + oi.j2) / 2;
          int phasefactor = Z.modelspace->phase((oa.j2 + ok.j2 + oj.j2 + twoj1) / 2 + Jij + Lambda);
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            double hatfactor = sqrt( ( 2 * Jij + 1) * ( 2 * J3 + 1) * (twoj1 + 1) * ( twoj2 + 1) );
            double sixj = AngMom::SixJ(twoj2 / 2.,     oa.j2 / 2.,      J3,
                                       oj.j2 / 2.,     twoj1 / 2.,      Lambda);
            sixj       *= AngMom::SixJ(oj.j2 / 2.,     oi.j2 / 2.,      Jij,
                                       ok.j2 / 2.,     twoj1 / 2.,      J3);
            zsum += phasefactor * hatfactor * sixj * Y1(j, a) * X3.GetME_pn(J3, Jlm, twoj2, i, k, a, l, m, n);
          }
        }

        // Now connect on the ket side
        for (auto a : X.GetOneBodyChannel(on.l, on.j2, on.tz2))
        {
          // eq.(10.1)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != on.j2)
            continue;
          zsum -= Y3.GetME_pn(Jij, twoj1, Jlm, twoj2, i, j, k, l, m, a) * X1(a, n);
        }
        for (auto a : Y.GetOneBodyChannel(on.l, on.j2, on.tz2))
        {
          // eq.(10.2)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != on.j2)
            continue;
          int phasefactor = Z.modelspace->phase(( on.j2 + twoj1 ) / 2 + Jlm + Lambda);
          double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1) );
          double sixj = AngMom::SixJ(on.j2 / 2.,     oa.j2 / 2.,      Lambda,
                                     twoj1 / 2.,     twoj2 / 2.,      Jlm);
          zsum += phasefactor * hatfactor * sixj * X3.GetME_pn(Jij, Jlm, twoj1, i, j, k, l, m, a) * Y1(a, n);
        }
        for (auto a : X.GetOneBodyChannel(ol.l, ol.j2, ol.tz2))
        {
          // eq.(10.3)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != ol.j2)
            continue;
          int J3min = std::abs(on.j2 - om.j2) / 2;
          int J3max = (on.j2 + om.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            double hatfactor = sqrt( (2 * Jlm + 1) * ( 2 * J3 + 1) );
            double sixj = AngMom::SixJ(ol.j2 / 2.,     om.j2 / 2.,      Jlm,
                                       on.j2 / 2.,     twoj2 / 2.,      J3);
            zsum -= hatfactor * sixj * Y3.GetME_pn(Jij, twoj1, J3, twoj2, i, j, k, n, m, a) * X1(a, l);
          }
        }
        for (auto a : Y.GetOneBodyChannel(ol.l, ol.j2, ol.tz2))
        {
          // eq.(10.4)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != ol.j2)
            continue;
          int J3min = std::abs(on.j2 - om.j2) / 2;
          int J3max = (on.j2 + om.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            if ( oa.j2 + ol.j2 < Lambda * 2 and std::abs( oa.j2 - ol.j2 ) > Lambda * 2 )
              continue;
            int phasefactor = Z.modelspace->phase(( ol.j2 + twoj1 ) / 2 + J3 + Lambda);
            double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1) * ( 2 * J3 + 1) * ( 2 * Jlm + 1) );
            double sixj = AngMom::SixJ(on.j2 / 2.,     om.j2 / 2.,      J3,
                                       ol.j2 / 2.,     twoj2 / 2.,      Jlm);
            sixj       *= AngMom::SixJ(ol.j2 / 2.,     oa.j2 / 2.,      Lambda,
                                       twoj1 / 2.,     twoj2 / 2.,      J3);
            zsum += phasefactor * hatfactor * sixj * X3.GetME_pn(Jij, J3, twoj1, i, j, k, n, m, a) * Y1(a, l);
          }
        }
        for (auto a : X.GetOneBodyChannel(om.l, om.j2, om.tz2))
        {
          // eq.(10.5)
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != om.j2)
            continue;
          int J3min = std::abs(ol.j2 - on.j2) / 2;
          int J3max = (ol.j2 + on.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            int phasefactor = Z.modelspace->phase(( om.j2 + on.j2 ) / 2 + J3 + Jlm);
            double hatfactor = sqrt( ( 2 * J3 + 1) * ( 2 * Jlm + 1) );
            double sixj = AngMom::SixJ(om.j2 / 2.,     ol.j2 / 2.,      Jlm,
                                       on.j2 / 2.,     twoj2 / 2.,      J3);
            zsum += phasefactor * hatfactor * sixj * Y3.GetME_pn(Jij, twoj1, J3, twoj2, i, j, k, l, n, a) * X1(a, m);
          }
        }
        for (auto a : Y.GetOneBodyChannel(om.l, om.j2, om.tz2))
        {
          // eq.(10.6)    
          Orbit &oa = Z.modelspace->GetOrbit(a);
          if (oa.j2 != om.j2)
            continue;      
          int J3min = std::abs(ol.j2 - on.j2) / 2;
          int J3max = (ol.j2 + on.j2) / 2;
          for (int J3 = J3min; J3 <= J3max; J3++ )
          {
            if ( oa.j2 + om.j2 < Lambda * 2 and std::abs( oa.j2 - om.j2 ) > Lambda * 2 )
              continue;
            int phasefactor = Z.modelspace->phase(( on.j2 + twoj1 ) / 2 + Jlm + Lambda);
            double hatfactor = sqrt( ( 2 * J3 + 1) * ( 2 * Jlm + 1) *  (twoj1 + 1) * ( twoj2 + 1));
            double sixj = AngMom::SixJ(on.j2 / 2.,     ol.j2 / 2.,      J3,
                                       om.j2 / 2.,     twoj2 / 2.,      Jlm);
            sixj       *= AngMom::SixJ(om.j2 / 2.,     oa.j2 / 2.,      Lambda,
                                       twoj1 / 2.,     twoj2 / 2.,      J3);                           
            zsum += phasefactor * hatfactor * sixj * X3.GetME_pn(Jij, J3, twoj1, i, j, k, l, n, a) * Y1(a, m);
          }
        }

        Z3.AddToME_pn_ch(ch_bra, ch_ket, ibra, iket, zsum);

      } // for iket
    } // for ich  -> {ch_bra,ch_ket,ibra}
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm133st


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1,J2)Lamda_ijkl =  sum_{a,b,j1,j2} (na-nb) (-)^(J1+ja+j2+Lamda) Sqrt[(2j1+1) (2j2+1)]
  ///                                                       { Lamda  J1     J2 } A_ab B^(J1j1,J2j2)Lamda_ijbkla
  ///                                                       { ja     j2     j1 }
  //                                        sum_{a,b,j1}    (na-nb) (-)^(J2+ja+j1) (2j1+1)
  ///                                                       { J2     Lamda  J1 } B_ab A^(J1j1,J2j1)0_ijbkla
  ///                                                       { jb     j1     ja }
  ///
  void comm132st(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X1 = X.OneBody;
    auto &Y1 = Y.OneBody;
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z2 = Z.TwoBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();
    double tstart = omp_get_wtime();
    std::vector<size_t> ch_bra_list, ch_ket_list;

    for (auto &iter : Z.TwoBody.MatEl)
    {
      ch_bra_list.push_back(iter.first[0]);
      ch_ket_list.push_back(iter.first[1]);
    }
    int nch = ch_bra_list.size();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int ich = 0; ich < nch; ich++)
    {
      size_t ch_bra = ch_bra_list[ich];
      size_t ch_ket = ch_ket_list[ich];
      //      TwoBodyChannel &tbc = Z.modelspace->GetTwoBodyChannel(ch2);
      TwoBodyChannel &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      int nbras = tbc_bra.GetNumberKets();
      int nkets = tbc_ket.GetNumberKets();
      for (int ibra = 0; ibra < nbras; ibra++)
      {
        Ket &bra = tbc_bra.GetKet(ibra);
        size_t i = bra.p;
        size_t j = bra.q;
        size_t ket_min = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ket_min; iket < nkets; iket++)
        {
          Ket &ket = tbc_ket.GetKet(iket);
          size_t k = ket.p;
          size_t l = ket.q;
          double zijkl = 0;
          for (size_t a : Z.modelspace->all_orbits)
          {
            Orbit &oa = Z.modelspace->GetOrbit(a);
            for (size_t b : Z.modelspace->all_orbits)
            {
              Orbit &ob = Z.modelspace->GetOrbit(b);
              if ( oa.j2 != ob.j2 )
                continue;
              
              int twoj1min = std::abs(2 * J1 - ob.j2);
              int twoj1max = 2 * J1 + ob.j2;
              int twoj2min = std::abs(2 * J2 - oa.j2);
              int twoj2max = 2 * J2 + oa.j2;
              double xab = X1(a, b);
              if (std::abs(xab) > 1e-8)
                for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                {
                  for (int twoj2 = twoj2min; twoj2 <= twoj2max; twoj2 += 2)
                  {
                    int phasefactor = Z.modelspace->phase((oa.j2 + twoj2) / 2 + J1 + Lambda);
                    double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1) );
                    double sixj = AngMom::SixJ( Lambda    ,     J1,              J2,
                                                oa.j2 / 2.,     twoj2 / 2.,      twoj1 / 2.);
                    double yijbkla = Y3.GetME_pn(J1, twoj1, J2, twoj2, i, j, b, k, l, a);
                    zijkl += phasefactor * hatfactor * sixj * (oa.occ - ob.occ) * (xab * yijbkla );
                  } // twoj2
                } // twoj1


              twoj1min = std::max(std::abs(2 * J1 - ob.j2), std::abs(2 * J2 - oa.j2));
              twoj1max = std::min(2 * J1 + ob.j2, 2 * J2 + oa.j2);
              double yab = Y1(a, b);
              if (std::abs(yab) > 1e-8)
                for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                {
                  double xijbkla = X3.GetME_pn(J1, J2, twoj1, i, j, b, k, l, a);
                  if (std::abs(xijbkla) < 1e-8)
                    continue;
                  int phasefactor = Z.modelspace->phase((oa.j2 + twoj1) / 2 + J2);
                  double hatfactor = (twoj1 + 1);
                  double sixj = AngMom::SixJ( J2    ,         Lambda,          J1,
                                              ob.j2 / 2.,     twoj1 / 2.,      oa.j2 / 2.);
                  zijkl -= phasefactor * hatfactor * sixj * (oa.occ - ob.occ) * (yab * xijbkla);
                } // twoj1
            } // b
          } // a
          zijkl /= sqrt((1. + bra.delta_pq()) * (1. + ket.delta_pq()));
          Z2.AddToTBME(ch_bra, ch_ket, ibra, iket, zijkl);
        } // iket
      } // ibra
    } // ch2
  
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm132st

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1,J2)Lamda_ijkl = 1/6 sum_abcd sum_{J3,j1,j2} (-)^( J1+Lamda+jd+j2 ) sqrt( (2j1+1) * (2j2+1) )
  ///                                       *  (na nb nc(1-nd) + (1-na)(1-nb)(1-nc)nd)  { J1 J2 Lamda }
  ///                                                                                   { j2 j1 jd    }
  ///                                       *  ( X^(J1j1,J3j1)0_ijdabc Y^(J3j1,J2j2)Lamda_abckld  
  ///                                          - Y^(J1j1,J3j2)Lamda_ijdabc X^(J3j2,J2j2)0_abckld )
  ///
  void comm332_ppph_hhhpst(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z2 = Z.TwoBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();
    double tstart = omp_get_wtime();
    std::vector<int> bra_channels;
    std::vector<int> ket_channels;
    for (auto &itmat : Z.TwoBody.MatEl)
    {
      bra_channels.push_back(itmat.first[0]);
      ket_channels.push_back(itmat.first[1]);
    }
    int nmat = bra_channels.size();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ch2 = 0; ch2 < nmat; ch2++)
    {
      int ch_bra = bra_channels[ch2];
      int ch_ket = ket_channels[ch2];
      TwoBodyChannel &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      int nbras = tbc_bra.GetNumberKets();
      int nkets = tbc_ket.GetNumberKets();
      for (int ibra = 0; ibra < nbras; ibra++)
      {
        Ket &bra = tbc_bra.GetKet(ibra);
        size_t i = bra.p;
        size_t j = bra.q;
        Orbit &oi = Z.modelspace->GetOrbit(i);
        Orbit &oj = Z.modelspace->GetOrbit(j);
        int ket_min = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ket_min; iket < nkets; iket++)
        {
          Ket &ket = tbc_ket.GetKet(iket);
          size_t k = ket.p;
          size_t l = ket.q;
          double zijkl = 0;
          for (size_t a : Z.modelspace->all_orbits)
          {
            Orbit &oa = Z.modelspace->GetOrbit(a);
            for (size_t b : Z.modelspace->all_orbits)
            {
              Orbit &ob = Z.modelspace->GetOrbit(b);
              for (size_t c : Z.modelspace->all_orbits)
              {
                Orbit &oc = Z.modelspace->GetOrbit(c);
                for (size_t d : Z.modelspace->all_orbits)
                {
                  Orbit &od = Z.modelspace->GetOrbit(d);
                  double occfactor = oa.occ * ob.occ * oc.occ * (1 - od.occ) - (1 - oa.occ) * (1 - ob.occ) * (1 - oc.occ) * od.occ;
                  if (std::abs(occfactor) < 1e-7)
                    continue;
                  //if ((oi.l + oj.l + od.l + oa.l + ob.l + oc.l) % 2 > 0)
                  //  continue;
                  //if ((oi.tz2 + oj.tz2 + od.tz2) - (oa.tz2 + ob.tz2 + oc.tz2) != 0)
                  //  continue;
                  int J3min = std::abs(oa.j2 - ob.j2) / 2;
                  int J3max = (oa.j2 + ob.j2) / 2;
                  for (int J3 = J3min; J3 <= J3max; J3++)
                  {
                    int twoj1min = std::abs(2 * J1 - od.j2);
                    int twoj1max = 2 * J1 + od.j2;
                    int twoj2min = std::abs(2 * J2 - od.j2);
                    int twoj2max = 2 * J2 + od.j2;
                    for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                    {
                      for (int twoj2 = twoj2min; twoj2 <= twoj2max; twoj2 += 2)
                      {
                        int phasefactor = Z.modelspace->phase((od.j2 + twoj2) / 2 + J1 + Lambda);
                        double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1) );
                        double sixj = AngMom::SixJ(J1,            J2,              Lambda,
                                                   twoj2 / 2.,    twoj1 / 2.,      od.j2 / 2.);
                        // Eq.(7.1)
                        if (  twoj1 >= std::abs(2 * J3 - oc.j2) and twoj1 <= (2 * J3 + oc.j2) )
                        {
                          double xijdabc = X3.GetME_pn(J1, J3, twoj1, i, j, d, a, b, c);
                          double yabckld = Y3.GetME_pn(J3, twoj1, J2, twoj2, a, b, c, k, l, d);
                          zijkl += 1. / 6 * occfactor * phasefactor * hatfactor * sixj * (xijdabc * yabckld);
                        }
                        // Eq.(7.2)
                        if (  twoj2 >= std::abs(2 * J3 - oc.j2) and twoj2 <= (2 * J3 + oc.j2) )
                        {
                          double yijdabc = Y3.GetME_pn(J1, twoj1, J3, twoj2, i, j, d, a, b, c);
                          double xabckld = X3.GetME_pn(J3, J2, twoj2, a, b, c, k, l, d);
                          zijkl -= 1. / 6 * occfactor * phasefactor * hatfactor * sixj * (yijdabc * xabckld);
                        }
                      } // twoj1
                    } // twoj2
                  } // J2
                } // d
              } // c
            } // b
          } // a
          
          zijkl /= sqrt((1. + bra.delta_pq()) * (1. + ket.delta_pq()));
          Z2.AddToTBME(ch_bra, ch_ket, ibra, iket, zijkl);
        } // iket
      } // ibra
    } // ch2
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm332_ppph_hhhpst

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1,J2)Lamda_ijkl = 1/4 (1-P^J1_ij) (1-P^J2_kl) sum_abcd sum_{J3 J4 j1 j2 j3 j4}
  ///                                      ( (1-na)(1-nb)nc nd - na nb(1-nc)(1-nd) )
  ///                                      ( (-)^(J2+J3+J4+Lamda+jj+jl) (2j1+1) (2j2+1) sqrt((2J1+1)(2J2+1)(2j2+1) (2j3+1))
  ///                                      { ji j1 J2 } { jj ji J1 } { jk jl J2 } { Lamda J1 J2 }
  ///                                      { jk j4 J4 } { j4 j2 J4 } { j3 j4 J3 } { j4    j3 j2 }
  ///                                       X^(J3j1,J4j1)0_abicdk Y^(J4j2,J3j3)Lamda_cdjabl
  ///                                      - sum_{J5} (-)^(ji+jj+jk+j1+j2+j3+J3) (2j1+1) (2j4+1) 
  ///                                        sqrt((2J1+1)(2J2+1)(2j3+1) (2j4+1))
  ///                                      { ji j1 J3    } { J2 Lamda J1 } { J4 J3 J5 } { J4 J3 J5 } { jk jl J2 }
  ///                                      { j2 j4 Lamda } { ji jj    j4 } { j4 jk j2 } { jl jj j3 } { jj j4 J5 }
  ///                                       Y^(J3j1,J4j2)Lamda_abicdk X^(J4j3,J3j3)0_cdjabl 
  ///                                      )
  ///
  void comm332_pphhst(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z2 = Z.TwoBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();
    double tstart = omp_get_wtime();
    std::map<int, double> e_fermi = Z.modelspace->GetEFermi();

    std::vector<int> bra_channels;
    std::vector<int> ket_channels;
    for (auto &itmat : Z.TwoBody.MatEl)
    {
      bra_channels.push_back(itmat.first[0]);
      ket_channels.push_back(itmat.first[1]);
    }
    int nmat = bra_channels.size();
    int nch = Z.modelspace->GetNumberTwoBodyChannels();
    #pragma omp parallel for schedule(dynamic, 1) if (not Z.modelspace->scalar3b_transform_first_pass)
    for (int ch = 0; ch < nmat; ch++) // loop bra and ket channels
    {
      int ch_bra = bra_channels[ch];
      int ch_ket = ket_channels[ch];
      TwoBodyChannel &tbc_bra = Z.modelspace->GetTwoBodyChannel(ch_bra);
      TwoBodyChannel &tbc_ket = Z.modelspace->GetTwoBodyChannel(ch_ket);
      int J1 = tbc_bra.J;
      int J2 = tbc_ket.J;
      int nbras = tbc_bra.GetNumberKets();
      int nkets = tbc_ket.GetNumberKets();

      for (int ibra = 0; ibra < nbras; ibra++)
      {
        Ket &bra = tbc_bra.GetKet(ibra);
        int i = bra.p;
        int j = bra.q;
        int ji2 = bra.op->j2;
        int jj2 = bra.oq->j2;
        double ji = 0.5 * ji2;
        double jj = 0.5 * jj2;
        double d_ei = std::abs(2 * bra.op->n + bra.op->l - e_fermi[bra.op->tz2]);
        double d_ej = std::abs(2 * bra.oq->n + bra.oq->l - e_fermi[bra.oq->tz2]);
        int ket_min = (ch_bra == ch_ket) ? ibra : 0;
        for (int iket = ket_min; iket < nkets; iket++)
        {
          Ket &ket = tbc_ket.GetKet(iket);
          int k = ket.p;
          int l = ket.q;
          int jk2 = ket.op->j2;
          int jl2 = ket.oq->j2;
          double jk = 0.5 * jk2;
          double jl = 0.5 * jl2;
          double d_ek = std::abs(2 * ket.op->n + ket.op->l - e_fermi[ket.op->tz2]);
          double d_el = std::abs(2 * ket.oq->n + ket.oq->l - e_fermi[ket.oq->tz2]);

          double zijkl = 0;

          for (int ch_ab = 0; ch_ab < nch; ch_ab++) 
          {
            TwoBodyChannel &tbc_ab = Z.modelspace->GetTwoBodyChannel(ch_ab);
            int J3 = tbc_ab.J;  // Jab
            int nkets_ab = tbc_ab.GetNumberKets();
            for (int iket_ab = 0; iket_ab < nkets_ab; iket_ab++)
            {
              Ket &ket_ab = tbc_ab.GetKet(iket_ab);
              int a = ket_ab.p;
              int b = ket_ab.q;
              double na = ket_ab.op->occ;
              double nb = ket_ab.oq->occ;
              for (int ch_cd = 0; ch_cd < nch; ch_cd++) 
              {
                TwoBodyChannel &tbc_cd = Z.modelspace->GetTwoBodyChannel(ch_cd);
                int J4 = tbc_cd.J;  // Jab
                int nkets_cd = tbc_cd.GetNumberKets();
                for (int iket_cd = 0; iket_cd < nkets_cd; iket_cd++)
                {
                  Ket &ket_cd = tbc_cd.GetKet(iket_cd);
                  int c = ket_cd.p;
                  int d = ket_cd.q;
                  double nc = ket_cd.op->occ;
                  double nd = ket_cd.oq->occ;
                  double occupation_factor = na * nb * (1 - nc) * (1 - nd) - (1 - na) * (1 - nb) * nc * nd;
                  if (std::abs(occupation_factor) < 1e-6)
                    continue;

                  double symmetry_factor = 4.; // we only sum a<=b and c<=d, so we undercount by a factor of 4
                  if (a == b)
                    symmetry_factor *= 0.5; // if a==b or c==d, then the permutation doesn't give a new state, so there's less undercounting
                  if (c == d)
                    symmetry_factor *= 0.5;

                  // Eq(6.3) (6.4)
                  int twoj1min = std::abs(2 * J3 - ji2);
                  int twoj1max = 2 * J3 + ji2;
                  for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                  {
                    int twoj2min = std::abs(2 * J4 - jj2);
                    int twoj2max = 2 * J4 + jj2;
                    for (int twoj2 = twoj2min; twoj2 <= twoj2max; twoj2 += 2)
                    {
                      // Eq.(6.3)
                      int twoj3min = std::abs(2 * J3 - jl2);
                      int twoj3max = 2 * J3 + jl2;
                      for (int twoj3 = twoj3min; twoj3 <= twoj3max; twoj3 += 2)
                      {
                        int twoj4min = std::abs(2 * J4 - ji2);
                        int twoj4max = 2 * J4 + ji2;
                        for (int twoj4 = twoj4min; twoj4 <= twoj4max; twoj4 += 2)
                        {
                          if (  twoj1 >= std::abs(2 * J4 - jk2) and twoj1 <= (2 * J4 + jk2) )
                          {
                            int phasefactor = Z.modelspace->phase((jj2+ jl2 + twoj3 + twoj4) / 2 + J2 + J3 + J4 + Lambda);
                            double hatfactor = (twoj1 + 1) * (twoj4 + 1) * sqrt( (twoj2 + 1) * ( twoj3 + 1) * (2 * J1 + 1) * (2 * J2 + 1) );
                            
                            double sixj = AngMom::SixJ(ji2   / 2.,    twoj1 / 2.,      J3,
                                                       jk2   / 2.,    twoj4 / 2.,      J4);
                            sixj       *= AngMom::SixJ(jj2   / 2.,    ji2   / 2.,      J1,
                                                       twoj4 / 2.,    twoj2 / 2.,      J4);
                            sixj       *= AngMom::SixJ(jk2   / 2.,    jl2   / 2.,      J2,
                                                       twoj3 / 2.,    twoj4 / 2.,      J3);
                            sixj       *= AngMom::SixJ(Lambda,        J1,              J2,
                                                       twoj4 / 2.,    twoj3 / 2.,      twoj2 / 2.);
                            double xabickd = X3.GetME_pn(J3, J4, twoj1, a, b, i, c, d, k);
                            double ycdjabl = Y3.GetME_pn(J4, twoj2, J3, twoj3, c, d, j, a, b, l);
                            zijkl += 1. / 4 * symmetry_factor * occupation_factor * phasefactor * hatfactor * sixj * (xabickd * ycdjabl);
                            
                          } // delta
                        } // towj4
                      } // twoj3

                      // Eq.(6.4)  k <-> l
                      twoj3min = std::abs(2 * J3 - jk2);
                      twoj3max = 2 * J3 + jk2;
                      for (int twoj3 = twoj3min; twoj3 <= twoj3max; twoj3 += 2)
                      {
                        int twoj4min = std::abs(2 * J4 - ji2);
                        int twoj4max = 2 * J4 + ji2;
                        for (int twoj4 = twoj4min; twoj4 <= twoj4max; twoj4 += 2)
                        {
                          if (  twoj1 >= std::abs(2 * J4 - jl2) and twoj1 <= (2 * J4 + jl2) )
                          {
                            int phasefactor = Z.modelspace->phase((jj2+ jl2 + twoj3 + twoj4) / 2 + J3 + J4 + Lambda);
                            double hatfactor = (twoj1 + 1) * (twoj4 + 1) * sqrt( (twoj2 + 1) * ( twoj3 + 1) * (2 * J1 + 1) * (2 * J2 + 1) );
                            double sixj = AngMom::SixJ(ji2   / 2.,    twoj1 / 2.,      J3,
                                                      jl2   / 2.,    twoj4 / 2.,      J4);
                            sixj       *= AngMom::SixJ(jj2   / 2.,    ji2   / 2.,      J1,
                                                      twoj4 / 2.,    twoj2 / 2.,      J4);
                            sixj       *= AngMom::SixJ(jl2   / 2.,    jk2   / 2.,      J2,
                                                      twoj3 / 2.,    twoj4 / 2.,      J3);
                            sixj       *= AngMom::SixJ(Lambda,       J1,              J2,
                                                      twoj4 / 2.,    twoj3 / 2.,      twoj2 / 2.);
                            double xabicdl = X3.GetME_pn(J3, J4, twoj1, a, b, i, c, d, l);
                            double ycdjabk = Y3.GetME_pn(J4, twoj2, J3, twoj3, c, d, j, a, b, k);
                            zijkl += 1. / 4 * symmetry_factor * occupation_factor * phasefactor * hatfactor * sixj * (xabicdl * ycdjabk);
                          }
                        } // towj4
                      } // twoj3   
                    } // twoj2
                  } // twoj1

                  // Eq(6.5) (6.6)
                  twoj1min = std::abs(2 * J3 - jj2);
                  twoj1max = 2 * J3 + jj2;
                  for (int twoj1 = twoj1min; twoj1 <= twoj1max; twoj1 += 2)
                  {
                    int twoj2min = std::abs(2 * J4 - ji2);
                    int twoj2max = 2 * J4 + ji2;
                    for (int twoj2 = twoj2min; twoj2 <= twoj2max; twoj2 += 2)
                    {
                      // Eq.(6.5) i<->j
                      int twoj3min = std::abs(2 * J3 - jl2);
                      int twoj3max = 2 * J3 + jl2;
                      for (int twoj3 = twoj3min; twoj3 <= twoj3max; twoj3 += 2)
                      {
                        int twoj4min = std::abs(2 * J4 - jj2);
                        int twoj4max = 2 * J4 + jj2;
                        for (int twoj4 = twoj4min; twoj4 <= twoj4max; twoj4 += 2)
                        {
                          if (  twoj1 >= std::abs(2 * J4 - jk2) and twoj1 <= (2 * J4 + jk2) )
                          {
                            int phasefactor = Z.modelspace->phase((jj2+ jl2 + twoj3 + twoj4) / 2 + J1 + J2 + J3 + J4 + Lambda);
                            double hatfactor = (twoj1 + 1) * (twoj4 + 1) * sqrt( (twoj2 + 1) * ( twoj3 + 1) * (2 * J1 + 1) * (2 * J2 + 1) );
                            
                            double sixj = AngMom::SixJ(jj2   / 2.,    twoj1 / 2.,      J3,
                                                       jk2   / 2.,    twoj4 / 2.,      J4);
                            sixj       *= AngMom::SixJ(ji2   / 2.,    jj2   / 2.,      J1,
                                                       twoj4 / 2.,    twoj2 / 2.,      J4);
                            sixj       *= AngMom::SixJ(jk2   / 2.,    jl2   / 2.,      J2,
                                                       twoj3 / 2.,    twoj4 / 2.,      J3);
                            sixj       *= AngMom::SixJ(Lambda,        J1,              J2,
                                                       twoj4 / 2.,    twoj3 / 2.,      twoj2 / 2.);
                            double xabjcdk = X3.GetME_pn(J3, J4, twoj1, a, b, j, c, d, k);
                            double ycdiabl = Y3.GetME_pn(J4, twoj2, J3, twoj3, c, d, i, a, b, l);
                            zijkl += 1. / 4 * symmetry_factor * occupation_factor * phasefactor * hatfactor * sixj * (xabjcdk * ycdiabl);
                          }
                        } // towj4
                      } // twoj3

                      // Eq.(6.6) i<->j k<->l
                      twoj3min = std::abs(2 * J3 - jk2);
                      twoj3max = 2 * J3 + jk2;
                      for (int twoj3 = twoj3min; twoj3 <= twoj3max; twoj3 += 2)
                      {
                        int twoj4min = std::abs(2 * J4 - jj2);
                        int twoj4max = 2 * J4 + jj2;
                        for (int twoj4 = twoj4min; twoj4 <= twoj4max; twoj4 += 2)
                        {
                          if (  twoj1 >= std::abs(2 * J4 - jl2) and twoj1 <= (2 * J4 + jl2) )
                          {
                            int phasefactor = Z.modelspace->phase((jj2+ jl2 + twoj3 + twoj4) / 2 + J1 + J3 + J4 + Lambda);
                            double hatfactor = (twoj1 + 1) * (twoj4 + 1) * sqrt( (twoj2 + 1) * ( twoj3 + 1) * (2 * J1 + 1) * (2 * J2 + 1) );
                            double sixj = AngMom::SixJ(jj2   / 2.,    twoj1 / 2.,      J3,
                                                       jl2   / 2.,    twoj4 / 2.,      J4);
                            sixj       *= AngMom::SixJ(ji2   / 2.,    jj2   / 2.,      J1,
                                                       twoj4 / 2.,    twoj2 / 2.,      J4);
                            sixj       *= AngMom::SixJ(jl2   / 2.,    jk2   / 2.,      J2,
                                                       twoj3 / 2.,    twoj4 / 2.,      J3);
                            sixj       *= AngMom::SixJ(Lambda,        J1,              J2,
                                                       twoj4 / 2.,    twoj3 / 2.,      twoj2 / 2.);
                            double xabjcdl = X3.GetME_pn(J3, J4, twoj1, a, b, j, c, d, l);
                            double ycdiabk = Y3.GetME_pn(J4, twoj2, J3, twoj3, c, d, i, a, b, k);
                            zijkl += 1. / 4 * symmetry_factor * occupation_factor * phasefactor * hatfactor * sixj * (xabjcdl * ycdiabk);
                          }
                        } // towj4
                      } // twoj3  
                    } // twoj2
                  } // twoj1
                } // for iket_cd
              } // ch_cd
            } // iket_ab
          } // ch_ab
          // make it a normalized TBME
          zijkl /= sqrt((1. + bra.delta_pq()) * (1. + ket.delta_pq()));
          // the AddToTBME routine automatically takes care of the hermitian conjugate as well
          Z2.AddToTBME(ch_bra, ch_ket, ibra, iket, zijkl);
        } // for iket
      } // for ibra
    } // for ch
 
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1j1,J2j2)Lamda_ijklmn =  1/2 sum_{ab} ((1-na)(1-nb)-na nb) 
  ///                                                * (  sum_J3  PJ1J3(ij/k) X^(J1,J1)0_ijab Y^(J1j1,J2j2)Lamda_abklmn
  ///                                                    -sum_J3J4  PJ3J4(ij/k) (-)^(j2+jk+Lamda+J1) sqrt((2j1+1) * (2j2+1)) 
  ///                                                             { J3 Lambda J1 }Y^(J1,J3)Lamda_ijab X^(J3j2,J2j2)0_abklmn
  ///                                                             { j1 jk     j2 }           
  ///                                                    -sum_J3  PJ1J3(lm/n) Y^(J1j1,J2j2)Lamda_ijkabn X^(J2,J2)0_ablm 
  ///                                                    +sum_J3J4  PJ3J4(lm/n) (-)^(j2+jn+Lamda+J3) sqrt((2j1+1) * (2j2+1))  
  ///                                                             { J3 J1 lambda } X^(J1j1,J3j1)0_ijkabn Y^(J3,J2)Lamda_ablm )
  ///                                                             { j2 j1 jn     }
  ///
  /// THIS VERSION (scalar) IS STILL TOO SLOW FOR GOING BEYOND EMAX=2... (Rangar)
  /// It's also ture for tensor (Bingcheng)
  void comm233_pp_hhst(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X2 = X.TwoBody;
    auto &Y2 = Y.TwoBody;
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z3 = Z.ThreeBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();

    // Permutations of indices which are needed to produce antisymmetrized matrix elements  P(ij/k) |ijk> = |ijk> - |kji> - |ikj>
    const std::array<ThreeBodyStorage::Permutation, 3> index_perms = {ThreeBodyStorage::ABC, ThreeBodyStorage::CBA, ThreeBodyStorage::ACB};
    double tstart = omp_get_wtime();
    std::vector<std::array<size_t, 3>> bra_ket_channels;
    for (auto &it : Z.ThreeBody.Get_ch_start())
    {
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(it.first.ch_bra);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      for (size_t ibra = 0; ibra < nbras3; ibra++)
      {
        bra_ket_channels.push_back({it.first.ch_bra, it.first.ch_ket, static_cast<size_t>(ibra)}); // (ch_bra, ch_ket,ibra)
      }
    }

    size_t n_bra_ket_ch = bra_ket_channels.size();
    int n2bch = Z.modelspace->GetNumberTwoBodyChannels();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t ibra_ket = 0; ibra_ket < n_bra_ket_ch; ibra_ket++)
    {
      size_t ch3bra = bra_ket_channels[ibra_ket][0];
      size_t ch3ket = bra_ket_channels[ibra_ket][1];
      size_t ibra = bra_ket_channels[ibra_ket][2];
      auto &Tbc_bra = Z.modelspace->GetThreeBodyChannel(ch3bra);
      auto &Tbc_ket = Z.modelspace->GetThreeBodyChannel(ch3ket);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      size_t nkets3 = Tbc_ket.GetNumberKets();
      int twoj1 = Tbc_bra.twoJ; 
      int twoj2 = Tbc_ket.twoJ; 

      auto &bra = Tbc_bra.GetKet(ibra);
      size_t i = bra.p;
      size_t j = bra.q;
      size_t k = bra.r;
      Orbit &oi = Z.modelspace->GetOrbit(i);
      Orbit &oj = Z.modelspace->GetOrbit(j);
      Orbit &ok = Z.modelspace->GetOrbit(k);
      double ji = 0.5 * oi.j2;
      double jj = 0.5 * oj.j2;
      double jk = 0.5 * ok.j2;
      int J1 = bra.Jpq;

      size_t iket_max = nkets3;
      if (ch3bra == ch3ket)
        iket_max = ibra + 1;
      for (size_t iket = 0; iket < iket_max; iket++)
      {
        auto &ket = Tbc_ket.GetKet(iket);
        size_t l = ket.p;
        size_t m = ket.q;
        size_t n = ket.r;
        Orbit &ol = Z.modelspace->GetOrbit(l);
        Orbit &om = Z.modelspace->GetOrbit(m);
        Orbit &on = Z.modelspace->GetOrbit(n);
        double jl = 0.5 * ol.j2;
        double jm = 0.5 * om.j2;
        double jn = 0.5 * on.j2;
        int J2 = ket.Jpq;

        double zijklmn = 0;

        // Now we need to loop over the permutations in ijk and then lmn
        for (auto perm_ijk : index_perms) // {ijk} -> {123}
        {
          size_t I1, I2, I3;
          Z3.Permute(perm_ijk, i, j, k, I1, I2, I3);
          Orbit &o1 = Z.modelspace->GetOrbit(I1);
          Orbit &o2 = Z.modelspace->GetOrbit(I2);
          Orbit &o3 = Z.modelspace->GetOrbit(I3);

          for (int ch_ab = 0; ch_ab < n2bch; ch_ab++) 
          {
            TwoBodyChannel &tbc_ab = Z.modelspace->GetTwoBodyChannel(ch_ab);
            int J3 = tbc_ab.J;  // Jab
            int nkets_ab = tbc_ab.GetNumberKets();
            //if (J3 < std::abs(o1.j2 - o2.j2) / 2 or J3 > (o1.j2 + o2.j2) / 2)
            //  continue;
            // if (perm_ijk == ThreeBodyStorage::ABC and J3 != J1)
            //   continue;
            double Pijk = Z3.PermutationPhase(perm_ijk) * Z3.RecouplingCoefficient(perm_ijk, ji, jj, jk, J3, J1, twoj1);
            for (int iket_ab = 0; iket_ab < nkets_ab; iket_ab++)
            {
              Ket &ket_ab = tbc_ab.GetKet(iket_ab);
              int a = ket_ab.p;
              int b = ket_ab.q;
              double na = ket_ab.op->occ;
              double nb = ket_ab.oq->occ;
              Orbit &oa = *ket_ab.op;
              Orbit &ob = *ket_ab.oq;

              double symmetry_factor = 2.; // we only sum a<=b, so we undercount by a factor of 2
              if (a == b)
                symmetry_factor *= 0.5; // if a==b, then the permutation doesn't give a new state, so there's less undercounting

              if ((o1.l + o2.l + oa.l + ob.l + X.parity) % 2 > 0 and (o1.l + o2.l + oa.l + ob.l + Y.parity) % 2 > 0)
                continue;
              if ((std::abs(o1.tz2 + o2.tz2 - oa.tz2 - ob.tz2) != 2 * X.GetTRank()) and (std::abs(o1.tz2 + o2.tz2 - oa.tz2 - ob.tz2) != 2 * Y.GetTRank()))
                continue;

              double occupation_factor = ((1 - na) * (1 - nb) - na * nb);  
              if (std::abs(occupation_factor) < 1e-7)
                continue;
              if (perm_ijk != ThreeBodyStorage::ABC or J3 == J1)
              {
                double x12ab = X2.GetTBME_J( J3, J3, I1, I2, a, b);
                double yab3lmn = Y3.GetME_pn( J3, twoj1, J2, twoj2, a, b, I3, l, m, n);
                zijklmn += 1. / 2 * symmetry_factor * occupation_factor* Pijk * x12ab * yab3lmn;
              }

              int J4_min = std::max(std::abs(o1.j2 - o2.j2) / 2, std::abs(J3 - Lambda));
              int J4_max = std::min((o1.j2 + o2.j2) / 2, J3 + Lambda);
              for (int J4 = J4_min; J4 <= J4_max; J4++)
              {
                if (perm_ijk == ThreeBodyStorage::ABC and J4 != J1)
                  continue;
                if (I1 == I2 and J4 % 2 != 0)
                  continue;
                double sixj = AngMom::SixJ(J3,          Lambda,        J4,
                                           twoj1 / 2.,  o3.j2 / 2.,    twoj2 / 2.);
                if (std::abs(sixj) < 1e-7)
                  continue;
                
                double Pijk2 = Z3.PermutationPhase(perm_ijk) * Z3.RecouplingCoefficient(perm_ijk, ji, jj, jk, J4, J1, twoj1);

                int phasefactor = Z.modelspace->phase((o3.j2 + twoj2) / 2 + J4 + Lambda);
                double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1));
                double y12ab = Y2.GetTBME_J(J4, J3, I1, I2, a, b);
                double xab3lmn = X3.GetME_pn( J3, J2, twoj2, a, b, I3, l, m, n);
                zijklmn -= 1. / 2 * sixj * symmetry_factor * phasefactor * hatfactor * occupation_factor * Pijk2 * ( y12ab * xab3lmn );
              }
            } // ab
          } // loop 2b channel
        } // for perm_ijk

        for (auto perm_lmn : index_perms) // {lmn} -> {123}
        {
          size_t I1, I2, I3;
          Z3.Permute(perm_lmn, l, m, n, I1, I2, I3);
          Orbit &o1 = Z.modelspace->GetOrbit(I1);
          Orbit &o2 = Z.modelspace->GetOrbit(I2);
          Orbit &o3 = Z.modelspace->GetOrbit(I3);

          for (int ch_ab = 0; ch_ab < n2bch; ch_ab++) 
          {
            TwoBodyChannel &tbc_ab = Z.modelspace->GetTwoBodyChannel(ch_ab);
            int J3 = tbc_ab.J;  // Jab
            int nkets_ab = tbc_ab.GetNumberKets();
            // if (J3 < std::abs(o1.j2 - o2.j2) / 2 or J3 > (o1.j2 + o2.j2) / 2)
            //   continue;

            double Plmn = Z3.PermutationPhase(perm_lmn) * Z3.RecouplingCoefficient(perm_lmn, jl, jm, jn, J3, J2, twoj2);
            for (int iket_ab = 0; iket_ab < nkets_ab; iket_ab++)
            {
              Ket &ket_ab = tbc_ab.GetKet(iket_ab);
              int a = ket_ab.p;
              int b = ket_ab.q;
              double na = ket_ab.op->occ;
              double nb = ket_ab.oq->occ;
              Orbit &oa = *ket_ab.op;
              Orbit &ob = *ket_ab.oq;

              double symmetry_factor = 2.; // we only sum a<=b, so we undercount by a factor of 2
              if (a == b)
                symmetry_factor *= 0.5; // if a==b, then the permutation doesn't give a new state, so there's less undercounting

              if ((o1.l + o2.l + oa.l + ob.l + X.parity) % 2 > 0 and (o1.l + o2.l + oa.l + ob.l + Y.parity) % 2 > 0)
                continue;
              if ((std::abs(o1.tz2 + o2.tz2 - oa.tz2 - ob.tz2) != 2 * X.GetTRank()) and (std::abs(o1.tz2 + o2.tz2 - oa.tz2 - ob.tz2) != 2 * Y.GetTRank()))
                continue;
              double occupation_factor = ((1 - na) * (1 - nb) - na * nb);  
              if (std::abs(occupation_factor) < 1e-7)
                continue;
              if (perm_lmn != ThreeBodyStorage::ABC or J3 == J2)
              {
                double xab12 = X2.GetTBME_J(J3, J3, a, b, I1, I2);
                double yijkab3 = Y3.GetME_pn( J1, twoj1, J3, twoj2, i, j, k, a, b, I3 );
                zijklmn -= 1. / 2 * symmetry_factor * occupation_factor * Plmn * ( yijkab3 * xab12 );
              }
            
              int J4_min = std::max(std::abs(o1.j2 - o2.j2) / 2, std::abs(J3 - Lambda));
              int J4_max = std::min((o1.j2 + o2.j2) / 2, J3 + Lambda);
              for (int J4 = J4_min; J4 <= J4_max; J4++)
              {
                if (perm_lmn == ThreeBodyStorage::ABC and J4 != J2)
                  continue;
                if (I1 == I2 and J4 % 2 != 0)
                  continue;
                double sixj = AngMom::SixJ(J3,          Lambda,        J4,
                                           twoj2 / 2.,  o3.j2 / 2.,    twoj1 / 2.);
                if (std::abs(sixj) < 1e-7)
                  continue;

                double Plmn2 = Z3.PermutationPhase(perm_lmn) * Z3.RecouplingCoefficient(perm_lmn, jl, jm, jn, J4, J2, twoj2);

                int phasefactor = Z.modelspace->phase((o3.j2 + twoj2) / 2 + J3 + Lambda);
                double hatfactor = sqrt( (twoj1 + 1) * ( twoj2 + 1));
                double yab12 = Y2.GetTBME_J(J3, J4, a, b, I1, I2);
                double xijkab3 = X3.GetME_pn(J1, J3, twoj1, i, j, k, a, b, I3);
                zijklmn += 1. / 2 * sixj * symmetry_factor * phasefactor * hatfactor * occupation_factor * Plmn2 * (xijkab3 * yab12);
              }
            } // ab
          } // loop 2b channel
        } // for perm_lmn

        Z3.AddToME_pn_ch(ch3bra, ch3ket, ibra, iket, zijklmn); 
      } // for iket
    } // for ch3 and ibra

    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm233_pp_hhst

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1j1,J2j2)Lamda_ijklmn = sum_ab sum_{J3J4j3j4} (na-nb) PJ1J3(ij/k) PJ2J3(lm/n)
  ///                                                          (-1)^{J1+J2+J4+j2+j4+jk+jn+Lambda} sqrt( (2J3+1) * (2J4 + 1) )
  ///                                                          (2j3+1) (2j4 +1) { jb j3 J2 } { jk jb J3 } { jn ja J4 }
  ///                                                                           { ja j4 J1 } { j4 j1 J1 } { j4 j2 J2 }
  ///                                                          { Lambda J4 J3 } X^(J1j3,J2j3)0_ijalmb Y^(J3,J4)Lambda_bkan 
  ///                                                          { j4     j1 j2 }
  ///                                           - sum_ab sum_{J3J4J5j3j4} (na-nb) PJ1J3(ij/k) PJ2J3(lm/n)
  ///                                                          (-1)^{J3+jk+jb} sqrt( (2j3+1) * (2j4 + 1) )
  ///                                                          (2J3+1) (2J4+1) (2J5+1) { J1 Lambda J4 } { J1 Lambda J4 } { J4 J2 J5 }
  ///                                                                                  { j4 ja     j3 } { j2 jk     j1 } { jb ja j4 }
  ///                                                            { jn jk J5 } { J5 J4 J2 }  Y^(J1j3,J2j4)Lambda_ijalmb X^(J3,J3)0_bkan 
  ///                                                            { jb ja J3 } { j2 jn jk } 
  ///                                                                                                    
  ///                                                                                                    
  ///
  void comm233_phst(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X2 = X.TwoBody;
    auto &Y2 = Y.TwoBody;
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z3 = Z.ThreeBody;
    int Lambda = Z.GetJRank();
    Z.modelspace->PreCalculateSixJ();

    // Permutations of indices which are needed to produce antisymmetrized matrix elements  P(ij/k) |ijk> = |ijk> - |kji> - |ikj>
    const std::array<ThreeBodyStorage::Permutation, 3> index_perms = {ThreeBodyStorage::ABC, ThreeBodyStorage::CBA, ThreeBodyStorage::ACB};
    double tstart = omp_get_wtime();
    std::vector<std::array<size_t, 3>> bra_ket_channels;
    for (auto &it : Z.ThreeBody.Get_ch_start())
    {
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(it.first.ch_bra);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      for (size_t ibra = 0; ibra < nbras3; ibra++)
      {
        bra_ket_channels.push_back({it.first.ch_bra, it.first.ch_ket, static_cast<size_t>(ibra)}); // (ch_bra, ch_ket,ibra)
      }
    }
    size_t n_bra_ket_ch = bra_ket_channels.size();

    #pragma omp parallel for schedule(dynamic,1) 
    for (size_t ibra_ket = 0; ibra_ket < n_bra_ket_ch; ibra_ket++)
    {
      size_t ch3bra = bra_ket_channels[ibra_ket][0];
      size_t ch3ket = bra_ket_channels[ibra_ket][1];
      size_t ibra = bra_ket_channels[ibra_ket][2];
      auto &Tbc_bra = Z.modelspace->GetThreeBodyChannel(ch3bra);
      auto &Tbc_ket = Z.modelspace->GetThreeBodyChannel(ch3ket);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      size_t nkets3 = Tbc_ket.GetNumberKets();
      // int twoJ = Tbc_bra.twoJ; // Scalar commutator so J is the same in bra and ket channel
      // double Jtot = 0.5 * twoJ;
      int twoj1 = Tbc_bra.twoJ; 
      int twoj2 = Tbc_ket.twoJ; 

      auto &bra = Tbc_bra.GetKet(ibra);
      size_t i = bra.p;
      size_t j = bra.q;
      size_t k = bra.r;
      Orbit &oi = Z.modelspace->GetOrbit(i);
      Orbit &oj = Z.modelspace->GetOrbit(j);
      Orbit &ok = Z.modelspace->GetOrbit(k);
      double ji = 0.5 * oi.j2;
      double jj = 0.5 * oj.j2;
      double jk = 0.5 * ok.j2;
      int J1 = bra.Jpq;
      size_t iket_max = nkets3;
      
      if (ch3bra == ch3ket)
        iket_max = ibra + 1;
      for (size_t iket = 0; iket < iket_max; iket++)
      {
        auto &ket = Tbc_ket.GetKet(iket);
        size_t l = ket.p;
        size_t m = ket.q;
        size_t n = ket.r;
        Orbit &ol = Z.modelspace->GetOrbit(l);
        Orbit &om = Z.modelspace->GetOrbit(m);
        Orbit &on = Z.modelspace->GetOrbit(n);
        double jl = 0.5 * ol.j2;
        double jm = 0.5 * om.j2;
        double jn = 0.5 * on.j2;
        int J2 = ket.Jpq;

        double zijklmn = 0;

        // Now we need to loop over the permutations in ijk and then lmn
        for (auto perm_ijk : index_perms) // {ijk} -> {123}
        {
          // if (perm_ijk != index_perms[0]) continue;
          size_t I1, I2, I3;
          Z3.Permute(perm_ijk, i, j, k, I1, I2, I3);
          Orbit &o1 = Z.modelspace->GetOrbit(I1);
          Orbit &o2 = Z.modelspace->GetOrbit(I2);
          Orbit &o3 = Z.modelspace->GetOrbit(I3);

          int J1p_min = J1;
          int J1p_max = J1;
          if (perm_ijk != ThreeBodyStorage::ABC)
          {
            J1p_min = std::max(std::abs(o1.j2 - o2.j2), std::abs(twoj1 - o3.j2)) / 2;
            J1p_max = std::min(o1.j2 + o2.j2, twoj1 + o3.j2) / 2;
          }

          for (int J1p = J1p_min; J1p <= J1p_max; J1p++)
          {
            double Pijk = Z3.PermutationPhase(perm_ijk) * Z3.RecouplingCoefficient(perm_ijk, ji, jj, jk, J1p, J1, twoj1);

            for (auto perm_lmn : index_perms) // {lmn} -> {456}
            {
              //  if (perm_lmn != index_perms[0]) continue;
              size_t I4, I5, I6;
              Z3.Permute(perm_lmn, l, m, n, I4, I5, I6);
              Orbit &o4 = Z.modelspace->GetOrbit(I4);
              Orbit &o5 = Z.modelspace->GetOrbit(I5);
              Orbit &o6 = Z.modelspace->GetOrbit(I6);

              int J2p_min = J2;
              int J2p_max = J2;
              if (perm_lmn != ThreeBodyStorage::ABC)
              {
                J2p_min = std::max(std::abs(o4.j2 - o5.j2), std::abs(twoj2 - o6.j2)) / 2;
                J2p_max = std::min(o4.j2 + o5.j2, twoj2 + o6.j2) / 2;
              }
              for (int J2p = J2p_min; J2p <= J2p_max; J2p++)
              {
                double Plmn = Z3.PermutationPhase(perm_lmn) * Z3.RecouplingCoefficient(perm_lmn, jl, jm, jn, J2p, J2, twoj2);

                for (size_t a : Z.modelspace->all_orbits)
                {
                  Orbit &oa = Z.modelspace->GetOrbit(a);
                  for (size_t b : Z.modelspace->all_orbits)
                  {
                    Orbit &ob = Z.modelspace->GetOrbit(b);

                    double occupation_factor = oa.occ - ob.occ;
                    if (std::abs(occupation_factor) < 1e-7)
                      continue;

                    if ((o3.l + ob.l + o6.l + oa.l + X.parity) % 2 > 0 and (o3.l + ob.l + o6.l + oa.l + Y.parity) % 2 > 0)
                      continue;
                    if (std::abs(o3.tz2 + ob.tz2 - o6.tz2 - oa.tz2) != X.GetTRank() and std::abs(o3.tz2 + ob.tz2 - o6.tz2 - oa.tz2) != Y.GetTRank())
                      continue;
                    int Jb3_min = std::abs(ob.j2 - o3.j2) / 2;
                    int Jb3_max = (ob.j2 + o3.j2) / 2;
                    for (int Jb3 = Jb3_min; Jb3 <= Jb3_max; Jb3++)
                    {
                      int twoj3_min = std::abs(oa.j2 - 2 * J1p);
                      int twoj3_max = oa.j2 + 2 * J1p;
                      for (int twoj3 = twoj3_min; twoj3 <= twoj3_max; twoj3 += 2)
                      {
                        int twoj4_min = std::abs(oa.j2 - 2 * J2p);
                        int twoj4_max = oa.j2 + 2 * J2p;
                        for (int twoj4 = twoj4_min; twoj4 <= twoj4_max; twoj4 += 2)
                        {
                          int Ja6_min = std::abs(oa.j2 - o6.j2) / 2;
                          int Ja6_max = (oa.j2 + o6.j2) / 2;
                          for (int Ja6 = Ja6_min; Ja6 <= Ja6_max; Ja6++)
                          {
                            double sixj  = AngMom::SixJ(ob.j2 / 2.,  twoj3 / 2.,    J2p,
                                                        oa.j2 / 2.,  twoj4 / 2.,    J1p);

                                  sixj *= AngMom::SixJ(o3.j2 / 2.,  ob.j2 / 2.,    Jb3,
                                                        twoj4 / 2.,  twoj1 / 2.,    J1p);

                                  sixj *= AngMom::SixJ(o6.j2 / 2.,  oa.j2 / 2.,    Ja6,
                                                        twoj4 / 2.,  twoj2 / 2.,    J2p);

                                  sixj *= AngMom::SixJ(Lambda,      Ja6,           Jb3,
                                                        twoj4 / 2.,  twoj1 / 2.,    twoj2 / 2.);
                            if (std::abs(sixj) < 1e-7)
                              continue;
                            int phasefactor = Z.modelspace->phase(( o3.j2  + o6.j2 + twoj2 + twoj4) / 2 + J1p + J2p + Ja6 + Lambda);
                            double hatfactor =  (twoj3 + 1) * (twoj4 + 1) * sqrt( (twoj1 + 1) * ( twoj2 + 1) * (2 * Ja6 + 1) * (2 * Jb3 + 1) );
                            double x12a45b = X3.GetME_pn(J1p, J2p, twoj3, I1, I2, a, I4, I5, b);
                            double yb3a6 = Y2.GetTBME_J(Jb3, Ja6, b, I3, a, I6);
                            zijklmn += occupation_factor * phasefactor * hatfactor * sixj * Pijk * Plmn * ( x12a45b * yb3a6 );
                          } // Ja6
                        } // twoj4


                        twoj4_min = std::abs(ob.j2 - 2 * J2p);
                        twoj4_max = ob.j2 + 2 * J2p;
                        for (int twoj4 = twoj4_min; twoj4 <= twoj4_max; twoj4 += 2)
                        {
                          int J3_min = std::abs(Lambda - J1p);
                          int J3_max = (Lambda + J1p);
                          for (int J3 = J3_min; J3 <= J3_max; J3++)
                          {
                            int J4_min = std::abs(J3 - J2p);
                            int J4_max = (J3 + J2p);
                            for (int J4 = J4_min; J4 <= J4_max; J4++)
                            {
                              double sixj  = AngMom::SixJ(J1p,         Lambda,        J3,
                                                          twoj4 / 2.,  oa.j2 / 2.,    twoj3 / 2.);

                                    sixj *= AngMom::SixJ( J1p,         Lambda,        J3,
                                                          twoj2 / 2.,  o3.j2 / 2.,    twoj1 / 2.);

                                    sixj *= AngMom::SixJ( J3,          J2p,           J4,
                                                          ob.j2 / 2.,  oa.j2 / 2.,    twoj4 / 2.);

                                    sixj *= AngMom::SixJ( o6.j2 / 2.,  o3.j2 / 2.,    J4,
                                                          ob.j2 / 2.,  oa.j2 / 2.,    Jb3);

                                    sixj *= AngMom::SixJ( J4,          J3,            J2p,
                                                          twoj2 / 2.,  o6.j2 / 2.,    o3.j2 / 2.);

                              if (std::abs(sixj) < 1e-7)
                                continue;
                              int phasefactor = Z.modelspace->phase(( o3.j2  + ob.j2 ) / 2 + Jb3 );
                              double hatfactor = (2 * Jb3 + 1) * (2 * J3 + 1) * (2 * J4 + 1) * sqrt( (twoj1 + 1) * ( twoj2 + 1) * (twoj3 + 1) * (twoj4 + 1) );
                              
                              double y12a45b = Y3.GetME_pn(J1p, twoj3, J2p, twoj4, I1, I2, a, I4, I5, b);
                              double xb3a6 = X2.GetTBME_J(Jb3, b, I3, a, I6);
                              zijklmn -= occupation_factor * phasefactor * hatfactor * sixj * Pijk * Plmn * ( y12a45b * xb3a6 );

                            } // J4
                          } // J3
                        } // twoj4

                      } // twoj3
                    } // Jb3

                  } // b
                } // a
              } // J2p
            } // perm_lmn
          } // J1p
        } // perm_ijk
        Z3.AddToME_pn_ch(ch3bra, ch3ket, ibra, iket, zijklmn); // this needs to be modified for beta decay
      } // for iket
        //    }//ibra
    } // ch
  
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm233_phst

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// Expression:    Z^(J1j1,J2j2)Lamda_ijklmn = 1/6 sum_abc sum_J3 (na nb nc + (1-na)(1-nb)(1-nc)) 
  ///                                              (  X^(J1j1,J3j1)0_ijkabc Y^(J3j1,J2j2)Lamda_abclmn 
  ///                                               - Y^(J1j1,J3j2)Lamda_ijkabc X^(J3j2,J2j2)0_abclmn )
  ///
  void comm333_ppp_hhhst(const Operator &X, const Operator &Y, Operator &Z)
  {
    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z3 = Z.ThreeBody;

    std::vector<std::array<size_t, 3>> bra_ket_channels;
    for (auto &it : Z.ThreeBody.Get_ch_start())
    {
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(it.first.ch_bra);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      for (size_t ibra = 0; ibra < nbras3; ibra++)
      {
        bra_ket_channels.push_back({it.first.ch_bra, it.first.ch_ket, static_cast<size_t>(ibra)}); // (ch_bra, ch_ket,ibra)
      }
    }
    size_t n_bra_ket_ch = bra_ket_channels.size();
    double tstart = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t ibra_ket = 0; ibra_ket < n_bra_ket_ch; ibra_ket++)
    {
      size_t ch3bra = bra_ket_channels[ibra_ket][0];
      size_t ch3ket = bra_ket_channels[ibra_ket][1];
      size_t ibra = bra_ket_channels[ibra_ket][2];
      auto &Tbc_bra = Z.modelspace->GetThreeBodyChannel(ch3bra);
      auto &Tbc_ket = Z.modelspace->GetThreeBodyChannel(ch3ket);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      size_t nkets3 = Tbc_ket.GetNumberKets();

      size_t iket_max = nkets3;
      if (ch3bra == ch3ket)
        iket_max = ibra + 1;
      for (size_t iket = 0; iket < iket_max; iket++)
      {
        double zijklmn = 0;

        for (size_t iket_abc = 0; iket_abc < nbras3; iket_abc++)
        {
          Ket3 &ket_abc = Tbc_bra.GetKet(iket_abc);
          double counting_factor = 6.0; // in general, 6 different permutations of abc
          if (ket_abc.p == ket_abc.r)
            counting_factor = 1.0; // if a=c, we must have a=b=c because we store a<=b<=c, and there is only one permutation
          else if ((ket_abc.p == ket_abc.q) or (ket_abc.q == ket_abc.r))
            counting_factor = 3; // if two orbits match, only 3 different permutations

          double nanbnc = ket_abc.op->occ * ket_abc.oq->occ * ket_abc.oR->occ;
          double nanbnc_bar = (1 - ket_abc.op->occ) * (1 - ket_abc.oq->occ) * (1 - ket_abc.oR->occ);

          double xijkabc = X3.GetME_pn_ch(ch3bra, ch3bra, ibra, iket_abc);
          double yabclmn = Y3.GetME_pn_ch(ch3bra, ch3ket, iket_abc, iket);

          zijklmn += 1. / 6 * counting_factor * (nanbnc + nanbnc_bar) * (xijkabc * yabclmn );
        } // for abc

        for (size_t iket_abc = 0; iket_abc < nkets3; iket_abc++)
        {
          Ket3 &ket_abc = Tbc_ket.GetKet(iket_abc);
          double counting_factor = 6.0; // in general, 6 different permutations of abc
          if (ket_abc.p == ket_abc.r)
            counting_factor = 1.0; // if a=c, we must have a=b=c because we store a<=b<=c, and there is only one permutation
          else if ((ket_abc.p == ket_abc.q) or (ket_abc.q == ket_abc.r))
            counting_factor = 3; // if two orbits match, only 3 different permutations

          double nanbnc = ket_abc.op->occ * ket_abc.oq->occ * ket_abc.oR->occ;
          double nanbnc_bar = (1 - ket_abc.op->occ) * (1 - ket_abc.oq->occ) * (1 - ket_abc.oR->occ);

          double yijkabc = Y3.GetME_pn_ch(ch3bra, ch3ket, ibra, iket_abc);
          double xabclmn = X3.GetME_pn_ch(ch3ket, ch3ket, iket_abc, iket);

          zijklmn -= 1. / 6 * counting_factor * (nanbnc + nanbnc_bar) * ( yijkabc * xabclmn);
        } // for abc


        Z3.AddToME_pn_ch(ch3bra, ch3ket, ibra, iket, zijklmn); // this needs to be modified for beta decay

      } // iket : lmn
      //    }//ibra : ijk
    } // chbra, chket
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;
  } // comm333_ppp_hhhst

  void comm333_pph_hhpst(const Operator &X, const Operator &Y, Operator &Z)
  {

    auto &X3 = X.ThreeBody;
    auto &Y3 = Y.ThreeBody;
    auto &Z3 = Z.ThreeBody;
    std::map<int, double> e_fermi = Z.modelspace->GetEFermi();
    double tstart = omp_get_wtime();
    Z.modelspace->PreCalculateSixJ();
    int parityY = Y.GetParity();
    int parityZ = parityY;
    int TzY = Y.GetTRank();
    int TzZ = TzY;
    int JrankZ = Y.GetJRank();
    int Lambda = Y.GetJRank();

    // Permutations of indices which are needed to produce antisymmetrized matrix elements  P(ij/k) |ijk> = |ijk> - |kji> - |ikj>
    const std::array<ThreeBodyStorage::Permutation, 3> index_perms = {ThreeBodyStorage::ABC, ThreeBodyStorage::CBA, ThreeBodyStorage::ACB};

    std::vector<std::array<size_t, 3>> bra_ket_channels;
    for (auto &it : Z.ThreeBody.Get_ch_start())
    {
      ThreeBodyChannel &Tbc_bra = Z.modelspace->GetThreeBodyChannel(it.first.ch_bra);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      for (size_t ibra = 0; ibra < nbras3; ibra++)
      {
        bra_ket_channels.push_back({it.first.ch_bra, it.first.ch_ket, static_cast<size_t>(ibra)}); // (ch_bra, ch_ket,ibra)
      }
    }
    size_t n_bra_ket_ch = bra_ket_channels.size();

    size_t nch2 = Z.modelspace->GetNumberTwoBodyChannels();
    size_t nch3 = Z.modelspace->GetNumberThreeBodyChannels();


    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t ibra_ket = 0; ibra_ket < n_bra_ket_ch; ibra_ket++)
    {
      size_t ch3bra = bra_ket_channels[ibra_ket][0];
      size_t ch3ket = bra_ket_channels[ibra_ket][1];
      size_t ibra = bra_ket_channels[ibra_ket][2];
      auto &Tbc_bra = Z.modelspace->GetThreeBodyChannel(ch3bra);
      auto &Tbc_ket = Z.modelspace->GetThreeBodyChannel(ch3ket);
      size_t nbras3 = Tbc_bra.GetNumberKets();
      size_t nkets3 = Tbc_ket.GetNumberKets();

      int twoj1 = Tbc_bra.twoJ; 
      int twoj2 = Tbc_ket.twoJ; 

      auto &bra = Tbc_bra.GetKet(ibra);
      size_t i = bra.p;
      size_t j = bra.q;
      size_t k = bra.r;
      Orbit &oi = Z.modelspace->GetOrbit(i);
      Orbit &oj = Z.modelspace->GetOrbit(j);
      Orbit &ok = Z.modelspace->GetOrbit(k);
      double ji = 0.5 * oi.j2;
      double jj = 0.5 * oj.j2;
      double jk = 0.5 * ok.j2;
      int J1 = bra.Jpq;

      size_t iket_max = nkets3;
      if (ch3bra == ch3ket)
        iket_max = ibra + 1;
      for (size_t iket = 0; iket < iket_max; iket++)
      {
        auto &ket = Tbc_ket.GetKet(iket);
        size_t l = ket.p;
        size_t m = ket.q;
        size_t n = ket.r;
        Orbit &ol = Z.modelspace->GetOrbit(l);
        Orbit &om = Z.modelspace->GetOrbit(m);
        Orbit &on = Z.modelspace->GetOrbit(n);
        double jl = 0.5 * ol.j2;
        double jm = 0.5 * om.j2;
        double jn = 0.5 * on.j2;
        int J2 = ket.Jpq;

        double zijklmn = 0;
        // Now we need to loop over the permutations in ijk and then lmn
        for (auto perm_ijk : index_perms) // {ijk} -> {123}
        {
          // if (perm_ijk != index_perms[0]) continue;
          size_t I1, I2, I3;
          Z3.Permute(perm_ijk, i, j, k, I1, I2, I3);
          Orbit &o1 = Z.modelspace->GetOrbit(I1);
          Orbit &o2 = Z.modelspace->GetOrbit(I2);
          Orbit &o3 = Z.modelspace->GetOrbit(I3);

          int J1p_min = J1;
          int J1p_max = J1;
          if (perm_ijk != ThreeBodyStorage::ABC)
          {
            J1p_min = std::max(std::abs(o1.j2 - o2.j2), std::abs(twoj1 - o3.j2)) / 2;
            J1p_max = std::min(o1.j2 + o2.j2, twoj1 + o3.j2) / 2;
          }
          for (int J1p = J1p_min; J1p <= J1p_max; J1p++)
          {
            double Pijk = Z3.PermutationPhase(perm_ijk) * Z3.RecouplingCoefficient(perm_ijk, ji, jj, jk, J1p, J1, twoj1);

            for (auto perm_lmn : index_perms) // {lmn} -> {456}
            {
              //  if (perm_lmn != index_perms[0]) continue;
              size_t I4, I5, I6;
              Z3.Permute(perm_lmn, l, m, n, I4, I5, I6);
              Orbit &o4 = Z.modelspace->GetOrbit(I4);
              Orbit &o5 = Z.modelspace->GetOrbit(I5);
              Orbit &o6 = Z.modelspace->GetOrbit(I6);

              int J2p_min = J2;
              int J2p_max = J2;
              if (perm_lmn != ThreeBodyStorage::ABC)
              {
                J2p_min = std::max(std::abs(o4.j2 - o5.j2), std::abs(twoj2 - o6.j2)) / 2;
                J2p_max = std::min(o4.j2 + o5.j2, twoj2 + o6.j2) / 2;
              }
              for (int J2p = J2p_min; J2p <= J2p_max; J2p++)
              {
                double Plmn = Z3.PermutationPhase(perm_lmn) * Z3.RecouplingCoefficient(perm_lmn, jl, jm, jn, J2p, J2, twoj2);

                for (size_t ch2 = 0; ch2 < nch2; ch2++)
                {
                  auto &tbc_ab = Z.modelspace->GetTwoBodyChannel(ch2);

                  //if (std::abs(Tbc.twoTz - 2 * tbc_ab.Tz) == 5)
                  //  continue; // TODO there are probably other checks at the channel level...
                  size_t nkets_ab = tbc_ab.GetNumberKets();
                  int Jab = tbc_ab.J;
                  for (size_t iket_ab = 0; iket_ab < nkets_ab; iket_ab++)
                  {
                    Ket &ket_ab = tbc_ab.GetKet(iket_ab);
                    size_t a = ket_ab.p;
                    size_t b = ket_ab.q;
                    Orbit &oa = Z.modelspace->GetOrbit(a);
                    Orbit &ob = Z.modelspace->GetOrbit(b);
                    if (std::abs(oa.occ * ob.occ) < 1e-6 and std::abs((1 - oa.occ) * (1 - ob.occ)) < 1e-6)
                      continue;

                    for (auto c : Z.modelspace->all_orbits)
                    {
                      Orbit &oc = Z.modelspace->GetOrbit(c);
                      double occ_factor = oa.occ * ob.occ * (1 - oc.occ) + (1 - oa.occ) * (1 - ob.occ) * oc.occ;
                      if (std::abs(occ_factor) < 1e-6)
                        continue;
                      if (a == b)
                        occ_factor *= 0.5; // because we only sum b<a
                      double jc = 0.5 * oc.j2;

                      if ( (    ((oa.l + ob.l + o3.l + o4.l + o5.l + oc.l + parityZ) % 2 == 0) 
                           and ((oa.tz2 + ob.tz2 + o3.tz2 - o4.tz2 - o5.tz2 - oc.tz2) == 2 * TzZ ) )
                           or
                           (   ((o1.l + o2.l + oc.l + oa.l + ob.l + o6.l + parityZ) % 2 == 0) 
                           and ((o1.tz2 + o2.tz2 + oc.tz2 - oa.tz2 - ob.tz2 - o6.tz2) == 2 * TzZ)))
                      {
                        int twoj3_min = std::abs(oc.j2 - 2 * J1p); // J1 + jc
                        int twoj3_max = oc.j2 + 2 * J1p;
                        for (int twoj3 = twoj3_min; twoj3 <= twoj3_max; twoj3 += 2)
                        {
                          int twoj4_min = std::abs(o3.j2 - 2 * Jab); // Jab + jk
                          int twoj4_max = o3.j2 + 2 * Jab;
                          for (int twoj4 = twoj4_min; twoj4 <= twoj4_max; twoj4 += 2)
                          {
                            int twoj5_min = std::abs(oc.j2 - 2 * J2p); // J2 + jc
                            int twoj5_max = oc.j2 + 2 * J2p;
                            for (int twoj5 = twoj5_min; twoj5 <= twoj5_max; twoj5 += 2)
                            {
                              int J3_min = std::abs(J1p - Jab);
                              int J3_max = (J1p + Jab);
                              for (int J3 = J3_min; J3 <= J3_max; J3++)
                              {
                                double sixj  = AngMom::SixJ(J1p,         Jab,          J3,
                                                            o6.j2 / 2.,  oc.j2 / 2.,   twoj3 / 2.);

                                       sixj *= AngMom::SixJ(J1p,         Jab,          J3,
                                                            twoj4 / 2.,  twoj1 / 2.,   o3.j2 / 2.);

                                       sixj *= AngMom::SixJ(o6.j2 / 2.,  oc.j2 / 2.,   J3,
                                                            twoj5 / 2.,  twoj2 / 2.,   J2p);

                                       sixj *= AngMom::SixJ(twoj4 / 2.,  twoj5 / 2.,   Lambda,
                                                            twoj2 / 2.,  twoj1 / 2.,   J3);
                                if (std::abs(sixj) < 1e-7)
                                  continue;
                                int phasefactor = Z.modelspace->phase(( o3.j2  + o6.j2 + oc.j2 + twoj3) / 2 + J1p + J2p + Jab + J3 + Lambda);
                                double hatfactor =  (twoj3 + 1) * (2 * J3 + 1) * sqrt( (twoj1 + 1) * ( twoj2 + 1) * ( twoj4 + 1) * (twoj5 + 1) );

                                double x12cab6 = X3.GetME_pn(J1p, Jab, twoj3, I1, I2, c, a, b, I6);
                                double yab345c = Y3.GetME_pn(Jab, twoj4, J2p, twoj5, a, b, I3, I4, I5, c);
                                zijklmn -= occ_factor * phasefactor * hatfactor * sixj * Pijk * Plmn * ( x12cab6 * yab345c );

                              } // J3
                            } // twoj5
                          } // twoj4


                          //------------------------------------------------------------------------
                          twoj4_min = std::abs(o6.j2 - 2 * Jab); // Jab + jn
                          twoj4_max = o6.j2 + 2 * Jab;
                          for (int twoj4 = twoj4_min; twoj4 <= twoj4_max; twoj4 += 2)
                          {
                            int twoj5_min = std::abs(o3.j2 - 2 * Jab); // Jab + jk
                            int twoj5_max = o3.j2 + 2 * Jab;
                            for (int twoj5 = twoj5_min; twoj5 <= twoj5_max; twoj5 += 2)
                            {
                              int twoj6_min = std::abs(oc.j2 - 2 * Jab); // Jab + jc
                              int twoj6_max = oc.j2 + 2 * Jab;
                              for (int twoj6 = twoj6_min; twoj6 <= twoj6_max; twoj6 += 2)
                              {
                                int J3_min = std::abs(J1p - Lambda);
                                int J3_max = (J1p + Lambda);
                                for (int J3 = J3_min; J3 <= J3_max; J3++)
                                {
                                  double sixj  = AngMom::SixJ(J1p,         Lambda,          J3,
                                                              twoj4 / 2.,  oc.j2 / 2.,      twoj3 / 2.);

                                        sixj *= AngMom::SixJ(J1p,         Lambda,          J3,
                                                             twoj2 / 2.,  o3.j2 / 2.,      twoj1 / 2.);

                                        sixj *= AngMom::SixJ(oc.j2 / 2.,  twoj4 / 2.,   J3,
                                                             o6.j2 / 2.,  twoj6 / 2.,   Jab);

                                        sixj *= AngMom::SixJ(oc.j2 / 2.,  twoj5 / 2.,   J2p,
                                                             o3.j2 / 2.,  twoj6 / 2.,   Jab);

                                        sixj *= AngMom::SixJ(o6.j2 / 2.,  twoj6 / 2.,   J3,
                                                             o3.j2 / 2.,  twoj2 / 2.,   J2p);

                                  if (std::abs(sixj) < 1e-7)
                                    continue;
                                  int phasefactor = Z.modelspace->phase(( o3.j2  + twoj4 + oc.j2 + twoj2) / 2 );
                                  double hatfactor =  (twoj5 + 1) * (twoj6 + 1) * (2 * J3 + 1) * sqrt( (twoj1 + 1) * ( twoj2 + 1) * ( twoj3 + 1) * (twoj4 + 1) );

                                  double xab345c = X3.GetME_pn(Jab, J2p, twoj5, a, b, I3, I4, I5, c);
                                  double y12cab6 = Y3.GetME_pn(J1p, twoj3, Jab, twoj4, I1, I2, c, a, b, I6);
                                  zijklmn -= occ_factor * phasefactor * hatfactor * sixj * Pijk * Plmn * ( xab345c * y12cab6 );
                                } // J3
                              
                              }// twoj6
                            } // twoj5
                          } // twoj4
                        } // twoj3
                      }

                    } // for c
                  } // for iket_ab
                } // for ch2

              } // J2p
            } // perm_lmn          
          } // J1p
        } // perm_ijk


        Z3.AddToME_pn_ch(ch3bra, ch3ket, ibra, iket, zijklmn);
      } // iket : lmn
    } // chbra, chket //ibra : ijk
  
    X.profiler.timer[__func__] += omp_get_wtime() - tstart;  
  
  } // comm333_pph_hhpst

}// namespace Commutator
