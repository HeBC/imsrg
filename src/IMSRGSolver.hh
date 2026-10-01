///////////////////////////////////////////////////////////////////////////////////
//    IMSRGSolver.hh, part of  imsrg++
//    Copyright (C) 2018  Ragnar Stroberg
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

#ifndef IMSRGSolver_h
#define IMSRGSolver_h 1

#include <fstream>
#include <string>
#include <deque>
#include <vector>
#include "Operator.hh"
#include "Generator.hh"
#include "IMSRGProfiler.hh"
#include "ReadWrite.hh"

//using namespace std;


class IMSRGSolver
{

  public:

//  private:
  ModelSpace* modelspace;
//  ReadWrite* rw;
  Operator* H_0; 
  std::deque<Operator> FlowingOps;
  Operator H_saved;
  Operator Eta;
  std::deque<Operator> Omega;
  Generator generator;
  int istep;
  double s;
  double ds;
  double ds_max;
  double smax;
  double norm_domega;
  double omega_norm_max;
  double eta_criterion;
  std::string method;
  std::string flowfile;
  std::string scratchdir;
  IMSRGProfiler profiler;
  int n_omega_written;
  int max_omega_written;
  bool magnus_adaptive;
  bool hunter_gatherer;
  bool perturbative_triples;

  // Opt-in: cache rank-two sources for returns across exponential boundaries.
  // The original option retains one gatherer plus hunter; two gatherers is separate.
  bool use_two_omega_bch = false;
  // Two large gatherers plus the small hunter: [G1,h] -> [G1,G2,h].
  // G1 is capped by omega_norm_max and freezes when G2 is first created.
  bool use_two_gatherers = false;
  double hunter_norm_max = 0.1;
  bool first_gatherer_source_valid = false;
  Operator first_gatherer_saved;
  Operator first_gatherer_source;
  bool gatherer_source_valid = false;
  Operator gatherer_source;
  std::vector<double> gatherer_source_settings;
  void UpdateGathererCache();
  void SetUseTwoOmegaBCH(bool b=true);
  bool GetUseTwoOmegaBCH() const { return use_two_omega_bch; }
  void SetUseTwoGatherers(bool b=true);
  bool GetUseTwoGatherers() const { return use_two_gatherers; }
  void SetHunterNormMax(double x); // Two-gatherer mode only; zero gathers every step.
  double GetHunterNormMax() const { return hunter_norm_max; }
  double EffectiveHunterNormMax() const { return use_two_gatherers ? hunter_norm_max : omega_norm_max; }
  void InvalidateGathererCache(bool first=true);
  void ValidateGathererState() const;

  // Ordinary Magnus splitting with retained 223 sources across ALL segments.
  // No gatherer is merged; SetOmegaNormMax remains the split threshold.
  bool use_factorized_split_bch = false;
  bool split_cache_valid = false;
  size_t split_cache_count = 0;
  std::deque<Operator> split_sources; // new Htilde_j only, never cumulative
  std::vector<double> split_cache_settings;
  void SetUseFactorizedSplitBCH(bool b=true);
  bool GetUseFactorizedSplitBCH() const { return use_factorized_split_bch; }
  size_t GetSplitSourceCount() const { return split_sources.size(); }
  void InvalidateSplitCache();
  void UpdateSplitCache();
  void FreezeSplitOmega();
  std::string SplitOmegaFilename(size_t i) const;
  const Operator &ReadSplitOmega(size_t i, Operator &buffer) const;
  void WriteSplitOmega(size_t i, Operator &op) const;
  std::pair<Operator, Operator> ApplySplitStage(const Operator &H_in, const Operator &Omega_i,
      const std::deque<Operator> &sources, int first, int direction, bool collect_source) const;
  Operator TransformSplit(const Operator &OpIn, int first, bool inverse=false) const;

  double Elast;
  double cumulative_error;
//  double pert_triples_this_omega;
//  double pert_triples_sum;
  
  // Per step, ds may not grow more than 1.2 times its previous value.
  double ds_max_growth_factor_ = 1.2;
  // When abs(E_MP2(s + ds)) > abs(E_MP2(s)), ds will "back off" by this factor.
  double ds_backoff_factor_ = 0.5;
  // Flag to signal when we are in the soft landing phase.
  // Soft landing means no more growth, only backoff
  bool in_soft_landing_phase_ = false;
  // We need the original ds_0 for when we change the generator
  double ds_0 = ds;


  ~IMSRGSolver();
  IMSRGSolver();
  IMSRGSolver( Operator& H_in);
  void NewOmega();
  void GatherOmega(); // hunter-gatherer mode of updating omega
  void SetHin( Operator& H_in);
//  void SetReadWrite( ReadWrite& r){rw = &r; scratchdir = rw->GetScratchDir();};
  void SetScratchDir(std::string sdir);
  std::string GetScratchDir( ) {return scratchdir; };
  void SetReadWrite( ReadWrite& r){SetScratchDir(r.GetScratchDir());}; // for backwards compatibility
  void Reset();
  void AddOperator(Operator& Op){FlowingOps.push_back(Op);};
  Operator GetOperator(size_t i){return FlowingOps.at(i);};
  void UpdateEta(); // Force eta to be calculated. For debugging.

  void SetMethod(std::string m){method=m;};
  void Solve();
  void Solve_magnus_euler();
  void Solve_magnus_backoff();
  void Solve_magnus_modified_euler();
  void Solve_flow_RK4();

  Operator Transform(Operator& OpIn);
  Operator Transform(Operator&& OpIn);
  Operator InverseTransform(Operator& OpIn);
  Operator GetOmega(int i){return Omega[i];};
  std::deque<Operator>& GetOmega() { InvalidateGathererCache(); InvalidateSplitCache(); return Omega; }
  void SetOmega(size_t i, Operator& om);
  size_t GetOmegaSize(){return Omega.size();};
  int GetNOmegaWritten(){return n_omega_written;};
  Operator Transform_Partial(Operator& OpIn, int n);
  Operator Transform_Partial(Operator&& OpIn, int n);

  void SetFlowFile(std::string s);
  void SetDs(double d){ds = d; ds_0 = d;};
  void SetDsmax(double d);
  void SetdOmega(double d){norm_domega = d;};
  void SetSmax(double d){smax = d;};
  void SetGenerator(std::string g);
  void SetDenominatorPartitioning(std::string dp);
  void SetOmegaNormMax(double x);
  void SetODETolerance(float x){ode_e_abs=x;ode_e_rel=x;};
  void SetEtaCriterion(float x){eta_criterion = x;};
  void SetMagnusAdaptive(bool b=true){magnus_adaptive = b;};
  void SetHunterGatherer(bool b=true);
  void SetPerturbativeTriples(bool b=true){perturbative_triples = b;};

  int GetSystemDimension();
  double GetS(){return s;};
  Operator& GetH_s(){return FlowingOps[0];};
  void SetH_s( Operator& Hset){ FlowingOps[0] = Hset;};
  Operator& GetEta(){return Eta;};
  Generator& GetGenerator(){return generator;};

  void UpdateOmega();
  void UpdateH();

  void WriteFlowStatus(std::ostream&);
  void WriteFlowStatusHeader(std::ostream&);
  void WriteFlowStatus(std::string);
  void WriteFlowStatusHeader(std::string);

  void SetDenominatorCutoff(double c){generator.SetDenominatorCutoff(c);};
  void SetDenominatorDelta(double d){generator.SetDenominatorDelta(d);};
  void SetDenominatorDeltaIndex(int i){generator.SetDenominatorDeltaIndex(i);};
  void SetDenominatorDeltaOrbit(std::string o){generator.SetDenominatorDeltaOrbit(o);};

  void FlushOmegaToScratch();
  void CleanupScratch();

  double EstimateStepError();
  double EstimateBCHError( );

//  double GetPerturbativeTriples();
  double CalculatePerturbativeTriples();
  double CalculatePerturbativeTriples(Operator &Op_0);

  // This is used to get flow info from odeint
  class ODE_Monitor
  {
    public:
     ODE_Monitor(IMSRGSolver& solver)
           : imsrgsolver(solver), times(solver.times), E0(solver.E0),
              eta1(solver.eta1),eta2(solver.eta2) {};
     IMSRGSolver& imsrgsolver;
     std::vector<double>& times;
     std::vector<double>& E0;
     std::vector<double>& eta1;
     std::vector<double>& eta2;
    //     void operator() (const vector<Operator>& x, double t)
     void operator() (const std::deque<Operator>& x, double t)
     {
        times.push_back(t);
        E0.push_back(x.front().ZeroBody);
        eta1.push_back(imsrgsolver.Eta.OneBodyNorm());
        eta2.push_back(imsrgsolver.Eta.TwoBodyNorm());
     }
     void report()
     {
        for (size_t i=0; i<times.size(); ++i)
        {
           std::cout << times[i] << "  " << E0[i] << "  "
                     << eta1[i]  << "  " << eta2[i] << std::endl;
        }
     }
  };


  ODE_Monitor ode_monitor;

  std::vector<double> times;
  std::vector<double> E0;
  std::vector<double> eta1;
  std::vector<double> eta2;
  std::string ode_mode;
  float ode_e_abs;
  float ode_e_rel;


//  void operator()( const vector<Operator>& x, vector<Operator>& dxdt, const double t);
  void operator()( const std::deque<Operator>& x, std::deque<Operator>& dxdt, const double t);
//  void Solve_ode();
  void Solve_flow_euler();
  void Solve_ode_adaptive();
  void Solve_ode_magnus();


};





#endif

