#include "M0nu.hh"
#include "AngMom.hh"
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_laguerre.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>

using namespace PhysConst;

/// Namespace for functions related to neutrinoless double beta decay
namespace M0nu
{

  /// Converts (a,b,c,d) in base (maxa+1,maxb+1,maxc+1,maxd+1) to an ordered decimal integer
  /// eg: for (maxa,maxb,maxc,maxd) = (1,1,1,1) decimalgen would convert (0,1,1,0) to 6, ie: a binary to decimal converter
  /// NOTE: to make comparisons between such decimals, keep (maxa,maxb,maxc,maxd) consistent
  /// PS: I tots made this thing up, did some testing and it seemed to work...
  /// ... hopefully it's good, but could be a source of weird bugs if anything comes up with missing integrals wrt dq, see GetIntegral(...)
  int decimalgen(int a, int b, int c, int d, int maxb, int maxc, int maxd)
  {
    int coeff1 = maxd + 1;
    int coeff2 = (maxc + 1)*coeff1;
    int coeff3 = (maxb + 1)*coeff2;
    return coeff3*a + coeff2*b + coeff1*c + d; // eg: (0,1,1,0) -> (2*2*2)*0 + (2*2)*1 + (2)*1 + 0 = 6
  }

  //Return the phase due to a certain quantity, i.e. (-1)^x
  int phase(int x)
  {
    return x%2==0 ? 1 : -1;
  }

  //Radial wave function written in momentum space
  double HO_Radial_psi_mom(int n, int l, double hw, double p)
  {
     double b = sqrt( (HBARC*HBARC) / (hw * M_NUCLEON*0.5));
     double x = p*b;
     double Norm = 2*sqrt( gsl_sf_fact(n) * pow(2,n+l) / SQRTPI / gsl_sf_doublefact(2*n+2*l+1) * pow(b,3.0) );
     double L = gsl_sf_laguerre_n(n,l+0.5,x*x);
     double psi = phase(n)*Norm * pow(x,l) * exp(-x*x*0.5) * L;
     return psi;
  }



  //The following four functions are used in the defintions of the neutrino potentials
  //They are implemented from Equation (4.14) of Charlie Payne's thesis. 
  //Note that they take the momentum square as arquments.
  double gv_func(double qsq)
  {
    double gV = NUCLEON_VECTOR_G/pow((1.0 + (qsq/(CUTV*CUTV))),2); // from Equation (4.14) of my thesis
    return gV;
  }

  double ga_func(double qsq)
  {
    double gA = NUCLEON_AXIAL_G/pow((1.0 + (qsq/(CUTA*CUTA))),2);
    return gA;
  }

  double gp_func(double qsq)
  {
    double gP = (2*M_PROTON*ga_func(qsq))/(qsq + M_PION_CHARGED*M_PION_CHARGED);
    return gP;
  }

  double gm_func(double qsq)
  {
    double gM = MAGMOM*gv_func(qsq);
    return gM;
  }

  /// Different form component required for the form factors
  /// separated by their vector, vector-axial, induced pseudo-scalar...
  /// parts as defined in JHEP12(2018)097

  double hF_VV(double qsq)
  {
    return gv_func(qsq)*gv_func(qsq)/(NUCLEON_VECTOR_G*NUCLEON_VECTOR_G);
  }

  double hGT_AA(double qsq)
  {
    return ga_func(qsq)*ga_func(qsq) / (NUCLEON_AXIAL_G * NUCLEON_AXIAL_G);
  }

  double hGT_AP(double qsq)
  {
    double gA = ga_func(qsq);
    double gP = gp_func(qsq);
    return -(gA * gP * qsq) / (3 * M_NUCLEON * NUCLEON_AXIAL_G * NUCLEON_AXIAL_G);
  }

  double hGT_PP(double qsq)
  {
    double Mnucsq = M_NUCLEON * M_NUCLEON; // the proton mass squared [MeV^2]
    double gP = gp_func(qsq);
    return (pow(gP * qsq, 2) / (12 * Mnucsq * NUCLEON_AXIAL_G * NUCLEON_AXIAL_G));
  }

  double hGT_MM(double qsq)
  {
    double Mnucsq = M_NUCLEON * M_NUCLEON; // the proton mass squared [MeV^2]
    double gM = gm_func(qsq);
    return ((gM * gM * qsq) / (6 * Mnucsq* NUCLEON_AXIAL_G * NUCLEON_AXIAL_G));
  }

  double hT_AA(double qsq)
  {
    return ga_func(qsq) * ga_func(qsq) / (NUCLEON_AXIAL_G * NUCLEON_AXIAL_G);
  }

  double hT_AP(double qsq)
  {
    double gA = ga_func(qsq);
    double gP = gp_func(qsq);
    return (gA * gP * qsq) / (3 * M_NUCLEON * NUCLEON_AXIAL_G * NUCLEON_AXIAL_G);
  }

  double hT_PP(double qsq)
  {
    double Mnucsq = M_NUCLEON * M_NUCLEON; // the proton mass squared [MeV^2]
    double gP = gp_func(qsq);
    return -(pow(gP * qsq, 2) / (12 * Mnucsq * NUCLEON_AXIAL_G * NUCLEON_AXIAL_G));
  }

  double hT_MM(double qsq)
  {
    double Mnucsq = M_NUCLEON * M_NUCLEON; // the proton mass squared [MeV^2]
    double gM = gm_func(qsq);
    return ((gM * gM * qsq) / (12 * Mnucsq * NUCLEON_AXIAL_G * NUCLEON_AXIAL_G));
  }

  /// Form factors of the neutrino potential of Gamow-Teller transition
  /// Implemented from Equation (4.12) of Charlie Payne's thesis
  double GTFormFactor(double q)
  {
    double qsq     = q*q; // q squared [MeV^2]
    // double Mprosq  = M_PROTON*M_PROTON; // the proton mass squared [MeV^2]
    // double gA      = ga_func(qsq);
    // double gP      = gp_func(qsq); 
    // double gM      = gm_func(qsq); 
    // double ff      = ((gA*gA) - ((gA*gP*qsq)/(3*M_PROTON)) + (pow(gP*qsq,2)/(12*Mprosq)) + ((gM*gM*qsq)/(6*Mprosq)))/(NUCLEON_AXIAL_G*NUCLEON_AXIAL_G);
    double ff = hGT_AA(qsq)+hGT_AP(qsq)+hGT_PP(qsq)+hGT_MM(qsq); 
    return ff;
  }

  /// Form factors of the neutrino potential of Gamow-Teller transition
  /// Implemented from Equation (4.11) of Charlie Payne's thesis
  double FermiFormFactor(double q)
  {
    double qsq     = q*q; // q squared [MeV^2]
    // double gV      = gv_func(qsq);
    // double ff      = (gV*gV)/(NUCLEON_VECTOR_G*NUCLEON_VECTOR_G); //Equation (4.11) of Charlie's thesis
    double ff = hF_VV(qsq) ;
    return ff;
  }

  /// Form factors of the neutrino potential of Gamow-Teller transition
  /// Implemented from Equation (4.13) of Charlie Payne's thesis
  double TensorFormFactor(double q)
  {
    double qsq     = q*q; // q squared [MeV^2]
    // double Mprosq  = M_PROTON*M_PROTON; // the proton mass squared [MeV^2]
    // double gA      = ga_func(qsq);
    // double gP      = gp_func (qsq);
    // double gM      = gm_func(qsq);
    // double ff      = (((gA*gP*qsq)/(3*M_PROTON)) - (pow(gP*qsq,2)/(12*Mprosq)) + ((gM*gM*qsq)/(12*Mprosq)))/(NUCLEON_AXIAL_G*NUCLEON_AXIAL_G); //Equation (4.13) of Charlie's thesis
    double ff = hT_AP(qsq)+hT_PP(qsq)+hT_MM(qsq);
    return -ff;
  }


  double RegulatorNonLocal(double p, double pp)
  {
    double Lambda = 394;
    double nexp = 4;
    return exp(-pow(p*HBARC/Lambda,2*nexp))*exp(-pow(pp*HBARC/Lambda,2*nexp));
    // return  1;
  }

  double RegulatorLocal(double q)
  {
    double Lambda = 394;
    double nexp = 4;
    return exp(-pow(q*HBARC/Lambda,2*nexp));
  }

  //Abbreviation used in the definition of the partial waves expansion in
  //K. Erkelenz, R. Alzetta, and K. Holinde, Nucl. Phys. A 176, 413 (1971).
  //The factor of z^l from the paper is ignored since l=0 in all the needed cases.
  double A(double p, double pp, int J, double Eclosure, std::string transition, gsl_integration_glfixed_table * t, int norm, int size)
  {
    double A = 0;
    std::map<std::string, std::function<double(double)> > FFlist = {{"F",FermiFormFactor},{"GT",GTFormFactor},{"T",TensorFormFactor},{"C", FermiFormFactor}};
    // std::map<std::string, std::function<double(double)> > FFlist = {{"F",FermiFormFactor},{"GT",GTFormFactor},{"T",TensorFormFactor}}; // make a map from std::string to a function which takes in q and returns a form factor
    for (int i=0;i<size;i++)
    {
      double xi;
      double wi;
      double temp;
      gsl_integration_glfixed_point(-1,1,i,&xi,&wi,t);
      double q = sqrt(pp*pp+p*p-2*p*pp*xi);
      double ff = FFlist[transition](q);
      if (transition!="C")
      {
        temp = wi*ff/(q*q+q*Eclosure)*gsl_sf_legendre_Pl(J,xi);
      }
      else
      {
        temp = wi*ff*gsl_sf_legendre_Pl(J,xi);
      }
      if (norm == 1)
      {
        temp *= HBARC*HBARC/(q*q);
      }
      A += temp;
    }
    return A*PI*HBARC*HBARC;//HBARC*HBARC factor to make A in fm^2
  }

  //Following functions are for the precalculation and the caching
  //of the A function for the partial wave decomposition
  //for the p values later used in the quadrature
  uint64_t AHash(int index_p,int index_pp, int J, int norm)
  { 
     return   (((uint64_t)(index_p)) << 30)
            + (((uint64_t)(index_pp)) << 20)
            + (((uint64_t)(J)) << 10)
            + ((uint64_t)(norm));
  }

  void AUnHash(uint64_t key, uint64_t& index_p, uint64_t& index_pp, uint64_t& J, uint64_t& norm)
  {
     index_p  = (key >> 30) & 0x3FFL;
     index_pp = (key >> 20) & 0x3FFL;
     J        = (key >> 10) & 0x3FFL;
     norm     = (key      ) & 0x3FFL;
  }

  //Precomputes the values of the A functions  and caches them for efficiency
  std::unordered_map<uint64_t,double> PrecalculateA(int e2max,double Eclosure, std::string transition, int npoints)
  {
    std::unordered_map<uint64_t,double> AList;
    int Jmax = e2max+1;
    std::vector<uint64_t> KEYS;
    int maxnorm = 0;
    if (transition=="T")
    {
      maxnorm = 1;
    }
    for (int norm = 0; norm <= maxnorm; norm++)
    {
        for (int J=0; J<=Jmax;J++)
        {
          for (int index_p = 0 ; index_p< npoints; index_p++)
          {
            for (int index_pp = index_p; index_pp<npoints; index_pp++)
            {
              uint64_t key = AHash(index_p,index_pp,J,norm);
              KEYS.push_back(key);
              AList[key] = 0.0; // "Make sure eveything's in there to avoid a rehash in the parallel loop" (RS)
            }//end of for loop over j 
          }//end of for loop over i
        }//end of for loop over J
    }//end of for loop over norm
    gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc(npoints);
    int size = 100;
    gsl_integration_glfixed_table * t2 = gsl_integration_glfixed_table_alloc(size);
    #pragma omp parallel for schedule(dynamic, 1)// this works as long as the gsl_function handle is within this for-loop
    for (size_t n=0; n<KEYS.size(); n++)
    {
      uint64_t key = KEYS[n];
      uint64_t index_p,index_pp,J,norm;
      AUnHash(key, index_p,index_pp,J,norm);
      double p,pp,wi,wj;
      gsl_integration_glfixed_point(0,25,index_p,&p,&wi,t);
      gsl_integration_glfixed_point(0,25,index_pp,&pp,&wj,t);
      AList[key] = A(p*HBARC,pp*HBARC,J,Eclosure,transition,t2,norm, size); 
    }
    gsl_integration_glfixed_table_free(t);
    gsl_integration_glfixed_table_free(t2);
    return AList;
  }

  //Fetch the cached value of a specific A functions 
  double GetA(int index_p, int index_pp, int J,int norm, std::unordered_map<uint64_t,double> &AList)
  {
    double A;
    if (index_p >index_pp) std::swap(index_p,index_pp);
    uint64_t key = AHash(index_p,index_pp,J,norm);
    auto it = AList.find(key);
    if (it != AList.end()) // return what we've found
    {
      A =it -> second;
    }
    else // if we didn't find it, calculate it and add it to the list!
    {
        printf("DANGER!!!!!!!  Updating IntList inside a parellel loop breaks thread safety!\n");
        printf("   I shouldn't be here in GetA(%d, %d, %d, %d):   key =%llx",index_p,index_pp,J,norm,key);
        exit(EXIT_FAILURE);
    }
    return A; 
  }


  //Function for the partial wave decomposition given by Eq 4.20 of 
  //K. Erkelenz, R. Alzetta, and K. Holinde, Nucl. Phys. A 176, 413 (1971).
  //We treat the fermi and gt as the same since we include the spin part for the gt when
  //computing the TBMEs. Since for all the possible cases used a value of J equal to l in
  //A(p,pp,J,l), we pass L rather than J. Furthermore, l = 0 by definition
  //for this case. Fetches the values of A from the precomputed list for efficiency.
   double W_fermi_gt(double p, double pp,int index_p, int index_pp, int l, int lp, int J, std::unordered_map<uint64_t,double>& AList)
  {
    double W = 0;
    W += 2*GetA(index_p,index_pp,l,0,AList);
    return W;
  }

  //Function for the partial wave decomposition given by Eq 4.20 of 
  //K. Erkelenz, R. Alzetta, and K. Holinde, Nucl. Phys. A 176, 413 (1971).
  //We treat the fermi and gt as the same since we include the spin part for the gt when
  //computing the TBMEs. Since for all the possible cases used a value of J equal to l in
  //A(p,pp,J,l), we pass L rather than J. Furthermore, l = 0 by definition
  //for this case. Fetches the values of A from the precomputed list for efficiency.
  //W function is multiplied by the regulator 
  double W_contact(double p, double pp,int index_p, int index_pp, int l, int lp, int J, std::unordered_map<uint64_t,double>& AList)
  {
    double W = 0;
    W += 2*GetA(index_p,index_pp,0,0,AList);
    // W+=Regulator(p,pp);
    return W;
  }

  //Function for the partial wave decomposition given by Eq 4.24 of 
  //K. Erkelenz, R. Alzetta, and K. Holinde, Nucl. Phys. A 176, 413 (1971).
  //Only the cases for S=1 have been implemented since this decay can only happen with S=1
  //The operator is composed of 3x(tensor)-(spin) of the paper. Fetches the values of A from the precomputed list for efficiency.
  double W_tensor(double p, double pp,int index_p, int index_pp, int l, int lp, int J, std::unordered_map<uint64_t,double>& AList)
  {
    double W = 0;
    if (lp==J && J==l)
    {
       W += 6*((pp*pp+p*p)*GetA(index_p,index_pp,J,1,AList)-2*pp*p/(2*J+1)*(J*GetA(index_p,index_pp,J+1,1,AList)+(J+1)*GetA(index_p,index_pp,J-1,1,AList)))-2*GetA(index_p,index_pp,J,0,AList);
    }
    else if (lp ==J-1 && l==lp)
    {
      W += 6*((pp*pp+p*p)*GetA(index_p,index_pp,J-1,1,AList)-2*p*pp*GetA(index_p,index_pp,J,1,AList))/(2*J+1)-2*GetA(index_p,index_pp,J-1,0,AList);
    }
    else if (lp == J+1 && l==lp)
    {
      W += 6*(-(pp*pp+p*p)*GetA(index_p,index_pp,J+1,1,AList)+2*p*pp*GetA(index_p,index_pp,J,1,AList))/(2*J+1)-2*GetA(index_p,index_pp,J+1,0,AList);
    }
    else if (lp == J+1 && l==J-1)
    {
      W += -(12*sqrt(J*(J+1))/(2*J+1))*(p*p*GetA(index_p,index_pp,J+1,1,AList)+pp*pp*GetA(index_p,index_pp,J-1,1,AList)-2*p*pp*GetA(index_p,index_pp,J,1,AList));//Minus sign in front is to match with Takayuki's number but I'm not sure why it should be there...
    }
    else if (lp == J-1 && l==J+1)
    {
      W += -(12*sqrt(J*(J+1))/(2*J+1))*(p*p*GetA(index_p,index_pp,J-1,1,AList)+pp*pp*GetA(index_p,index_pp,J+1,1,AList)-2*p*pp*GetA(index_p,index_pp,J,1,AList));//"""
    }
    else
    {
      std::cout<<"Problem..."<<std::endl;
    }
    return W; 
  }


  //Integrand for the integral of the neutrino potential over mometum space. 
  double fq(double p, double pp,int index_p, int index_pp, int n, int l, int np, int lp, int J, double hw, std::string transition, double Eclosure, std::string src, std::unordered_map<uint64_t,double>& AList)
  { 
    std::map<std::string, std::function<double(double,double,int,int,int,int,int,std::unordered_map<uint64_t,double>& A)> > WList = {{"F",W_fermi_gt},{"GT",W_fermi_gt},{"T",W_tensor},{"C", W_contact}};
    double W = WList[transition](p,pp,index_p,index_pp,l,lp,J,AList);
    return p*p*pp*pp*HO_Radial_psi_mom(n,l,hw,p)*W*HO_Radial_psi_mom(np,lp,hw,pp);
  }

  // Integrand for the contact operator over momentum space
  double fq_contact(double p, double pp, int n, int l, int np, int lp, double hw)
  {
     return p*p*pp*pp*HO_Radial_psi_mom(n,l,hw,p)*RegulatorNonLocal(p,pp)*HO_Radial_psi_mom(np,lp,hw,pp);
  }

  //Performs the integral over moemntum space using Gauss-Legendre quadrature quadrilateral rule
  double integrate_dq(int n, int l, int np, int lp, int J, double hw, std::string transition, double Eclosure, std::string src, int npoints, gsl_integration_glfixed_table * t, std::unordered_map<uint64_t,double>& AList)
  { 
    double I = 0;
    if (transition=="GT" or transition=="F" or transition=="T")
    {
      for (int i = 0 ; i< npoints; i++)
      {
        double xi;
        double wi;
        gsl_integration_glfixed_point(0,25,i,&xi,&wi,t);
        for (int j = 0; j<npoints; j++)
        {
          double xj;
          double wj;
          gsl_integration_glfixed_point(0,25,j,&xj,&wj,t);
          I += wi*wj*fq(xi,xj,i,j,n,l,np,lp,J,hw,transition,Eclosure,src,AList); 
        }
      }
    }
    else if (transition=="C")
    {
      for (int i = 0 ; i< npoints; i++)
      {
        double xi;
        double wi;
        gsl_integration_glfixed_point(0,25,i,&xi,&wi,t);
        for (int j = 0; j<npoints; j++)
        {
          double xj;
          double wj;
          gsl_integration_glfixed_point(0,25,j,&xj,&wj,t);
          I += wi*wj*fq_contact (xi,xj,n,l,np,lp,hw); 
        }
      }
    }
    return I;
  }

  //Functions to precompute the integrals of the neutrino potentials for efficiency
  //when computing the TBMEs
  uint64_t IntHash(int n, int l, int np, int lp, int J)
  {
     return   (((uint64_t)(n)) << 40)
            + (((uint64_t)(l)) << 30)
            + (((uint64_t)(np)) << 20)
            + (((uint64_t)(lp)) << 10)
            + ((uint64_t)(J));
  }

  void IntUnHash(uint64_t key, uint64_t& n, uint64_t& l, uint64_t& np, uint64_t& lp, uint64_t& J)
  {
     n  = (key >> 40) & 0x3FFL;
     l  = (key >> 30) & 0x3FFL;
     np = (key >> 20) & 0x3FFL;
     lp = (key >> 10) & 0x3FFL;
     J  = (key      ) & 0x3FFL;
  }

  std::unordered_map<uint64_t,double> PreCalculateM0nuIntegrals(int e2max, double hw, std::string transition, double Eclosure, std::string src)
  {
    IMSRGProfiler profiler;
    double t_start_pci = omp_get_wtime(); // profiling (s)
    std::unordered_map<uint64_t,double> IntList;
    std::unordered_map<uint64_t,double> AList;
    int size=500;
    std::cout<<"calculating integrals wrt dp and dpp..."<<std::endl;
    if (transition != "C")
    {
      AList = PrecalculateA(e2max,Eclosure,transition,size);
      std::cout<<"Done precomputing A's."<<std::endl;
    }
    int maxn = e2max/2;
    int maxl = e2max;
    int maxnp = e2max/2;
    std::vector<uint64_t> KEYS;
    if (transition == "F" or transition == "GT")
    {
      for (int S = 0; S<=1; S++)
      {
        for (int n=0; n<=maxn; n++)
        {
          for (int l=0; l<=maxl-2*n; l++)
          {
            int tempminnp = n; // NOTE: need not start from 'int np=0' since IntHash(n,l,np,l) = IntHash(np,l,n,l), by construction
            for (int np=tempminnp; np<=maxnp; np++)
            {
              int minJ = abs(l-S);
              int tempmaxJ = l+S;
              for (int J = minJ; J<= tempmaxJ; J++)
              {
                uint64_t key = IntHash(n,l,np,l,J);
                KEYS.push_back(key);
                IntList[key] = 0.0; // "Make sure eveything's in there to avoid a rehash in the parallel loop" (RS)
              }
            }
          }
        }
      }
      
    }
    else if (transition == "T")
    {
      for (int n=0; n<=maxn; n++)
      {
        for (int l=1; l<=maxl-2*n; l++)
        {
          int tempminnp = n; // NOTE: need not start from 'int np=0' since IntHash(n,l,np,lp) = IntHash(np,lp,n,l), by construction
          //int tempminnp = 0;
          for (int np=tempminnp; np<=maxnp; np++)
          {
            int tempminlp = (n==np ? l : 1); // NOTE: need not start from 'int lp=0' since IntHash(n,l,np,lp) = IntHash(np,lp,n,l), by construction
            int maxlp = std::min(l+2,maxl);
            for (int lp = tempminlp; lp<=maxlp; lp++)
            { 
              if ((abs(lp-l) != 2) and (abs(lp-l) != 0)) continue;
              int minJ = std::max(abs(l-1),abs(lp-1));
              int tempmaxJ = std::min(l+1,lp+1);
              for (int J = minJ; J<= tempmaxJ; J++)
              {
                uint64_t key = IntHash(n,l,np,lp,J);
                KEYS.push_back(key);
                IntList[key] = 0.0; // "Make sure eveything's in there to avoid a rehash in the parallel loop" (RS)
              }
            }
          }
        }
      }
    }
    else if (transition =="C")
    {
      for (int n=0; n<=maxn; n++)
      {
        int l = 0;
        int tempminnp = n; // NOTE: need not start from 'int np=0' since IntHash(n,l,np,l) = IntHash(np,l,n,l), by construction
        for (int np=tempminnp; np<=maxnp; np++)
        {
          int J = 0;
          uint64_t key = IntHash(n,l,np,l,J);
          KEYS.push_back(key);
          IntList[key] = 0.0; // "Make sure eveything's in there to avoid a rehash in the parallel loop" (RS)
        }
      }
    }

    gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc(size);
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i=0; i<KEYS.size(); i++)
    {
      uint64_t key = KEYS[i];
      uint64_t n,l,np,lp,J;
      IntUnHash(key, n,l,np,lp,J);
      IntList[key] = integrate_dq(n,l,np,lp,J,hw,transition,Eclosure,src,size,t,AList); // these have been ordered by the above loops such that we take the "lowest" value of decimalgen(n,l,np,lp,maxl,maxnp,maxlp), see GetIntegral(...)
      // Uncomment if you want to verify integrals values
      // double prefact = 77.233;
      // std::stringstream intvalue;
      // intvalue<<n<<" "<<np<<" "<<prefact*IntList[key]<<std::endl;
      // std::cout<<intvalue.str();
      
    }
    gsl_integration_glfixed_table_free(t);
    
    std::cout<<"...done calculating the integrals"<<std::endl;
    std::cout<<"IntList has "<<IntList.bucket_count()<<" buckets and a load factor "<<IntList.load_factor()
      <<", estimated storage ~= "<<((IntList.bucket_count() + IntList.size())*(sizeof(size_t) + sizeof(void*)))/(1024.0*1024.0*1024.0)<<" GB"<<std::endl; // copied from (RS)
    profiler.timer["PreCalculateM0nuIntegrals"] += omp_get_wtime() - t_start_pci; // profiling (r)
    return IntList;
  }

  //Get an integral from the IntList cache or calculate it (parallelization dependent)
  double GetM0nuIntegral(int e2max, int n, int l, int np, int lp,int J, double hw, std::string transition, double Eclosure, std::string src, std::unordered_map<uint64_t,double> &IntList)
  {
    int maxl = e2max;
    int maxnp = e2max/2;
    int maxlp = e2max;
    int order1 = decimalgen(n,l,np,lp,maxl,maxnp,maxlp);
    int order2 = decimalgen(np,lp,n,l,maxl,maxnp,maxlp); // notice I was careful here with the order of maxl,maxnp,maxlp to make proper comparison
    if (order1 > order2)
    {
      std::swap(n,np); // using symmetry IntHash(n,l,np,lp) = IntHash(np,lp,n,l)
      std::swap(l,lp); // " " " " "
    }
    // long int key = IntHash(n,l,np,lp); // if I ever get that version working...
    // std::cout<<"n ="<<n<<", l ="<<l<<", np = "<<np<<", S = "<<S<<", J = "<<J<<std::endl;
    uint64_t key = IntHash(n,l,np,lp,J);
    auto it = IntList.find(key);
    if (it != IntList.end()) // return what we've found
    {
      return it -> second;
    }
    else // if we didn't find it, calculate it and add it to the list!
    {
      double integral;
      int size = 500;
      gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc(size);
      std::unordered_map<uint64_t,double> AList = PrecalculateA(e2max,Eclosure,transition,size);
      integral = integrate_dq(n, l, np, lp,J,hw, transition, Eclosure, src,size, t,AList);
      gsl_integration_glfixed_table_free(t);
      std::stringstream intvalue;
      intvalue<<n<<" "<<np<<" "<<l<<" "<<lp<<" "<<std::endl;
      std::cout<<intvalue.str();
      if (omp_get_num_threads() >= 2)
      {
        printf("DANGER!!!!!!!  Updating IntList inside a parellel loop breaks thread safety!\n");
        printf("   I shouldn't be here in GetIntegral(%d, %d, %d, %d, %d):   key =%llx   integral=%f\n",n,l,np,lp,J,key,integral);
        exit(EXIT_FAILURE);
      }
      IntList[key] = integral;
      
      return integral;
    }
  }


  /// Gamow Teller operator for neutrinoless double beta decay. Opertor is written in momentum space and takes the form
  /// \f{equation} 
  ///       O_{GT}(\bold{q'}) = \frac{R}{2\pi^2}\frac{h_{GT}(q')}{q'(q'+E_c)}(\boldsymbol{\sigma_1} \cdot \boldsymbol{\sigma_2}) \tau_{1,+}\tau_{2,+}
  /// \f}
  /// Where \f$ h_{GT} \f$ is the neutrino potenital  impletmented above and \f$ E_c \f$ is the closure energy.
  /// Operator is then evaluated in the lab frame oscialltor basis.
  /// More detail on how to obatin the form of the operator can be found in https://drive.google.com/file/d/1QaMfuvQ7I3NM5h_ppjyIPsap3SWmCe2o/view?usp=sharing
  /// and on how to evaluate in lab frame in https://drive.google.com/file/d/1C6E2HnzSJ1bzMoIKWfH1GaZKjqaAEluG/view?usp=sharing
  Operator GamowTeller(ModelSpace& modelspace, double Eclosure, std::string src)
  {
    bool reduced = true;
    double t_start, t_start_tbme, t_start_omp; // profiling (v)
    t_start = omp_get_wtime(); // profiling (s)
    std::string transition = "GT";
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    int e2max = modelspace.GetE2max(); // 2*emax
    Operator M0nuGT_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    M0nuGT_TBME.SetHermitian(); // it should be Hermitian
    int Anuc = modelspace.GetTargetMass(); // the mass number for the desired nucleus
    const double Rnuc = R0*pow(Anuc,1.0/3.0); // the nuclear radius [fm]
    const double prefact = Rnuc/(PI*PI); // factor in-front of M0nu TBME, extra global 2 for nutbar (as confirmed by benchmarking with Ca48 NMEs) [fm]
    modelspace.PreCalculateMoshinsky(); // pre-calculate the needed Moshinsky brackets, for efficiency
    std::unordered_map<uint64_t,double> IntList = PreCalculateM0nuIntegrals(e2max,hw,transition, Eclosure, src); // pre-calculate the needed integrals over dq and dr, for efficiency
    M0nuGT_TBME.profiler.timer["M0nuGT_1_sur"] += omp_get_wtime() - t_start; // profiling (r)
    // create the TBMEs of M0nu
    // auto loops over the TBME channels and such
    std::cout<<"calculating M0nu TBMEs..."<<std::endl;
    t_start_tbme = omp_get_wtime(); // profiling (s)
    for (auto& itmat : M0nuGT_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      double Jhat; // set below based on "reduced" variable
      if (reduced == false)
      {
        Jhat = 1.0; // for non-reduced elements, to compare with JE
      }
      else //if (reduced == "true")
      { 
        Jhat = sqrt(2*J+1); // the hat factor of J
      }
      t_start_omp = omp_get_wtime(); // profiling (s)
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          int eps_ab = 2*na + la + 2*nb + lb; // for conservation of energy in the Moshinsky brackets
          int eps_cd = 2*nc + lc + 2*nd + ld; // for conservation of energy in the Moshinsky brackets
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          for (int S=0; S<=1; S++) // sum over total spin...
          {
            int Seval;
            Seval = 2*S*(S + 1) - 3;
            int L_min = std::max(abs(la-lb),abs(lc-ld));
            int L_max = std::min(la+lb,lc+ld);
            for (int L = L_min; L <= L_max; L++) // ...and sum over orbital angular momentum
            {
              double sumMT = 0; // for the Moshinsky transformation
              double sumMTas = 0; // (anti-symmetric part)              
              double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
              double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
              double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
              double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
              double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
              double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
              double bulk = Seval*nNJab*nNJcd; // bulk product of the above
              double bulkas = Seval*nNJab*nNJdc; // (anti-symmetric part)
              int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
              for (int nr = 0; nr <= tempmaxnr; nr++)
              {
                double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of my thesis
                if ((npr >= 0) and ((eps_cd-eps_ab)%2 == 0))
                {
                  int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                  for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                  {
                    int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                    int tempmaxlr = floor((eps_ab + L)/2.0) - (nr + Ncom); // " " " " "
                    for (int lr = tempminlr; lr <= tempmaxlr; lr++)
                    {
                      int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of my thesis
                      double integral = 0;
                      double normJrel;
                      double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                      double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                      double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                      int minJrel= abs(lr-S);
                      int maxJrel = lr+S;
                      for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                      {
                        normJrel = sqrt((2*Jrel+1)*(2*L+1))*phase(L+lr+S+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                        integral += normJrel*normJrel*GetM0nuIntegral(e2max,nr,lr,npr,lr,Jrel,hw,transition,Eclosure,src,IntList); // grab the pre-calculated integral wrt dq and dr from the IntList of the modelspace class
                      }//end of for-loop over Jrel
                      sumMT += Df*Di*integral; // perform the Moshinsky transformation
                      sumMTas += Df*asDi*integral; // (anti-symmetric part)
                    } // end of for-loop over: lr
                  } // end of for-loop over: Ncom
                } // end of if: npr \in \Nat_0
              } // end of for-loop over: nr
              sumLS += bulk*sumMT; // perform the LS-coupling sum
              sumLSas += bulkas*sumMTas; // (anti-symmetric part)
            } // end of for-loop over: L
          } // end of for-loop over: S
          double Mtbme = asNorm(ia,ib)*asNorm(ic,id)*prefact*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas); // compute the final matrix element, anti-symmetrize
          // double Mtbme = prefact * Jhat * sumLS; // compute the final matrix element, not anti-symmetrize

          M0nuGT_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      } // end of for-loop over: ibra
      M0nuGT_TBME.profiler.timer["M0nuGT_3_omp"] += omp_get_wtime() - t_start_omp; // profiling (r)
    } // end of for-loop over: auto
    std::cout<<"...done calculating M0nu TBMEs"<<std::endl;
    M0nuGT_TBME.profiler.timer["M0nuGT_2_tbme"] += omp_get_wtime() - t_start_tbme; // profiling (r)
    M0nuGT_TBME.profiler.timer["M0nuGT_Op"] += omp_get_wtime() - t_start; // profiling (r)
    return M0nuGT_TBME;
  }

  /// Fermi operator for neutrinoless double beta decay. Opertor is written in momentum space and takes the form
  /// \f[ 
  ///       O_{F}(\bold{q'}) = \frac{R}{2\pi^2}\frac{h_{F}(q')}{q'(q'+E_c)} \tau_{1,+}\tau_{2,+}
  /// \f]
  /// Where \f$ h_{F} \f$ is the neutrino potenital  impletmented above and \f$ E_c \f$ is the closure energy.
   /// Operator is then evaluated in the lab frame oscialltor basis.
  /// More detail on how to obatin the form of the operator can be found in https://drive.google.com/file/d/1QaMfuvQ7I3NM5h_ppjyIPsap3SWmCe2o/view?usp=sharing
  /// and on how to evaluate in lab frame in https://drive.google.com/file/d/1C6E2HnzSJ1bzMoIKWfH1GaZKjqaAEluG/view?usp=sharing
  Operator Fermi(ModelSpace& modelspace, double Eclosure, std::string src)
  {
    bool reduced = true;
    double t_start, t_start_tbme, t_start_omp; // profiling (v)
    t_start = omp_get_wtime(); // profiling (s)
    std::string transition = "F";
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    int e2max = modelspace.GetE2max(); // 2*emax
    Operator M0nuF_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    std::cout<<"     reduced            =  "<<reduced<<std::endl;
    M0nuF_TBME.SetHermitian(); // it should be Hermitian
    int Anuc = modelspace.GetTargetMass(); // the mass number for the desired nucleus
    const double Rnuc = R0*pow(Anuc,1.0/3.0); // the nuclear radius [fm]
    const double prefact = Rnuc/(PI*PI); // factor in-front of M0nu TBME, extra global 2 for nutbar (as confirmed by benchmarking with Ca48 NMEs) [fm]
    modelspace.PreCalculateMoshinsky(); // pre-calculate the needed Moshinsky brackets, for efficiency
    std::unordered_map<uint64_t,double> IntList = PreCalculateM0nuIntegrals(e2max,hw,transition, Eclosure, src); // pre-calculate the needed integrals over dq and dr, for efficiency
    M0nuF_TBME.profiler.timer["M0nuF_1_sur"] += omp_get_wtime() - t_start; // profiling (r)
    // create the TBMEs of M0nu
    // auto loops over the TBME channels and such
    std::cout<<"calculating M0nu TBMEs..."<<std::endl;
    t_start_tbme = omp_get_wtime(); // profiling (s)
    for (auto& itmat : M0nuF_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      double Jhat; // set below based on "reduced" variable
      if (reduced == false)
      {
        Jhat = 1.0; // for non-reduced elements, to compare with JE
      }
      else //if (reduced == "R")
      {
        Jhat = sqrt(2*J + 1); // the hat factor of J
      }
      t_start_omp = omp_get_wtime(); // profiling (s)
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          int eps_ab = 2*na + la + 2*nb + lb; // for conservation of energy in the Moshinsky brackets
          int eps_cd = 2*nc + lc + 2*nd + ld; // for conservation of energy in the Moshinsky brackets
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          for (int S=0; S<=1; S++) // sum over total spin...
          {
            int L_min = std::max(std::max(abs(la-lb),abs(lc-ld)), abs(J-S));
            int L_max = std::min(std::min(la+lb,lc+ld),J+S);
            for (int L = L_min; L<=L_max;L++)
            {
              double sumMT = 0; // for the Moshinsky transformation
              double sumMTas = 0; // (anti-symmetric part)
              double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
              double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
              double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
              double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
              double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
              double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
              double bulk = nNJab*nNJcd; // bulk product of the above
              double bulkas = nNJab*nNJdc; // (anti-symmetric part)
              int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
              for (int nr = 0; nr <= tempmaxnr; nr++)
              {
                double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of my thesis
                if ((npr >= 0) and (npr == floor(npr)))
                {
                  int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                  for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                  {
                    int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                    int tempmaxlr = floor((eps_ab + L)/2.0) - (nr + Ncom); // " " " " "
                    for (int lr = tempminlr; lr <= tempmaxlr; lr++)
                    {
                      int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of my thesis
                      double integral = 0;
                      double normJrel;
                      double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                      double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                      double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                      int minJrel= abs(lr-S);
                      int maxJrel = lr+S;
                      for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                      {
                        normJrel = sqrt((2*Jrel+1)*(2*L+1))*phase(L+lr+S+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                        integral += normJrel*normJrel*GetM0nuIntegral(e2max,nr,lr,npr,lr,Jrel,hw,transition,Eclosure,src,IntList); // grab the pre-calculated integral wrt dq and dr from the IntList of the modelspace class
                      }//end of for-loop over Jrel
                      sumMT += Df*Di*integral; // perform the Moshinsky transformation
                      sumMTas += Df*asDi*integral; // (anti-symmetric part)
                    } // end of for-loop over: lr
                  } // end of for-loop over: Ncom
                } // end of if: npr \in \Nat_0
              } // end of for-loop over: nr
              sumLS += bulk*sumMT; // perform the LS-coupling sum
              sumLSas += bulkas*sumMTas; // (anti-symmetric part)
            } // end of for-loop over: L
          } // end of for-loop over: S
          double Mtbme = asNorm(ia,ib)*asNorm(ic,id)*prefact*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas); // compute the final matrix element, anti-symmetrize
          M0nuF_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      } // end of for-loop over: ibra
      M0nuF_TBME.profiler.timer["M0nuF_3_omp"] += omp_get_wtime() - t_start_omp; // profiling (r)
    } // end of for-loop over: auto
    std::cout<<"...done calculating M0nu TBMEs"<<std::endl;
    M0nuF_TBME.profiler.timer["M0nuF_2_tbme"] += omp_get_wtime() - t_start_tbme; // profiling (r)
    M0nuF_TBME.profiler.timer["M0nuF_adpt_Op"] += omp_get_wtime() - t_start; // profiling (r)
    return M0nuF_TBME;
  }




  /// Tensor operator for neutrinoless double beta decay. Opertor is written in momentum space and takes the form
  /// \f[ 
  ///        O_{T}(\bold{q'}) = -\frac{R}{2\pi^2}\frac{h_{T}(q')}{q'(q'+E_c)}[3(\boldsymbol{\sigma_1}\cdot\boldsymbol{\hat{q'}})(\boldsymbol{\sigma_2}\cdot\boldsymbol{\hat{q'}})-(\boldsymbol{\sigma_1} \cdot \boldsymbol{\sigma_2})] \tau_{1,+}\tau_{2,+}
  /// \f]
  /// Where \f$ h_{T} \f$ is the neutrino potenital  impletmented above and \f$ E_c \f$ is the closure energy. The minus factor up front as been included in the neutrino potential for simplicity.
  /// Operator is then evaluated in the lab frame oscialltor basis.
  /// More detail on how to obatin the form of the operator can be found in https://drive.google.com/file/d/1QaMfuvQ7I3NM5h_ppjyIPsap3SWmCe2o/view?usp=sharing
  /// and on how to evaluate in lab frame in https://drive.google.com/file/d/1C6E2HnzSJ1bzMoIKWfH1GaZKjqaAEluG/view?usp=sharing
  Operator Tensor(ModelSpace& modelspace, double Eclosure, std::string src)
  {
    bool reduced = true;
    double t_start, t_start_tbme, t_start_omp; // profiling (v)
    t_start = omp_get_wtime(); // profiling (s)
    std::string transition = "T";
    // run through the initial set-up routine
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    int e2max = modelspace.GetE2max(); // 2*emax
    Operator M0nuT_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    std::cout<<"     reduced            =  "<<reduced<<std::endl;
    M0nuT_TBME.SetHermitian(); // it should be Hermitian
    int Anuc = modelspace.GetTargetMass(); // the mass number for the desired nucleus
    const double Rnuc = R0*pow(Anuc,1.0/3.0); // the nuclear radius [MeV^-1]
    const double prefact = Rnuc/(PI*PI); // factor in-front of M0nu TBME, extra global 2 for nutbar (as confirmed by benchmarking with Ca48 NMEs) [MeV^-1]
    modelspace.PreCalculateMoshinsky(); // pre-calculate the needed Moshinsky brackets, for efficiency
    std::unordered_map<uint64_t,double> IntList = PreCalculateM0nuIntegrals(e2max,hw,transition, Eclosure, src); // pre-calculate the needed integrals over dq and dr, for efficiency
    M0nuT_TBME.profiler.timer["M0nuT_1_sur"] += omp_get_wtime() - t_start; // profiling (r)
    // create the TBMEs of M0nu
    // auto loops over the TBME channels and such
    std::cout<<"calculating M0nu TBMEs..."<<std::endl;
    t_start_tbme = omp_get_wtime(); // profiling (s)
    for (auto& itmat : M0nuT_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      double Jhat; // set below based on "reduced" variable
      if (reduced == false)
      {
        Jhat = 1.0; // for non-reduced elements, to compare with JE
      }
      else //if (reduced == "R")
      {
        Jhat = sqrt(2*J + 1); // the hat factor of J
      }
      t_start_omp = omp_get_wtime(); // profiling (s)
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky(), PreCalcT6j, and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          int eps_ab = 2*na + la + 2*nb + lb; // for conservation of energy in the Moshinsky brackets
          int eps_cd = 2*nc + lc + 2*nd + ld; // for conservation of energy in the Moshinsky brackets
          double sumLS = 0; // for the wave functions decomposition
          double sumLSas = 0; // (anti-symmetric part)
          int S = 1;
          int Lf_min = std::max(std::abs(la-lb), std::abs(J-S));
          int Lf_max = std::min(la+lb, J+S);
          for (int Lf = Lf_min; Lf<=Lf_max; Lf++) // sum over angular momentum coupled to l_a and l_b
          {
            double normab = sqrt((2*Lf+1)*(2*S+1)*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
            double nNJab = normab*AngMom::NineJ(la,lb,Lf,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
            int Li_min = std::max(std::abs(lc-ld), std::max(std::abs(J-S), std::abs(Lf-2)));
            int Li_max = std::min( lc+ld, std::min( J+S, Lf+2) );
            for (int Li = Li_min; Li <= Li_max; Li++) // sum over angular momentum coupled to l_c and l_d
            { 
              double sumMT = 0; // for the Moshinsky transformation
              double sumMTas = 0; // (anti-symmetric part)
              double normcd = sqrt((2*Li+1)*(2*S+1)*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
              double nNJcd = normcd*AngMom::NineJ(lc,ld,Li,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
              double nNJdc = normcd*AngMom::NineJ(ld,lc,Li,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)

              double bulk = nNJab*nNJcd; // bulk product of the above
              double bulkas = nNJab*nNJdc; // (anti-symmetric part)
              for ( int lr=1; lr<=eps_ab; lr++)
              {
                for (int nr=0; nr<=(eps_ab-lr)/2; nr++)
                {
                  int tempmaxNcom = std::min((eps_ab-2*nr-lr)/2, eps_cd);
                  for (int Ncom=0; Ncom<=tempmaxNcom; Ncom++ )
                  {
                    int Lam = eps_ab - 2*nr - lr - 2*Ncom;
                    if ( (Lam+2*Ncom) > eps_cd ) continue;
                    if ( (std::abs(Lam-lr)>Lf)  or ( (Lam+lr)<Lf) ) continue;
                    if ( (lr+Lam+eps_ab)%2>0 ) continue;
                    for (int npr=0; npr<=(eps_cd-2*Ncom-Lam)/2; npr++)
                    {
                        int lpr = eps_cd-2*Ncom-Lam-2*npr;
                        if (  (lpr+lr)%2 >0 ) continue;
                        if (lpr<1) continue;
                        if ( (std::abs(Lam-lpr)>Li)  or ( (Lam+lpr)<Li) ) continue;
                        double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,Lf); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                        double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lpr,nc,lc,nd,ld,Li); // " " " "
                        double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lpr,nd,ld,nc,lc,Li);// (anti-symmetric part)
                        double integral = 0;
                        double normJrel, normJrelp;
                        int minJrel = std::max(abs(lr-S),abs(lpr-S));
                        int maxJrel = std::min(lr+S,lpr+S);
                        for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                        {
                          if ( (std::abs(J-Jrel)>Lam)  or ( (Jrel+J)<Lam) ) continue;
                          normJrel  = sqrt((2*Jrel+1)*(2*Lf+1))*phase(Lf+lr+J+S)*AngMom::SixJ(Lam,lr,Lf,S,J,Jrel);
                          normJrelp = sqrt((2*Jrel+1)*(2*Li+1))*phase(Li+lpr+J+S)*AngMom::SixJ(Lam,lpr,Li,S,J,Jrel);
                          integral += normJrel*normJrelp*GetM0nuIntegral(e2max,nr,lr,npr,lpr,Jrel,hw,transition,Eclosure,src,IntList);
                        }
                        sumMT += Df*Di*integral; // perform the Moshinsky transformation
                        sumMTas += Df*asDi*integral; // (anti-symmetric part)
                      } // end of for-loop over: lpr
                    } // end of for-loop over: Ncom
                  } // end of for-loop over: nr
                } // end of for-loop over: lr
              sumLS += bulk*sumMT; // perform the LS-coupling sum
              sumLSas += bulkas*sumMTas; // (anti-symmetric part)
            } // end of for-loop over: Li
          } // end of for-loop over: Lf        
          // double Mtbme = asNorm(ia,ib)*asNorm(ic,id)*prefact*Jhat*sumLS; // compute the final matrix element, anti-symmetrize          
          double Mtbme = asNorm(ia,ib)*asNorm(ic,id)*prefact*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas); // compute the final matrix element, anti-symmetrize
          M0nuT_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      } // end of for-loop over: ibra
      M0nuT_TBME.profiler.timer["M0nuT_3_omp"] += omp_get_wtime() - t_start_omp; // profiling (r)
    } // end of for-loop over: auto
    M0nuT_TBME.profiler.timer["M0nuT_2_tbme"] += omp_get_wtime() - t_start_tbme; // profiling (r)
    M0nuT_TBME.profiler.timer["M0nuT _Op"] += omp_get_wtime() - t_start; // profiling (r)
    return M0nuT_TBME;
  }


  //This is function is to be able to compare with Jiangming's files for his contact TBME
  double jl(double j, double l)
  {
    return floor(l+j-1/2);
  }

  /// Contact operator for neutrinoless double beta decay. Opertor is written in momentum space. Only factor missing is the gvv/gA^2 which is multiplied to the NME
  /// at the end since gvv depends on the interactions. Values of gvv can be found in arXiv:2105.05415.
  /// The non-local  contact oprator in momentum space takes the form 
  /// \f[ 
  ///        O_{C}(p,p') = - 4  \frac{R}{pi} exp((\frac{p}{\Lambda})^{2n})exp((\frac{p'}{\Lambda})^{2n})
  /// \f]
  /// Operator is then evaluated in the lab frame oscialltor basis.
  Operator Contact(ModelSpace& modelspace, double Eclosure, std::string src)
  {
    bool reduced = true;
    double t_start, t_start_tbme, t_start_omp; // profiling (v)
    t_start = omp_get_wtime(); // profiling (s)
    std::string transition = "C";
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    int e2max = modelspace.GetE2max(); // 2*emax
    Operator M0nuC_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    std::cout<<"     reduced            =  "<<reduced<<std::endl;
    M0nuC_TBME.SetHermitian(); // it should be Hermitian
    int Anuc = modelspace.GetTargetMass(); // the mass number for the desired nucleus
    // const double scale_ct = 77.23250846634942;
    const double scale_ct = (M_NUCLEON*NUCLEON_AXIAL_G*NUCLEON_AXIAL_G*HBARC)/(4*F_PI*F_PI);
    const double Rnuc = R0*pow(Anuc,1.0/3.0); // the nuclear radius [fm]
    const double prefact = -Rnuc / (4 * SQRT2 * PI * PI) * scale_ct * scale_ct; // factor in-front of M0nu TBME, includes the -2 from the -2gvv and the 4*pi from angular integral as well as the 1/2 from the convention used by Cirgilano and all when computing gvv. Only thing missing is gvv/gA^2
    // const double prefact = Rnuc/(16*SQRT2*PI*PI*PI)*scale_ct*scale_ct; //factor to match with Jiagnming. Need to have the TBME multiplied by (-2gvv)*4*pi/ga^2 in the end.   

    modelspace.PreCalculateMoshinsky(); // pre-calculate the needed Moshinsky brackets, for efficiency
    std::unordered_map<uint64_t,double> IntList = PreCalculateM0nuIntegrals(e2max,hw,transition, Eclosure, src); // pre-calculate the needed integrals over dp and dpp
    M0nuC_TBME.profiler.timer["M0nuC_1_sur"] += omp_get_wtime() - t_start; // profiling (r)
    // create the TBMEs of M0nu
    // auto loops over the TBME channels and such
    std::cout<<"calculating M0nu TBMEs..."<<std::endl;
    t_start_tbme = omp_get_wtime(); // profiling (s)
    for (auto& itmat : M0nuC_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      double Jhat; // set below based on "reduced" variable
      if (reduced == false)
      {
        Jhat = 1.0; // for non-reduced elements, to compare with JE
      }
      else //if (reduced == "R")
      {
        Jhat = sqrt(2*J + 1); // the hat factor of J
      }
      t_start_omp = omp_get_wtime(); // profiling (s)
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        int na = oa.n;
        int nb = ob.n;
        int la = oa.l;
        int lb = ob.l;
        double ja = oa.j2/2.0;
        double jb = ob.j2/2.0;
        // double jla = jl(ja,la);
        // double jlb = jl(jb,lb);
        int eps_ab = 2*na + la + 2*nb + lb; // for conservation of energy in the Moshinsky brackets
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int nc = oc.n;
          int nd = od.n;
          int lc = oc.l;
          int ld = od.l;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          // double jlc = jl(jc,lc);
          // double jld = jl(jd,ld);
          int eps_cd = 2*nc + lc + 2*nd + ld; // for conservation of energy in the Moshinsky brackets
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          if (std::abs(eps_ab-eps_cd)%2>0) continue;
          int S=0;
          int L_min = std::max(std::max(abs(la-lb),abs(lc-ld)), abs(J-S));
          int L_max = std::min(std::min(la+lb,lc+ld),J+S);
          for (int L = L_min; L<=L_max;L++)
          {
            double sumMT = 0; // for the Moshinsky transformation
            double sumMTas = 0; // (anti-symmetric part)
            double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
            double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
            double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
            if (abs(nNJab)<1e-7) continue;
            double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
            double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
            double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
            if ((abs(nNJcd)<1e-7) or (abs(nNJdc)<1e-7))  continue;
            double bulk = nNJab*nNJcd; // bulk product of the above
            double bulkas = nNJab*nNJdc; // (anti-symmetric part)
            int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
            for (int nr = 0; nr <= tempmaxnr; nr++)
            {
              double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of my thesis
              if ((npr >= 0) and (npr == floor(npr)))
              {
                int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                {
                  int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                  if (tempminlr>0) continue; //Contact term can only have lr = 0 so no need to loop over lr
                  int lr = 0;
                  int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of my thesis
                  double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                  double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                  double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                  int Jrel = 0;
                  double normJrel = sqrt((2*L+1))*phase(L+lr+Jrel+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                  double integral = normJrel*normJrel*GetM0nuIntegral(e2max,nr,lr,npr,lr,Jrel,hw,transition,Eclosure,src,IntList); // grab the pre-calculated integral wrt dq and dr from the IntList of the modelspace class
                  sumMT += Df*Di*integral; // perform the Moshinsky transformation
                  sumMTas += normJrel*normJrel*Df*asDi*integral; // (anti-symmetric part) 
                } // end of for-loop over: Ncom
              } // end of if: npr \in \Nat_0
            } // end of for-loop over: nr
            sumLS += bulk*sumMT; // perform the LS-coupling sum
            sumLSas += bulkas*sumMTas; // (anti-symmetric part)
          } // end of for-loop over: L
          double Mtbme = asNorm(ia,ib)*asNorm(ic,id)*prefact*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas); // compute the final matrix element, anti-symmetrize
          // std::stringstream intvalue;
          // intvalue<<na<<", "<<nb<<", "<<nc<<", "<<nd<<", "<<jla<<", "<<jlb<<", "<<jlc<<", "<<jld<<", "<<J<<", "<<Mtbme<<std::endl;
          // std::cout<<intvalue.str();  //Prints values of the TBME to compare with Jiangming
          M0nuC_TBME.profiler.timer["M0nuC_3_omp"] += omp_get_wtime() - t_start_omp; // profiling (r)
          M0nuC_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      } // end of for-loop over: ibra
    } // end of for-loop over: auto
    std::cout<<"...done calculating M0nu TBMEs"<<std::endl;
    M0nuC_TBME.profiler.timer["M0nuC_2_tbme"] += omp_get_wtime() - t_start_tbme; // profiling (r)
    M0nuC_TBME.profiler.timer["M0nuC_adpt_Op"] += omp_get_wtime() - t_start; // profiling (r)
    return M0nuC_TBME;
  }


  /// Double Gamow-Teller operator. It is simply the 0vbb GT operator with the neutrino potential set to 1.
  /// More infromation on the process can be found in 10.1103/PhysRevLett.120.142502. 
  Operator DGT_Op(ModelSpace& modelspace)
  {
    Operator DGT_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    DGT_TBME.SetHermitian(); // it should be Hermitian
    modelspace.PreCalculateMoshinsky();
    // create the TBMEs of DGT
    // auto loops over the TBME channels and such
    std::cout<<"calculating DGT TBMEs..."<<std::endl;
    for (auto& itmat : DGT_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      // double Jhat = 1;
      double Jhat = sqrt(2*J + 1); // the hat factor of J
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          int eps_ab = 2*na+la+2*nb+lb;
          int eps_cd = 2*nc+lc+2*nd+ld;
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          if (la+lb == lc+ld)
          {
            for (int S=0; S<=1; S++) // sum over total spin...
            {
              int Seval;
              Seval = 2*S*(S + 1) - 3;
              int L_min = std::max(abs(la-lb),abs(lc-ld));
              int L_max = std::min(la+lb,lc+ld);
              for (int L = L_min; L <= L_max; L++) // ...and sum over orbital angular momentum
              {
                double sumMT = 0; // for the Moshinsky transformation
                double sumMTas = 0; // (anti-symmetric part)              
                double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
                double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
                double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
                double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
                double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
                double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
                double bulk = Seval*nNJab*nNJcd; // bulk product of the above
                double bulkas = Seval*nNJab*nNJdc; // (anti-symmetric part)
                int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
                for (int nr = 0; nr <= tempmaxnr; nr++)
                {
                  double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of my thesis
                  // if ((npr >= 0) and ((eps_cd-eps_ab)%2 == 0))
                  if (npr==nr)
                  {
                    int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                    for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                    {
                      int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                      int tempmaxlr = floor((eps_ab + L)/2.0) - (nr + Ncom); // " " " " "
                      for (int lr = tempminlr; lr <= tempmaxlr; lr++)
                      {
                        int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of my thesis
                        double integral = 0;
                        double normJrel;
                        double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                        double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                        double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                        int minJrel= abs(lr-S);
                        int maxJrel = lr+S;
                        for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                        {
                          normJrel = sqrt((2*Jrel+1)*(2*L+1))*phase(Lam+lr+S+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                          integral += normJrel*normJrel*1;
                        }//end of for-loop over Jrel
                        sumMT += Df*Di*integral; // perform the Moshinsky transformation
                        sumMTas += Df*asDi*integral; // (anti-symmetric part)
                      } // end of for-loop over: lr
                    } // end of for-loop over: Ncom
                  } // end of if: npr \in \Nat_0
                } // end of for-loop over: nr
                sumLS += bulk*sumMT; // perform the LS-coupling sum
                sumLSas += bulkas*sumMTas; // (anti-symmetric part)
              } // end of for-loop over: L
            } // end of for-loop over: S
          }//end of if la+lb==lc+ld
          
          double Mtbme = 2*asNorm(ia,ib)*asNorm(ic,id)*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas); // compute the final matrix element, anti-symmetrize
          DGT_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      }// end of for-loop over: ibra
    } // end of for-loop over: auto
    std::cout<<"...done calculating DGT TBMEs"<<std::endl;
    return DGT_TBME;
  }



  double HO_Radial_psi(int n, int l, double hw, double r)
  {
    double b = sqrt( (HBARC*HBARC) / (hw * M_NUCLEON*0.5) );
    double x = r/b;
    double Norm = 2*sqrt( gsl_sf_fact(n) * pow(2,n+l) / SQRTPI / gsl_sf_doublefact(2*n+2*l+1) / pow(b,3.0) );
    double L = gsl_sf_laguerre_n(n,l+0.5,x*x);
    double psi = Norm * pow(x,l) * exp(-x*x*0.5) * L;
    return psi;
  }


  double fq_radial_GT(double q, double Eclosure, double r12)
  {
    return gsl_sf_bessel_j0(q*r12)*q*GTFormFactor(q*HBARC)/(q+Eclosure/HBARC);
  }


  double integrate_dq_radial_GT(double Eclosure, double r12,  int npoints, gsl_integration_glfixed_table * t)
  {
   double I = 0; 
   for (int i = 0 ; i< npoints; i++)
    {
      double xi;
      double wi;
      gsl_integration_glfixed_point(0,30,i,&xi,&wi,t);
      I += wi*fq_radial_GT(xi,Eclosure,r12); 
    }
    return I;
  }


   std::unordered_map<uint64_t,double> PreCalculateM0nuIntegrals_R(int e2max, double hw, double Eclosure, double r12)
  {
    IMSRGProfiler profiler;
    double t_start_pci = omp_get_wtime(); // profiling (s)
    std::unordered_map<uint64_t,double> IntList;
    int size=1000;
    std::cout<<"calculating integrals wrt dq..."<<std::endl;
    int maxn = e2max/2;
    int maxl = e2max;
    int maxnp = e2max/2;
    std::vector<uint64_t> KEYS;
    for (int S = 0; S<=1; S++)
    {
      for (int n=0; n<=maxn; n++)
      {
        for (int l=0; l<=maxl; l++)
        {
          int tempminnp = n; // NOTE: need not start from 'int np=0' since IntHash(n,l,np,l) = IntHash(np,l,n,l), by construction
          for (int np=tempminnp; np<=maxnp; np++)
          {
            int minJ = abs(l-S);
            int tempmaxJ = l+S;
            for (int J = minJ; J<= tempmaxJ; J++)
            {
              uint64_t key = IntHash(n,l,np,l,J);
              KEYS.push_back(key);
              IntList[key] = 0.0; // "Make sure eveything's in there to avoid a rehash in the parallel loop" (RS)
            }
          }
        }
      }
    } 

    gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc(size);
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i=0; i<KEYS.size(); i++)
    {
      uint64_t key = KEYS[i];
      uint64_t n,l,np,lp,J;
      IntUnHash(key, n,l,np,lp,J);
      IntList[key] = r12*r12*HO_Radial_psi(n, l, hw, r12)*HO_Radial_psi(np, lp, hw, r12)*integrate_dq_radial_GT(Eclosure,r12,size,t); // these have been ordered by the above loops such that we take the "lowest" value of decimalgen(n,l,np,lp,maxl,maxnp,maxlp), see GetIntegral(...)
    }
    gsl_integration_glfixed_table_free(t);
    
    std::cout<<"...done calculating the integrals"<<std::endl;
    std::cout<<"IntList has "<<IntList.bucket_count()<<" buckets and a load factor "<<IntList.load_factor()
      <<", estimated storage ~= "<<((IntList.bucket_count() + IntList.size())*(sizeof(size_t) + sizeof(void*)))/(1024.0*1024.0*1024.0)<<" GB"<<std::endl; // copied from (RS)
    profiler.timer["PreCalculateM0nuIntegrals"] += omp_get_wtime() - t_start_pci; // profiling (r)
    return IntList;
  }

  //Get an integral from the IntList cache or calculate it (parallelization dependent)
  double GetM0nuIntegral_R(int e2max, int n, int l, int np, int lp,int J, double hw, double Eclosure, double r12, std::unordered_map<uint64_t,double> &IntList)
  {
    int maxl = e2max;
    int maxnp = e2max/2;
    int maxlp = e2max;
    int order1 = decimalgen(n,l,np,lp,maxl,maxnp,maxlp);
    int order2 = decimalgen(np,lp,n,l,maxl,maxnp,maxlp); // notice I was careful here with the order of maxl,maxnp,maxlp to make proper comparison
    if (order1 > order2)
    {
      std::swap(n,np); // using symmetry IntHash(n,l,np,lp) = IntHash(np,lp,n,l)
      std::swap(l,lp); // " " " " "
    }
    // long int key = IntHash(n,l,np,lp); // if I ever get that version working...
    // std::cout<<"n ="<<n<<", l ="<<l<<", np = "<<np<<", S = "<<S<<", J = "<<J<<std::endl;
    uint64_t key = IntHash(n,l,np,lp,J);
    auto it = IntList.find(key);
    if (it != IntList.end()) // return what we've found
    {
      return it -> second;
    }
    else // if we didn't find it, calculate it and add it to the list!
    {
      double integral;
      int size = 500;
      gsl_integration_glfixed_table * t = gsl_integration_glfixed_table_alloc(size);
      integral = r12*r12*HO_Radial_psi(n, l, hw, r12)*HO_Radial_psi(np, lp, hw, r12)*integrate_dq_radial_GT(Eclosure,r12,size,t);
      gsl_integration_glfixed_table_free(t);
      if (omp_get_num_threads() >= 2)
      {
        printf("DANGER!!!!!!!  Updating IntList inside a parellel loop breaks thread safety!\n");
        printf("   I shouldn't be here in GetIntegral(%d, %d, %d, %d, %d):   key =%llx   integral=%f\n",n,l,np,lp,J,key,integral);
        exit(EXIT_FAILURE);
      }
      IntList[key] = integral;
      
      return integral;
    }
  }

  Operator GamowTeller_R(ModelSpace& modelspace, double Eclosure, double r12)
  {
    bool reduced = true;
    r12 =  r12*SQRT2;
    double t_start, t_start_tbme, t_start_omp; // profiling (v)
    t_start = omp_get_wtime(); // profiling (s)
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    int e2max = modelspace.GetE2max(); // 2*emax
    Operator M0nuGT_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    M0nuGT_TBME.SetHermitian(); // it should be Hermitian
    int Anuc = modelspace.GetTargetMass(); // the mass number for the desired nucleus
    const double Rnuc = R0*pow(Anuc,1.0/3.0); // the nuclear radius [fm]
    const double prefact = 4*Rnuc/(PI); // factor in-front of M0nu TBME, extra global 2 from <p|\tau|n> = sqrt(2) [fm]
    modelspace.PreCalculateMoshinsky(); // pre-calculate the needed Moshinsky brackets, for efficiency
    std::unordered_map<uint64_t,double> IntList = PreCalculateM0nuIntegrals_R(e2max, hw, Eclosure, r12); // pre-calculate the needed integrals over dq and dr, for efficiency
    M0nuGT_TBME.profiler.timer["M0nuGT_1_sur"] += omp_get_wtime() - t_start; // profiling (r)
    // create the TBMEs of M0nu
    // auto loops over the TBME channels and such
    std::cout<<"calculating M0nu TBMEs..."<<std::endl;
    t_start_tbme = omp_get_wtime(); // profiling (s)
    for (auto& itmat : M0nuGT_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      double Jhat; // set below based on "reduced" variable
      if (reduced == false)
      {
        Jhat = 1.0; // for non-reduced elements, to compare with JE
      }
      else //if (reduced == "true")
      { 
        Jhat = sqrt(2*J+1); // the hat factor of J
      }
      t_start_omp = omp_get_wtime(); // profiling (s)
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          int eps_ab = 2*na + la + 2*nb + lb; // for conservation of energy in the Moshinsky brackets
          int eps_cd = 2*nc + lc + 2*nd + ld; // for conservation of energy in the Moshinsky brackets
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          for (int S=0; S<=1; S++) // sum over total spin...
          {
            int Seval;
            Seval = 2*S*(S + 1) - 3;
            int L_min = std::max(abs(la-lb),abs(lc-ld));
            int L_max = std::min(la+lb,lc+ld);
            for (int L = L_min; L <= L_max; L++) // ...and sum over orbital angular momentum
            {
              double sumMT = 0; // for the Moshinsky transformation
              double sumMTas = 0; // (anti-symmetric part)              
              double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
              double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
              double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
              double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
              double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
              double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
              double bulk = Seval*nNJab*nNJcd; // bulk product of the above
              double bulkas = Seval*nNJab*nNJdc; // (anti-symmetric part)
              int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
              for (int nr = 0; nr <= tempmaxnr; nr++)
              {
                double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of Charlie's thesis
                if ((npr >= 0) and ((eps_cd-eps_ab)%2 == 0))
                {
                  int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                  for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                  {
                    int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                    int tempmaxlr = floor((eps_ab + L)/2.0) - (nr + Ncom); // " " " " "
                    for (int lr = tempminlr; lr <= tempmaxlr; lr++)
                    {
                      int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of Charlie's thesis
                      double integral = 0;
                      double normJrel;
                      double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                      double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                      double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                      int minJrel= abs(lr-S);
                      int maxJrel = lr+S;
                      for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                      {
                        normJrel = sqrt((2*Jrel+1)*(2*L+1))*phase(L+lr+S+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                        integral += normJrel*normJrel*GetM0nuIntegral_R(e2max,nr,lr,npr,lr,Jrel,hw,Eclosure,r12,IntList); // grab the pre-calculated integral wrt dq from the IntList of the modelspace class
                      }//end of for-loop over Jrel
                      sumMT += Df*Di*integral; // perform the Moshinsky transformation
                      sumMTas += Df*asDi*integral; // (anti-symmetric part)
                    } // end of for-loop over: lr
                  } // end of for-loop over: Ncom
                } // end of if: npr \in \Nat_0
              } // end of for-loop over: nr
              sumLS += bulk*sumMT; // perform the LS-coupling sum
              sumLSas += bulkas*sumMTas; // (anti-symmetric part)
            } // end of for-loop over: L
          } // end of for-loop over: S
          double Mtbme = asNorm(ia,ib)*asNorm(ic,id)*prefact*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas); // compute the final matrix element, anti-symmetrize
          M0nuGT_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      } // end of for-loop over: ibra
      M0nuGT_TBME.profiler.timer["M0nuGT_3_omp"] += omp_get_wtime() - t_start_omp; // profiling (r)
    } // end of for-loop over: auto
    std::cout<<"...done calculating M0nu TBMEs"<<std::endl;
    M0nuGT_TBME.profiler.timer["M0nuGT_2_tbme"] += omp_get_wtime() - t_start_tbme; // profiling (r)
    M0nuGT_TBME.profiler.timer["M0nuGT_Op"] += omp_get_wtime() - t_start; // profiling (r)
    return M0nuGT_TBME;
  }


  Operator DGT_R(ModelSpace& modelspace, double r12)
  {
    r12 = r12*SQRT2;
    Operator DGT_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    DGT_TBME.SetHermitian(); // it should be Hermitian
    modelspace.PreCalculateMoshinsky();
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    // create the TBMEs of DGT
    // auto loops over the TBME channels and such
    std::cout<<"calculating DGT TBMEs..."<<std::endl;
    for (auto& itmat : DGT_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      // double Jhat = 1;
      double Jhat = sqrt(2*J + 1); // the hat factor of J
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          int eps_ab = 2*na+la+2*nb+lb;
          int eps_cd = 2*nc+lc+2*nd+ld;
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          if (la+lb == lc+ld)
          {
            for (int S=0; S<=1; S++) // sum over total spin...
            {
              int Seval;
              Seval = 2*S*(S + 1) - 3;
              int L_min = std::max(abs(la-lb),abs(lc-ld));
              int L_max = std::min(la+lb,lc+ld);
              for (int L = L_min; L <= L_max; L++) // ...and sum over orbital angular momentum
              {
                double sumMT = 0; // for the Moshinsky transformation
                double sumMTas = 0; // (anti-symmetric part)              
                double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
                double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
                double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
                double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
                double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
                double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
                double bulk = Seval*nNJab*nNJcd; // bulk product of the above
                double bulkas = Seval*nNJab*nNJdc; // (anti-symmetric part)
                int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
                for (int nr = 0; nr <= tempmaxnr; nr++)
                {
                  double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of my thesis
                  // if ((npr >= 0) and ((eps_cd-eps_ab)%2 == 0))
                  if (npr==nr)
                  {
                    int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                    for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                    {
                      int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                      int tempmaxlr = floor((eps_ab + L)/2.0) - (nr + Ncom); // " " " " "
                      for (int lr = tempminlr; lr <= tempmaxlr; lr++)
                      {
                        int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of Charlie's thesis
                        double integral = 0;
                        double normJrel;
                        double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                        double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                        double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                        int minJrel= abs(lr-S);
                        int maxJrel = lr+S;
                        for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                        {
                          normJrel = sqrt((2*Jrel+1)*(2*L+1))*phase(Lam+lr+S+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                          integral += normJrel*normJrel*r12*r12*HO_Radial_psi(nr, lr, hw, r12)*HO_Radial_psi(npr, lr, hw, r12);
                        }//end of for-loop over Jrel
                        sumMT += Df*Di*integral; // perform the Moshinsky transformation
                        sumMTas += Df*asDi*integral; // (anti-symmetric part)
                      } // end of for-loop over: lr
                    } // end of for-loop over: Ncom
                  } // end of if: npr \in \Nat_0
                } // end of for-loop over: nr
                sumLS += bulk*sumMT; // perform the LS-coupling sum
                sumLSas += bulkas*sumMTas; // (anti-symmetric part)
              } // end of for-loop over: L
            } // end of for-loop over: S
          }//end of if la+lb==lc+ld
          
          double Mtbme = 2*asNorm(ia,ib)*asNorm(ic,id)*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas)/sqrt(3); // compute the final matrix element, anti-symmetrize
          DGT_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      }// end of for-loop over: ibra
    } // end of for-loop over: auto
    std::cout<<"...done calculating DGT TBMEs"<<std::endl;
    return DGT_TBME;
  }


  /// Calculate B coefficient for Talmi integral. Formula given in Brody and Moshinsky
  /// "Tables of Transformation Brackets for Nuclear Shell-Model Calculations"
  long double TalmiB(int na, int la, int nb, int lb, int p)
  {
   if ( (la+lb)%2>0 ) return 0;
   
   int q = (la+lb)/2;

   if ( std::max(na+la+p, nb+lb+p) < 10 )
   {
     return AngMom::TalmiB( na, la, nb, lb, p);
   }
   long double logB1 = (lgamma(2*p+2)-lgamma(p+1) +0.5*(  lgamma(na+1)+lgamma(nb+1)-lgamma(na+la+1)-lgamma(nb+lb+1) + lgamma(2*na+2*la+2) + lgamma(2*nb+2*lb+2)) - (na+nb)*LOG2   ) ;
   long double B2 = 0;
   int kmin = std::max(0, p-q-nb);
   int kmax = std::min(na, p-q);
   for (int k=kmin;k<=kmax;++k)
   {
      B2  += exp(logB1+lgamma(la+k+1)-lgamma(k+1) +lgamma(p-(la-lb)/2-k+1) -lgamma(2*p-la+lb-2*k+2) 
                - lgamma(2*la+2*k+2) -lgamma(na-k+1)  
                - lgamma(nb - p + q + k+1) - lgamma(p-q-k+1) );
   }
   
   return AngMom::phase(p-q) *  B2;
  }

  double RadialIntegral_Gauss( int na, int la, int nb, int lb, double sigma )
  {
   long double I = 0;
   int pmin = (la+lb)/2;
   int pmax = pmin + na + nb;
   double kappa = 1.0 / (2*sigma*sigma); // Gaussian exp( -x^2 / (2 sigma^2)) =>  exp(-kappa x^2)

     for (int p=pmin;p<=pmax;++p)
     {
        I += TalmiB(na,la,nb,lb,p) * pow(1+kappa, -(p+1.5) )  ; // Talmi integral is analytic
     }
   return I;
  }

 
  double integrateRcom(int Ncom, int  Lambda, double  hw, double Rnucl)
  {
    double I = 1-RadialIntegral_Gauss(Ncom,Lambda,Ncom,Lambda,Rnucl);
    return I;
  }

  std::unordered_map<uint64_t,double> PreCalculateDGTRComIntegrals(int e2max, double hw, double Rnucl)
  {
    IMSRGProfiler profiler;
    double t_start_pci = omp_get_wtime(); // profiling (s)
    std::unordered_map<uint64_t,double> IntList;
    int size=500;
    std::cout<<"calculating integrals wrt dRcom..."<<std::endl;
    int maxn = e2max/2;
    int maxl = e2max;
    std::vector<uint64_t> KEYS;
    for (int n=0; n<=maxn; n++)
    {
      for (int l=0; l<=maxl-2*n; l++)
      {
        uint64_t key = IntHash(n,l,n,l,0);
        KEYS.push_back(key);
        IntList[key] = 0.0; // "Make sure eveything's in there to avoid a rehash in the parallel loop" (RS)
      }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i=0; i<KEYS.size(); i++)
    {
      uint64_t key = KEYS[i];
      uint64_t n,l,np,lp,J;
      IntUnHash(key, n,l,np,lp,J);
      IntList[key] = integrateRcom(n,l,hw,Rnucl);// these have been ordered by the above loops such that we take the "lowest" value of decimalgen(n,l,np,lp,maxl,maxnp,maxlp), see GetIntegral(...)
      
    }
    
    std::cout<<"...done calculating the integrals"<<std::endl;
    std::cout<<"IntList has "<<IntList.bucket_count()<<" buckets and a load factor "<<IntList.load_factor()
      <<", estimated storage ~= "<<((IntList.bucket_count() + IntList.size())*(sizeof(size_t) + sizeof(void*)))/(1024.0*1024.0*1024.0)<<" GB"<<std::endl; // copied from (RS)
    profiler.timer["PreCalculateM0nuIntegrals"] += omp_get_wtime() - t_start_pci; // profiling (r)
    return IntList;
  }

  //Get an integral from the IntList cache or calculate it (parallelization dependent)
  double GetDGTRcomIntegral(int Ncom, int Lam, double hw, double Rnucl, std::unordered_map<uint64_t,double> &IntList)
  {
    uint64_t key = IntHash(Ncom,Lam,Ncom,Lam,0);
    auto it = IntList.find(key);
    if (it != IntList.end()) // return what we've found
    {
      return it -> second;
    }
    else // if we didn't find it, calculate it and add it to the list!
    {
      double integral;
      int size = 500;
      integral = integrateRcom(Ncom,Lam,hw,Rnucl);
      if (omp_get_num_threads() >= 2)
      {
        printf("DANGER!!!!!!!  Updating IntList inside a parellel loop breaks thread safety!\n");
        printf("   I shouldn't be here in GetIntegral(%d, %d):   key =%llx   integral=%f\n",Ncom,Lam,key,integral);
        exit(EXIT_FAILURE);
      }
      IntList[key] = integral;
      
      return integral;
    }
  }

  Operator DGT_R_SurfaceLocalization(ModelSpace& modelspace, double r12)
  {
    r12 = r12*SQRT2;
    Operator DGT_TBME(modelspace,0,2,0,2); // NOTE: from the constructor -- Operator::Operator(ModelSpace& ms, int Jrank, int Trank, int p, int part_rank)
    DGT_TBME.SetHermitian(); // it should be Hermitian
    modelspace.PreCalculateMoshinsky();
    double hw = modelspace.GetHbarOmega(); // oscillator basis frequency [MeV]
    int Anuc = modelspace.GetTargetMass(); 
    const double Rnuc = R0*pow(Anuc,1.0/3.0); // the nuclear radius [fm]const
    int e2max = modelspace.GetE2max(); // 2*emax
    std::unordered_map<uint64_t,double> IntList = PreCalculateDGTRComIntegrals(e2max,hw,Rnuc);
    // create the TBMEs of DGT
    // auto loops over the TBME channels and such
    std::cout<<"calculating DGT TBMEs..."<<std::endl;
    for (auto& itmat : DGT_TBME.TwoBody.MatEl)
    {
      int chbra = itmat.first[0]; // grab the channel count from auto
      int chket = itmat.first[1]; // " " " " " "
      TwoBodyChannel& tbc_bra = modelspace.GetTwoBodyChannel(chbra); // grab the two-body channel
      TwoBodyChannel& tbc_ket = modelspace.GetTwoBodyChannel(chket); // " " " " "
      int nbras = tbc_bra.GetNumberKets(); // get the number of bras
      int nkets = tbc_ket.GetNumberKets(); // get the number of kets
      int J = tbc_bra.J; // NOTE: by construction, J := J_ab == J_cd := J'
      // double Jhat = 1;
      double Jhat = sqrt(2*J + 1); // the hat factor of J
      #pragma omp parallel for schedule(dynamic,1) // need to do: PreCalculateMoshinsky() and PreCalcIntegrals() [above] and then "#pragma omp critical" [below]
      for (int ibra=0; ibra<nbras; ibra++)
      {
        Ket& bra = tbc_bra.GetKet(ibra); // get the final state = <ab|
        int ia = bra.p; // get the integer label a
        int ib = bra.q; // get the integer label b
        Orbit& oa = modelspace.GetOrbit(ia); // get the <a| state orbit
        Orbit& ob = modelspace.GetOrbit(ib); // get the <b| state prbit
        for (int iket=0; iket<nkets; iket++)
        {
          Ket& ket = tbc_ket.GetKet(iket); // get the initial state = |cd>
          int ic = ket.p; // get the integer label c
          int id = ket.q; // get the integer label d
          Orbit& oc = modelspace.GetOrbit(ic); // get the |c> state orbit
          Orbit& od = modelspace.GetOrbit(id); // get the |d> state orbit
          int na = oa.n; // this is just...
          int nb = ob.n;
          int nc = oc.n;
          int nd = od.n;
          int la = oa.l;
          int lb = ob.l;
          int lc = oc.l;
          int ld = od.l;
          int eps_ab = 2*na+la+2*nb+lb;
          int eps_cd = 2*nc+lc+2*nd+ld;
          double ja = oa.j2/2.0;
          double jb = ob.j2/2.0;
          double jc = oc.j2/2.0;
          double jd = od.j2/2.0; // ...for convenience
          double sumLS = 0; // for the Bessel's Matrix Elemets (BMEs)
          double sumLSas = 0; // (anti-symmetric part)
          if (la+lb == lc+ld)
          {
            for (int S=0; S<=1; S++) // sum over total spin...
            {
              int Seval;
              Seval = 2*S*(S + 1) - 3;
              int L_min = std::max(abs(la-lb),abs(lc-ld));
              int L_max = std::min(la+lb,lc+ld);
              for (int L = L_min; L <= L_max; L++) // ...and sum over orbital angular momentum
              {
                double sumMT = 0; // for the Moshinsky transformation
                double sumMTas = 0; // (anti-symmetric part)              
                double tempLS = (2*L + 1)*(2*S + 1); // just for efficiency, only used in the three lines below
                double normab = sqrt(tempLS*(2*ja + 1)*(2*jb + 1)); // normalization factor for the 9j-symbol out front
                double nNJab = normab*AngMom::NineJ(la,lb,L,0.5,0.5,S,ja,jb,J); // the normalized 9j-symbol out front
                double normcd = sqrt(tempLS*(2*jc + 1)*(2*jd + 1)); // normalization factor for the second 9j-symbol
                double nNJcd = normcd*AngMom::NineJ(lc,ld,L,0.5,0.5,S,jc,jd,J); // the second normalized 9j-symbol
                double nNJdc = normcd*AngMom::NineJ(ld,lc,L,0.5,0.5,S,jd,jc,J); // (anti-symmetric part)
                double bulk = Seval*nNJab*nNJcd; // bulk product of the above
                double bulkas = Seval*nNJab*nNJdc; // (anti-symmetric part)
                int tempmaxnr = floor((eps_ab - L)/2.0); // just for the limits below
                for (int nr = 0; nr <= tempmaxnr; nr++)
                {
                  double npr = ((eps_cd - eps_ab)/2.0) + nr; // via Equation (4.73) of my thesis
                  // if ((npr >= 0) and ((eps_cd-eps_ab)%2 == 0))
                  if (npr==nr)
                  {
                    int tempmaxNcom = tempmaxnr - nr; // just for the limits below
                    for (int Ncom = 0; Ncom <= tempmaxNcom; Ncom++)
                    {
                      int tempminlr = ceil((eps_ab - L)/2.0) - (nr + Ncom); // just for the limits below
                      int tempmaxlr = floor((eps_ab + L)/2.0) - (nr + Ncom); // " " " " "
                      for (int lr = tempminlr; lr <= tempmaxlr; lr++)
                      {
                        int Lam = eps_ab - 2*(nr + Ncom) - lr; // via Equation (4.73) of Charlie's thesis
                        double integral = 0;
                        double normJrel;
                        double Rcomintegral;
                        double Df = modelspace.GetMoshinsky(Ncom,Lam,nr,lr,na,la,nb,lb,L); // Ragnar has -- double mosh_ab = modelspace.GetMoshinsky(N_ab,Lam_ab,n_ab,lam_ab,na,la,nb,lb,Lab);
                        double Di = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nc,lc,nd,ld,L); // " " " "
                        double asDi = modelspace.GetMoshinsky(Ncom,Lam,npr,lr,nd,ld,nc,lc,L); // (anti-symmetric part)
                        int minJrel= abs(lr-S);
                        int maxJrel = lr+S;
                        for (int Jrel = minJrel; Jrel<=maxJrel; Jrel++)
                        {
                          normJrel = sqrt((2*Jrel+1)*(2*L+1))*phase(Lam+lr+S+J)*AngMom::SixJ(Lam,lr,L,S,J,Jrel);
                          Rcomintegral = GetDGTRcomIntegral(Ncom, Lam, hw, Rnuc, IntList);
                          integral += normJrel*normJrel*r12*r12*HO_Radial_psi(nr, lr, hw, r12)*HO_Radial_psi(npr, lr, hw, r12)*exp(-r12*r12*0.5)*Rcomintegral;
                        }//end of for-loop over Jrel
                        sumMT += Df*Di*integral; // perform the Moshinsky transformation
                        sumMTas += Df*asDi*integral; // (anti-symmetric part)
                      } // end of for-loop over: lr
                    } // end of for-loop over: Ncom
                  } // end of if: npr \in \Nat_0
                } // end of for-loop over: nr
                sumLS += bulk*sumMT; // perform the LS-coupling sum
                sumLSas += bulkas*sumMTas; // (anti-symmetric part)
              } // end of for-loop over: L
            } // end of for-loop over: S
          }//end of if la+lb==lc+ld
          
          double Mtbme = 2*asNorm(ia,ib)*asNorm(ic,id)*Jhat*(sumLS - modelspace.phase(jc + jd - J)*sumLSas)/sqrt(3); // compute the final matrix element, anti-symmetrize
          DGT_TBME.TwoBody.SetTBME(chbra,chket,ibra,iket,Mtbme); // set the two-body matrix elements (TBME) to Mtbme
        } // end of for-loop over: iket
      }// end of for-loop over: ibra
    } // end of for-loop over: auto
    std::cout<<"...done calculating DGT TBMEs"<<std::endl;
    return DGT_TBME;
  }
}// end namespace M0nu


