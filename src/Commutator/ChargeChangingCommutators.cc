// J=0 signed-charge IMSRG(2) commutators. See Commutator.hh for conventions.
#include "Commutator.hh"
#include "AngMom.hh"
#include <array>
#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>

namespace Commutator
{
namespace
{

struct PairLocation
{
  int channel=-1, index=-1;
  double factor=0;
};

struct TensorGeometry
{
  // Geometry is local to one commutator. Pair lookup stores the native
  // antisymmetry phase and the sqrt(1+delta_pq) needed by GetTBME_J.
  ModelSpace& ms;
  int n, nch, jmax, nj;
  std::vector<int> l,j2,tz;
  std::vector<double> occ;
  std::vector<PairLocation> locations;
  std::map<std::array<int,2>,std::vector<int>> sectors;
  TensorGeometry(ModelSpace& space,bool normal_ordered):ms(space)
  {
    n=ms.GetNumberOrbits(); nch=ms.GetNumberTwoBodyChannels(); jmax=0;
    for(int i=0;i<n;++i)
    {
      const auto& o=ms.GetOrbit(i);
      l.push_back(o.l); j2.push_back(o.j2); tz.push_back(o.tz2);
      occ.push_back(normal_ordered?o.occ:0.);
      jmax=std::max(jmax,o.j2);
    }
    nj=(jmax+1)/2;
    locations.resize((jmax+1)*n*n);
    for(int ch=0;ch<nch;++ch)
    {
      auto& c=ms.GetTwoBodyChannel(ch);
      if(c.GetNumberKets()==0) continue;
      sectors[{c.J,c.parity}].push_back(ch);
      for(int r=0;r<c.GetNumberKets();++r)
      {
        auto& k=c.GetKet(r);
        const double f=std::sqrt(1.+(k.p==k.q));
        location(c.J,k.p,k.q)={ch,r,f};
        if(k.p!=k.q) location(c.J,k.q,k.p)={ch,r,f*k.Phase(c.J)};
      }
    }
  }
  PairLocation& location(int J,int i,int j) {return locations[(J*n+i)*n+j];}
  const PairLocation& location(int J,int i,int j)const {return locations[(J*n+i)*n+j];}
};

// Borrow all native coefficient matrices. rank_T is an absolute charge;
// q selects the signed component. Positive charge uses the stored adjoint
// two-body blocks; its one-body matrix already has the physical direction.
// Neutral inputs/outputs keep both triangles, including non-Hermitian data.
struct ChargeView
{
  const Operator& op;
  const TensorGeometry& g;
  int q;
  bool reduced;
  std::vector<const arma::mat*> matrices;
  arma::mat one;
  ChargeView(const Operator& input,int charge,const TensorGeometry& geom)
    :op(input),g(geom),q(charge),reduced(input.IsReduced()),matrices(g.nch*g.nch,nullptr),
     one(input.OneBody)
  {
    for(const auto& block:op.TwoBody.MatEl)
    {
      const int a=block.first[0], b=block.first[1];
      matrices[a*g.nch+b]=&block.second;
      matrices[b*g.nch+a]=&block.second;
    }
    for(int i=0;i<g.n;++i) for(int j=0;j<g.n;++j)
    {
      if(g.tz[i]-g.tz[j]!=2*q) one(i,j)=0.;
      else if(reduced) one(i,j)/=std::sqrt(g.j2[i]+1.);
    }
  }
  const arma::mat* block(int a,int b)const
  {
    const auto& ca=g.ms.GetTwoBodyChannel(a);
    const auto& cb=g.ms.GetTwoBodyChannel(b);
    if(ca.J!=cb.J || ca.parity!=cb.parity || ca.Tz-cb.Tz!=q) return nullptr;
    return matrices[a*g.nch+b];
  }
  arma::mat matrix(int a,int b)const
  {
    const auto* ptr=block(a,b);
    if(!ptr) return arma::mat(g.ms.GetTwoBodyChannel(a).GetNumberKets(),
                              g.ms.GetTwoBodyChannel(b).GetNumberKets(),arma::fill::zeros);
    arma::mat out=a<=b?arma::mat(*ptr):arma::mat(ptr->t());
    if(reduced) out/=std::sqrt(2*g.ms.GetTwoBodyChannel(a).J+1.);
    return out;
  }
  double two(int J,int i,int j,int k,int l)const
  {
    if(g.tz[i]+g.tz[j]-g.tz[k]-g.tz[l]!=2*q) return 0.;
    const auto& a=g.location(J,i,j); const auto& b=g.location(J,k,l);
    if(a.channel<0 || b.channel<0) return 0.;
    const auto* ptr=block(a.channel,b.channel);
    if(!ptr) return 0.;
    double value=a.channel<=b.channel?(*ptr)(a.index,b.index):(*ptr)(b.index,a.index);
    value*=a.factor*b.factor;
    if(reduced) value/=std::sqrt(2*J+1.);
    return value;
  }
};

// Sparse representation of the one-body operator acting on normalized
// antisymmetric two-particle states. This replaces the N-orbit Python loop
// for EVERY output matrix element in 122 with sparse/dense C++ products.
arma::sp_mat pair_one_body(const ChargeView& V,int bra,int ket)
{
  const auto& g=V.g;
  auto& cb=g.ms.GetTwoBodyChannel(bra); auto& ck=g.ms.GetTwoBodyChannel(ket);
  arma::sp_mat out(cb.GetNumberKets(),ck.GetNumberKets());
  if(cb.Tz-ck.Tz!=V.q) return out;
  for(int r=0;r<cb.GetNumberKets();++r)
  {
    auto& b=cb.GetKet(r); const int i=b.p,j=b.q;
    for(int s=0;s<ck.GetNumberKets();++s)
    {
      const auto& k=ck.GetKet(s); const int a=k.p,c=k.q;
      double value=(j==c?V.one(i,a):0.)+(i==a?V.one(j,c):0.);
      if(i!=j) value+=b.Phase(cb.J)*((i==c?V.one(j,a):0.)+(j==a?V.one(i,c):0.));
      if(value!=0.) out(r,s)=value*std::sqrt((1.+(i==j))/(1.+(a==c)));
    }
  }
  return out;
}

// Angular coefficients depend on j, not on n or proton/neutron labels.
// Build this small table ONCE per K, outside all radial matrix-element loops.
struct PandyaAngles
{
  const TensorGeometry& g;
  int K;
  std::vector<double> values;
  PandyaAngles(const TensorGeometry& geom,int rank):g(geom),K(rank),
    values(g.nj*g.nj*g.nj*g.nj*(g.jmax+1),0.)
  {
    for(int a=0;a<g.nj;++a) for(int b=0;b<g.nj;++b)
    {
      const int ja=2*a+1,jb=2*b+1;
      if(2*K<std::abs(ja-jb) || 2*K>ja+jb) continue;
      for(int c=0;c<g.nj;++c) for(int d=0;d<g.nj;++d)
      {
        const int jc=2*c+1,jd=2*d+1;
        if(2*K<std::abs(jc-jd) || 2*K>jc+jd) continue;
        int lo=std::max(std::abs(ja-jd),std::abs(jc-jb))/2;
        int hi=std::min(ja+jd,jc+jb)/2;
        for(int J=lo;J<=hi;++J)
          at(a,b,c,d,J)=-(2*J+1)*g.ms.GetSixJ(ja*.5,jb*.5,K,jc*.5,jd*.5,J);
      }
    }
  }
  double& at(int a,int b,int c,int d,int J)
  {return values[((((a*g.nj+b)*g.nj+c)*g.nj+d)*(g.jmax+1))+J];}
  double get(int i,int l,int a,int b,int J)const
  {return values[((((g.j2[i]/2*g.nj+g.j2[l]/2)*g.nj+g.j2[a]/2)*g.nj+g.j2[b]/2)*(g.jmax+1))+J];}
};

using PHPair=std::array<int,2>;
arma::mat pandya(const ChargeView& V,const std::vector<PHPair>& rows,
                 const std::vector<PHPair>& cols,const PandyaAngles& angles)
{
  const auto& g=V.g;
  arma::mat out(rows.size(),cols.size(),arma::fill::zeros);
  #pragma omp parallel for schedule(dynamic,8) if(rows.size()*cols.size()>4096)
  for(size_t r=0;r<rows.size();++r)
  {
    int i=rows[r][0],l=rows[r][1];
    for(size_t s=0;s<cols.size();++s)
    {
      int a=cols[s][0],b=cols[s][1];
      int lo=std::max(std::abs(g.j2[i]-g.j2[b]),std::abs(g.j2[a]-g.j2[l]))/2;
      int hi=std::min(g.j2[i]+g.j2[b],g.j2[a]+g.j2[l])/2;
      double value=0.;
      for(int J=lo;J<=hi;++J)
      {
        const double coeff=angles.get(i,l,a,b,J);
        if(coeff!=0.) value+=coeff*V.two(J,i,b,a,l);
      }
      out(r,s)=value;
    }
  }
  return out;
}

} // anonymous namespace

Operator CommutatorScalarTensor(const Operator& Xin,const Operator& Yin,
                           int qX,int qY,bool normal_ordered)
{
  if(Xin.GetJRank()!=0 || Yin.GetJRank()!=0 || Xin.GetParity()!=0 || Yin.GetParity()!=0)
    throw std::invalid_argument("Expected parity-even J=0 inputs");
  if(Xin.IsReduced() || !Yin.IsReduced())
    throw std::invalid_argument("Use ordinary X and angular-J-reduced Y");
  if(Xin.GetTRank()!=std::abs(qX) || Yin.GetTRank()!=std::abs(qY)
     || Xin.GetParticleRank()>2 || Yin.GetParticleRank()>2 || qX+qY>0 || qX+qY< -2)
    throw std::invalid_argument("Inconsistent signed charge or NO2B rank");
  auto& ms=*Xin.GetModelSpace();
  if(Yin.GetModelSpace()!=&ms) throw std::invalid_argument("Inputs must share one ModelSpace");
  TensorGeometry g(ms,normal_ordered);
  ChargeView X(Xin,qX,g),Y(Yin,qY,g);
  int qZ=qX+qY;
  Operator Z(ms,0,std::abs(qZ),0,2);
  if(Z.IsReduced()) Z.MakeNotReduced();
  Z.SetNonHermitian(); // Do not infer either neutral triangle.
  arma::mat one=X.one*Y.one-Y.one*X.one;
  // 110: opposite charges can give a nonzero scalar even though neither
  // operand is individually charge neutral. No Hermiticity shortcut is used.
  if(normal_ordered && qZ==0)
    for(int a=0;a<g.n;++a) for(int b=0;b<g.n;++b)
      Z.ZeroBody+=(g.j2[a]+1)*g.occ[a]*(1-g.occ[b])
        *(X.one(a,b)*Y.one(b,a)-Y.one(a,b)*X.one(b,a));
  std::vector<PHPair> active_one,output_one;
  for(int a=0;a<g.n;++a) for(int b=0;b<g.n;++b)
  {
    if(g.occ[a]!=g.occ[b] && (X.one(a,b)!=0. || Y.one(a,b)!=0.)) active_one.push_back({a,b});
    if(g.l[a]==g.l[b] && g.j2[a]==g.j2[b] && g.tz[a]-g.tz[b]==2*qZ) output_one.push_back({a,b});
  }
  // 121: occupation differences eliminate inactive intermediate pairs.
  if(normal_ordered)
  {
    #pragma omp parallel for schedule(dynamic)
    for(size_t u=0;u<output_one.size();++u)
    {
      int i=output_one[u][0],j=output_one[u][1];
      for(const auto& ab:active_one)
      {
        int a=ab[0],b=ab[1];
        int lo=std::max(std::abs(g.j2[b]-g.j2[i]),std::abs(g.j2[a]-g.j2[j]))/2;
        int hi=std::min(g.j2[b]+g.j2[i],g.j2[a]+g.j2[j])/2;
        for(int J=lo;J<=hi;++J)
          one(i,j)+=(g.occ[a]-g.occ[b])*(2*J+1)/(g.j2[i]+1.)
            *(X.one(a,b)*Y.two(J,b,i,a,j)-Y.one(a,b)*X.two(J,b,i,a,j));
      }
    }
  }

  // 122, 222pp/hh, 220 and 221 share normalized pair-space matrices.
  // Sum over EVERY intermediate Tz compatible with the signed charges,
  // rather than fixing it to the external bra/ket as the neutral-X kernel
  // does. Hold only one (J,parity) sector and one output pp/hh block.
  for(const auto& sector:g.sectors)
  {
    int J=sector.first[0];
    const auto& ids=sector.second;
    std::map<std::array<int,2>,arma::mat> xb,yb;
    std::map<std::array<int,2>,arma::sp_mat> lx,ly;
    for(int a:ids) for(int b:ids)
    {
      if(X.block(a,b)) xb[{a,b}]=X.matrix(a,b);
      if(Y.block(a,b)) yb[{a,b}]=Y.matrix(a,b);
      const int charge=ms.GetTwoBodyChannel(a).Tz-ms.GetTwoBodyChannel(b).Tz;
      if(charge==qX) lx[{a,b}]=pair_one_body(X,a,b);
      if(charge==qY) ly[{a,b}]=pair_one_body(Y,a,b);
    }
    for(int bra:ids) for(int ket:ids)
    {
      auto& cb=ms.GetTwoBodyChannel(bra); auto& ck=ms.GetTwoBodyChannel(ket);
      if(cb.Tz-ck.Tz!=qZ) continue;
      arma::mat pp(cb.GetNumberKets(),ck.GetNumberKets(),arma::fill::zeros),hh=pp,comm122=pp;
      for(int mid:ids)
      {
        auto& cm=ms.GetTwoBodyChannel(mid);
        arma::rowvec wp(cm.GetNumberKets()),wh(cm.GetNumberKets());
        for(int u=0;u<cm.GetNumberKets();++u)
        {
          const auto& ab=cm.GetKet(u);
          wp(u)=(1-g.occ[ab.p])*(1-g.occ[ab.q]); wh(u)=g.occ[ab.p]*g.occ[ab.q];
        }
        const std::array<int,2> a{{bra,mid}}, b{{mid,ket}};
        if(xb.count(a) && yb.count(b))
        {
          pp+=(xb.at(a).each_row()%wp)*yb.at(b);
          if(normal_ordered) hh+=(xb.at(a).each_row()%wh)*yb.at(b);
        }
        if(yb.count(a) && xb.count(b))
        {
          pp-=(yb.at(a).each_row()%wp)*xb.at(b);
          if(normal_ordered) hh-=(yb.at(a).each_row()%wh)*xb.at(b);
        }
        if(lx.count(a) && yb.count(b)) comm122+=lx.at(a)*yb.at(b);
        if(xb.count(a) && ly.count(b)) comm122+=xb.at(a)*ly.at(b);
        if(ly.count(a) && xb.count(b)) comm122-=ly.at(a)*xb.at(b);
        if(yb.count(a) && lx.count(b)) comm122-=yb.at(a)*lx.at(b);
      }
      if(normal_ordered)
      {
        if(qZ==0 && bra==ket)
          // 220 = sum_ab n_a n_b (2J+1) Mpp_ab,ab. Canonical normalized
          // pair sums already contain the unrestricted-sum symmetry factors.
          for(int r=0;r<cb.GetNumberKets();++r)
          {
            auto& ab=cb.GetKet(r);
            Z.ZeroBody+=(2*J+1)*g.occ[ab.p]*g.occ[ab.q]*pp(r,r);
          }
        for(const auto& ij:output_one)
        {
          int i=ij[0],j=ij[1];
          for(int c=0;c<g.n;++c)
          {
            const auto& a=g.location(J,c,i); const auto& b=g.location(J,c,j);
            if(a.channel==bra && b.channel==ket)
              // 221: n_c Mpp + (1-n_c) Mhh gives both retained reference
              // contractions. Do not drop this term for a two-body ladder.
              one(i,j)+=(2*J+1)/(g.j2[i]+1.)*a.factor*b.factor
                *(g.occ[c]*pp(a.index,b.index)+(1-g.occ[c])*hh(a.index,b.index));
          }
        }
      }
      if(bra<=ket) Z.TwoBody.MatEl.at({size_t(bra),size_t(ket)})=comm122+pp-hh;
    }
  }
  Z.OneBody=one;
  // Mask one-body outputs explicitly; a rank-2 charge has no 1b coefficient.
  for(int i=0;i<g.n;++i) for(int j=0;j<g.n;++j)
    if(g.l[i]!=g.l[j] || g.j2[i]!=g.j2[j] || g.tz[i]-g.tz[j]!=2*qZ) Z.OneBody(i,j)=0.;

  if(normal_ordered && Xin.TwoBodyNorm()!=0. && Yin.TwoBodyNorm()!=0.)
    // 222ph: ordered particle-hole pairs retain SIGNED Tz differences.
    // Only intermediate pairs with n_a != n_b participate. Release each
    // (K,parity) sector after inverse recoupling into the output operator.
    for(int K=0;K<=g.jmax;++K)
    {
      PandyaAngles angles(g,K);
      for(int parity=0;parity<=1;++parity)
      {
        std::array<std::vector<PHPair>,3> pairs,active;
        std::vector<int> index(g.n*g.n,-1);
        for(int i=0;i<g.n;++i) for(int l=0;l<g.n;++l)
        {
          if((g.l[i]+g.l[l])%2!=parity || 2*K<std::abs(g.j2[i]-g.j2[l]) || 2*K>g.j2[i]+g.j2[l]) continue;
          int d=(g.tz[i]-g.tz[l])/2+1;
          index[i*g.n+l]=pairs[d].size(); pairs[d].push_back({i,l});
          if(g.occ[i]!=g.occ[l]) active[d].push_back({i,l});
        }
        std::array<std::array<arma::mat,3>,3> ph;
        for(int d=0;d<3;++d) for(int e=0;e<3;++e)
        {
          if(d-e!=qZ || pairs[d].empty() || pairs[e].empty()) continue;
          ph[d][e].zeros(pairs[d].size(),pairs[e].size());
          for(int order=0;order<2;++order)
          {
            const auto& A=order==0?X:Y; const auto& B=order==0?Y:X;
            int mid=d-A.q;
            if(mid<0 || mid>=3 || mid-e!=B.q || active[mid].empty()) continue;
            arma::mat left=pandya(A,pairs[d],active[mid],angles);
            arma::mat right=pandya(B,active[mid],pairs[e],angles);
            for(size_t u=0;u<active[mid].size();++u)
            {
              auto ab=active[mid][u]; left.col(u)*=g.occ[ab[0]]-g.occ[ab[1]];
            }
            if(order==0) ph[d][e]+=left*right;
            else ph[d][e]-=left*right;
          }
        }
        const auto ph_element=[&](int i,int l,int k,int j)->double
        {
          int r=index[i*g.n+l],s=index[k*g.n+j];
          if(r<0 || s<0) return 0.;
          const auto& m=ph[(g.tz[i]-g.tz[l])/2+1][(g.tz[k]-g.tz[j])/2+1];
          return m.n_elem?m(r,s):0.;
        };
        for(auto& block:Z.TwoBody.MatEl)
        {
          auto& cb=ms.GetTwoBodyChannel(block.first[0]); auto& ck=ms.GetTwoBodyChannel(block.first[1]);
          if(cb.Tz-ck.Tz!=qZ) continue;
          const int J=cb.J;
          #pragma omp parallel for schedule(dynamic,8) if(block.second.n_elem>65536)
          for(int r=0;r<cb.GetNumberKets();++r)
          {
            auto& bra=cb.GetKet(r); int i=bra.p,j=bra.q;
            double phase=AngMom::phase((g.j2[i]+g.j2[j])/2-J);
            for(int s=0;s<ck.GetNumberKets();++s)
            {
              auto& ket=ck.GetKet(s); int k=ket.p,l=ket.q;
              double direct=ph_element(i,l,k,j),exchange=ph_element(j,l,k,i);
              if(direct==0. && exchange==0.) continue;
              // SixJ symmetry maps the inverse recoupling to the SAME table.
              double value=-(2*K+1.)/(2*J+1.)*(angles.get(i,l,k,j,J)*direct
                              -phase*angles.get(j,l,k,i,J)*exchange);
              block.second(r,s)+=value/std::sqrt((1.+(i==j))*(1.+(k==l)));
            }
          }
        }
      }
    }
  Z.MakeReduced();
  return Z;
}

} // namespace Commutator
