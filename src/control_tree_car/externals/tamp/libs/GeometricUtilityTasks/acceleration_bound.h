/*  ------------------------------------------------------------------
    Copyright 2016 Camille Phiquepal
    email: camille.phiquepal@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version. This program is distributed without
    any warranty. See the GNU General Public License for more details.
    You should have received a COPYING file of the full GNU General Public
    License along with this program. If not, see
    <http://www.gnu.org/licenses/>
    --------------------------------------------------------------  */

#pragma once

#include <math_utility.h>
#include <math.h>
#include <Kin/feature.h>
#include <geom_utility.h>
#include "Kin/TM_transition.h"
#include "Kin/TM_qItself.h"
#include "Kin/frame.h"
#include "Kin/flag.h"

using namespace std;

struct AccelerationBound:Feature{





  /*virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G)
  {
    rai::Frame * object = G.frames(object_index_);//G.getFrameByName( object_.c_str() ); // avoid doing that all the time (save frma index)
    arr posObject, posJObject;
    G.kinematicsPos(posObject, posJObject, object, offset_);    // get function to minimize and its jacobian in state G
    //
    //posObject = G.q;
    //posJObject = diag(1, 3);
    //
    y.resize( dim_ );
    y( 0 ) = posObject( 0 ) > upper_y_ ? posObject( 0 ) - upper_y_ : 0; // left
    y( 1 ) = posObject( 0 ) < lower_y_ ? posObject( 0 ) - lower_y_ : 0; // right
    //y( 2 ) = 0; // right
    y( 2 ) = posObject( 1 ) > upper_y_ ? posObject( 1 ) - upper_y_ : 0; // left
    y( 3 ) = posObject( 1 ) < lower_y_ ? posObject( 1 ) - lower_y_ : 0; // right
    //y( 5 ) = 0; // right
    //y( 2 ) = 0; //posObject( 1 ); // centerline

    if(&J) // jacobian
    {
      J.resize( dim_, posJObject.dim(1) );

//      if(posObject( 1 ) > max_y_) // left
//        J(0, 1) = 1; // left
//      else
//        J(0, 1) = 0;
//      J(0,0) = 0; J(0,2) = 0;

//      if(posObject( 1 ) < -max_y_) // right
//        J(1, 1) = 1; // left
//      else
//        J(1, 1) = 0;
//      J(1,0) = 0; J(1,2) = 0;

      if(posObject( 0 ) > upper_y_) // left
        J.setMatrixBlock( posJObject.row( 0 ), 0 , 0 ); // left
      else
        J.setMatrixBlock( zeros(0, J.d1), 0 , 0 );

      if(posObject( 0 ) < lower_y_) // right
        J.setMatrixBlock( -posJObject.row( 0 ), 1 , 0 ); // right
      else
        J.setMatrixBlock( zeros(0, J.d1), 1 , 0 );

      //J.setMatrixBlock( posJObject.row( 0 ), 2 , 0 ); // centerline
      if(posObject( 1 ) > upper_y_) // left
        J.setMatrixBlock( posJObject.row( 1 ), 2 , 0 ); // left
      else
        J.setMatrixBlock( zeros(1, J.d1), 2 , 0 );

      if(posObject( 1 ) < lower_y_) // right
        J.setMatrixBlock( -posJObject.row( 1 ), 3 , 0 ); // right
      else
        J.setMatrixBlock( zeros(1, J.d1), 3 , 0 );

      //J.setMatrixBlock( posJObject.row( 1 ), 4 , 0 ); // centerline
      //J.setMatrixBlock( posJObject.row( 1 ), 2 , 0 ); // centerline
    }
    
  }*/

  //static const uint dim_ = 4; //3; (with centerline)
  const double upper_y_ = 3;
  const double lower_y_ = -3;
  const rai::Vector offset_{0, 0, 0}; // part of the vehicle which should stay within the boundaries (here car front)
  uint object_index_{0};
  double posCoeff, velCoeff, accCoeff;  ///< coefficients to blend between velocity and acceleration penalization
  arr H_rate_diag;            ///< cost rate (per TIME, not step), given as diagonal of the matrix H
  double H_rate;  ///< cost rate (per TIME, not step), given as scalar, will be multiplied by Joint->H (given in ors file)
  bool effectiveJointsOnly;


  AccelerationBound(const std::string & object_name, double upperbound, double lowerbound, const rai::KinematicWorld& G,bool effectiveJointsOnly=false)
      : object_index_(getFrameIndex(G, object_name))
    , upper_y_(upperbound)
    , lower_y_(lowerbound)
  {
  accCoeff = 1;
  order = 2;
    H_rate = rai::getParameter<double>("Hrate", 1.);
  arr H_diag;
  if(rai::checkParameter<arr>("Hdiag")) {
    H_diag = rai::getParameter<arr>("Hdiag");
  } else {
    H_diag = G.getHmetric(); //G.naturalQmetric();
  }
  H_rate_diag = H_rate*H_diag;
  }


  virtual void phi(arr& y, arr& J, const WorldL& Ktuple)
  {
     order=2;

  bool handleSwitches=effectiveJointsOnly;
  uint qN=Ktuple(0)->q.N;
  for(uint i=0; i<Ktuple.N; i++) if(Ktuple(i)->q.N!=qN) { handleSwitches=true; break; }
  
  double tau = Ktuple(-1)->frames(0)->tau; // - Ktuple(-2)->frames(0)->time;
  
  if(!handleSwitches) { //simple implementation
    //-- transition costs
    y.resize(Ktuple.last()->q.N).setZero();
    
    //individual weights
    double hbase = H_rate*sqrt(tau), tau2=tau*tau;
//    hbase = H_rate;
    arr h = zeros(y.N);
    for(rai::Joint *j:Ktuple.last()->fwdActiveJoints) for(uint i=0; i<j->qDim(); i++) {
        h(j->qIndex+i) = hbase*j->H;
        if(j->frame->flags && !(j->frame->flags & (1<<FL_normalControlCosts))) {
          h(j->qIndex+i)=0.;
        }
      }

      arr a=Ktuple.elem(-2)->q; //N-2
      a *= -2.;
      a += Ktuple.elem(-3)->q; //N-3
      a += Ktuple.elem(-1)->q; //N-1
      a *= (accCoeff/tau2);
      /*a(0)=a( 0 ) > upper_y_ ? a( 0 )*10: a(0);
      a(0)=a( 0 ) < lower_y_ ? a( 0 )*10: a(0);
      a(1)=a( 1 ) > upper_y_ ? a( 1 )*10: a(1);
      a(1)=a( 1 ) < lower_y_ ? a( 1 )*10: a(1);*/
      a(0)=a( 0 ) > upper_y_ ? a( 0 )-upper_y_: 0;
      a(0)=a( 0 ) < lower_y_ ? a( 0 )-lower_y_: 0;
      a(1)=a( 1 ) > upper_y_ ? a( 1 )-upper_y_: 0;
      a(1)=a( 1 ) < lower_y_ ? a( 1 )-lower_y_: 0;
      y = a;
      

    
    //multiply with h...
    y *= h;
    
    if(!!J) {
      arr Jtau;  Ktuple(-1)->jacobianTime(Jtau, Ktuple(-1)->frames(0));  expandJacobian(Jtau, Ktuple, -1);
//      arr Jtau2;  Ktuple(-2)->jacobianTime(Jtau2, Ktuple(-2)->frames(0));  expandJacobian(Jtau2, Ktuple, -2);
//      arr Jtau = Jtau1 - Jtau2;
      
      uint n = Ktuple.last()->q.N;
      J.resize(y.N, Ktuple.N, n).setZero();
      for(uint i=0; i<n; i++) {
          uint j = i*J.d1*J.d2 + i;
          J.elem(j+(Ktuple.N-3)*J.d2) += accCoeff/tau2;
          J.elem(j+(Ktuple.N-2)*J.d2) += -2.*accCoeff/tau2;
          J.elem(j+(Ktuple.N-1)*J.d2) += accCoeff/tau2;
      }
      J.reshape(y.N, Ktuple.N*n);
      
      /*
      if(a(0)>lower_y_&&a(0)<upper_y_)
      {
        J.elem(0)=0;
      }  
    if(a(1)>lower_y_&&a(1)<upper_y_)
      {
        J.elem(1)=0;
      }*/
      J = h%J;
      J += (-1.5/tau)*y*Jtau;
      
    }
  } else { //with switches
    rai::Array<rai::Joint*> matchingJoints = getMatchingJoints(Ktuple.sub(-1-order,-1), effectiveJointsOnly);
    double h = H_rate*sqrt(tau), tau2=tau*tau;
    
    uint ydim=0;
    uintA qidx(Ktuple.N);
    for(uint i=0; i<matchingJoints.d0; i++) ydim += matchingJoints(i,0)->qDim();
    y.resize(ydim).setZero();
    if(!!J) {
      qidx(0)=0;
      for(uint i=1; i<Ktuple.N; i++) qidx(i) = qidx(i-1)+Ktuple(i-1)->q.N;
      J.resize(ydim, qidx.last()+Ktuple.last()->q.N).setZero();
    }
    
    uint m=0;
    for(uint i=0; i<matchingJoints.d0; i++) {
      rai::Array<rai::Joint*> joints = matchingJoints[i];
      uint jdim = joints(0)->qDim(), qi1=0, qi2=0, qi3=0;
      for(uint j=0; j<jdim; j++) {
        if(order>=0) qi1 = joints.elem(-1)->qIndex+j;
        if(order>=1) qi2 = joints.elem(-2)->qIndex+j;
        if(order>=2 && accCoeff) qi3 = joints.elem(-3)->qIndex+j;
        rai::Joint *jl = joints.last();
        double hj = h * jl->H;
        if(jl->frame->flags && !(jl->frame->flags & (1<<FL_normalControlCosts))) hj=0.;
        //TODO: adding vels + accs before squareing does not make much sense!
        if(order>=0 && posCoeff) y(m) += posCoeff*hj       * (Ktuple.elem(-1)->q(qi1));
        if(order>=1 && velCoeff) y(m) += (velCoeff*hj/tau) * (Ktuple.elem(-1)->q(qi1) -    Ktuple.elem(-2)->q(qi2));
        if(order>=2 && accCoeff) y(m) += (accCoeff*hj/tau2)* (Ktuple.elem(-1)->q(qi1) - 2.*Ktuple.elem(-2)->q(qi2) + Ktuple.elem(-3)->q(qi3));
        if(!!J) {
          if(order>=0 && posCoeff) { J(m, qidx.elem(-1)+qi1) += posCoeff*hj; }
          if(order>=1 && velCoeff) { J(m, qidx.elem(-1)+qi1) += velCoeff*hj/tau;  J(m, qidx.elem(-2)+qi2) += -velCoeff*hj/tau; }
          if(order>=2 && accCoeff) { J(m, qidx.elem(-1)+qi1) += accCoeff*hj/tau2; J(m, qidx.elem(-2)+qi2) += -2.*accCoeff*hj/tau2; J(m, qidx.elem(-3)+qi3) += accCoeff*hj/tau2; }
        }
        m++;
      }
    }
    CHECK_EQ(m, ydim,"");
  }
  }

   virtual uint dim_phi(const WorldL& G)
   {
          bool handleSwitches=effectiveJointsOnly;
  uint qN=G(0)->q.N;
  for(uint i=0; i<G.N; i++) if(G.elem(i)->q.N!=qN) { handleSwitches=true; break; }
  
  if(!handleSwitches) {
    return G.last()->getJointStateDimension();
  } else {
//    for(uint i=0;i<G.N;i++) cout <<i <<' ' <<G(i)->joints.N <<' ' <<G(i)->q.N <<' ' <<G(i)->getJointStateDimension() <<endl;
    rai::Array<rai::Joint*> matchingJoints = getMatchingJoints(G.sub(-1-order,-1), effectiveJointsOnly);
    uint ydim=0;
    for(uint i=0; i<matchingJoints.d0; i++) {
//      cout <<i <<' ' <<matchingJoints(i,0)->qIndex <<' ' <<matchingJoints(i,0)->qDim() <<' ' <<matchingJoints(i,0)->name <<endl;
      ydim += matchingJoints(i,0)->qDim();
    }
    return ydim;
  }
  return uint(-1);
   }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) { HALT("can only be of higher order"); }
 
  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("acceleration_bound");
  }

  virtual uint dim_phi(const rai::KinematicWorld& G) { return G.getJointStateDimension(); }
  virtual Graph getSpec(const rai::KinematicWorld& K){ return Graph({{"feature", "Transition"}}); }


};
