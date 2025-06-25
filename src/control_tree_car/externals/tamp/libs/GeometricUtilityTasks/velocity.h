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

#include <Kin/taskMaps.h>
#include <Kin/proxy.h>
#include <Kin/frame.h>

#include <math_utility.h>

//===========================================================================

struct Velocity:Feature{

  Velocity( const std::string & object, double desired_v )
    : object_( object )
    , desired_v_(desired_v)
  {

  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("Velocity");
  }

  virtual void phi(arr& y, arr& J, const WorldL& Gs) override
  {
    //TaskMap::phi(y,J,Ks,tau,t);
    CHECK(order==1,"");
    CHECK(Gs.size() >= 1,"");

    rai::Frame *object = Gs(1)->getFrameByName( object_.c_str() );
    const auto Xoffset = -0.5 * object->shape->size(0);

    // initialize y and J
    y.resize(1);//zeros(dim_phi(Gs, t));
    if(&J){
      uintA qidx(Gs.N);
      qidx(0)=0;
      for(uint i=1;i<Gs.N;i++) qidx(i) = qidx(i-1)+Gs(i-1)->q.N;
      J = zeros(y.N, qidx.last()+Gs.last()->q.N);
    }

    // get speed vector
    arr y_vel,Jvel;
    TM_Default vel(TMT_posDiff, object->ID, rai::Vector(Xoffset,0,0));
    vel.order = 1;
    vel.__phi(y_vel, Jvel, Gs);

    // get omega quat
//    arr omega_quat,Jomega_quat;
//    TaskMap_Default om(quatDiffTMT, object->ID, rai::Vector(Xoffset,0,0));
//    om.order = 1;
//    om.phi(omega_quat, Jomega_quat, Gs, tau, t);

    // get orientation vector
    arr y_vec,Jvec;
    TM_Default vec(TMT_vec, object->ID, rai::Vector(1,0,0));
    vec.order = 0;
    vec.phi(y_vec, Jvec, *Gs(1));

    // commit results
    y(0) = -desired_v_ + scalarProduct(y_vel, y_vec);

    if(&J)
    {
      const auto offset = J.d1 / Gs.N;

      CHECK( Jvec.d1 == offset, "" );

      for(auto i = 0; i < Gs.size(); ++i)
      {
        auto JvelSub = Jvel.sub(0, -1, i * offset, (i+1) * offset -1);
        if( i == 1 )
        {
          J.setMatrixBlock( ~y_vel * Jvec + ~y_vec * JvelSub, 0, i * offset );
        }
        else
        {
          J.setMatrixBlock( ~y_vec * JvelSub, 0, i * offset );
        }
      }
    }
  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) override
  {
    CHECK(false,"The phi function taking the list of kinematic worlds should be taken");
  }

  virtual uint dim_phi(const rai::KinematicWorld& K) override
  {
    //CHECK(false,"The phi function taking the list of kinematic worlds should be taken");

    return dim_;
  }

private:
  static const uint dim_ = 1;
  std::string object_;
  double desired_v_;
};
