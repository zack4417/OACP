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
#include <geom_utility.h>

#include <Kin/feature.h>
#include <Kin/taskMaps.h>

#include <Kin/proxy.h>
#include <math.h>

using namespace std;

struct AxisBound:Feature{

  enum Axis
  {
    X = 0,
    Y,
    Z,
    Combine
  };

  enum BoundType
  {
    MIN = 0,
    MAX,
    EQUAL
  };

  AxisBound( const std::string & object, const enum Axis & axis, const enum BoundType & boundType, const rai::KinematicWorld& G )
    : object_index_(getFrameIndex(G, object))
    , boundType_( boundType )
  {
    if( axis == X ) id_ = 0;
    else if( axis == Y ) id_ = 1;
    else if( axis == Z ) id_ = 2;
    else if( axis == Combine ) id_ = 3;

    sign_ = ( ( boundType_ == MIN ) ? -1 : 1 );
  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G)
  {
    rai::Frame * object = G.frames(object_index_); // avoid doing that all the time (save frma index)
    arr posObject, posJObject;
    G.kinematicsPos(posObject, posJObject, object);    // get function to minimize and its jacobian in state G

    // fast version
//    posObject = G.q;
//    posJObject = diag(1, 3);
    //

    y.resize( dim_ );
    if(id_==3)
    {
      y(0)=sign_ *sqrt(pow(posObject(0),2)+pow(posObject(1),2));
    }
    else
    {
    y( 0 ) = sign_ * posObject( id_ );
    }

    if(&J)
    {
      if(id_==3)
      {
        J.resize(dim_, posJObject.dim(1));
        const auto theta = std::atan2(posObject(1), posObject(0));

          J(0,0) = posJObject(0, 0);
          J(0,1) = posJObject(1, 1);
          J(0,2) = 0;
      }
      else
      {
      J.resize( dim_, posJObject.dim(1) );
      J.setMatrixBlock( sign_ * posJObject.row( id_ ), 0 , 0 );    // jacobian
      }
    }
  }

  virtual uint dim_phi(const rai::KinematicWorld& G)
  {
    return dim_;
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("AxisBound");
  }

private:
  static const uint dim_ = 1;
  const uint object_index_;
  const BoundType boundType_;
  std::size_t id_= 0;
  double sign_ = 1;
};
