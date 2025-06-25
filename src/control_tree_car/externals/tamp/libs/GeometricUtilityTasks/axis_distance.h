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

#include <Kin/feature.h>
#include <Kin/taskMaps.h>

#include <Kin/proxy.h>

using namespace std;

struct AxisDistance:Feature{

  enum Axis
  {
    X = 0,
    Y
  };

  enum BoundType
  {
    MIN = 0,
    MAX
  };

  enum DistanceType
  {
    ABS = 0,
    SIGNED
  };

  AxisDistance( const std::string & object_1, const std::string & object_2, double bound, const enum Axis & axis, const enum BoundType & boundType, const enum DistanceType & distType, const double k = 1.0 )
    : object_1_( object_1 )
    , object_2_( object_2 )
    , bound_( bound )
    , boundType_( boundType )
    , id_( axis == X ? 0 : 1 )
    , distanceType_( distType )
    , k_( k )
  {

  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G)
  {
    rai::Frame *object_1 = G.getFrameByName( object_1_.c_str() );
    arr posObject_1, posJObject_1;
    G.kinematicsPos(posObject_1, posJObject_1, object_1);    // get function to minimize and its jacobian in state G

    rai::Frame *object_2 = G.getFrameByName( object_2_.c_str() );
    arr posObject_2, posJObject_2;
    G.kinematicsPos(posObject_2, posJObject_2, object_2);    // get function to minimize and its jacobian in state G

    const double sign = ( ( boundType_ == MIN ) ? 1 : -1 );

    arr tmp_y = zeros( dim_ );
    arr tmp_J = zeros( dim_, posJObject_1.dim(1) );

    if( distanceType_ == ABS )
    {
      const double diff = posObject_1( id_ ) - posObject_2( id_ );
      const double dist = fabs( diff );

      //const arr Jdiff =  posJObject_1 - posJObject_2;
      const arr Jdist = ( diff > 0 ? posJObject_1.row( id_ ) - posJObject_2.row( id_ ) :
                                     posJObject_2.row( id_ ) - posJObject_1.row( id_ ) );

      tmp_y( 0 ) = - k_ * sign * ( dist - bound_ );

      tmp_J.setMatrixBlock( - k_ * sign * Jdist, 0 , 0 );    // jacobian
    }
    else
    {
      const double diff = posObject_1( id_ ) - posObject_2( id_ );
      const arr Jdiff =  posJObject_1.row( id_ ) - posJObject_2.row( id_ );

      tmp_y( 0 ) = - k_ * sign * ( diff - bound_ );

      tmp_J.setMatrixBlock( - k_ * sign * Jdiff, 0 , 0 );    // jacobian
    }
    // commit results
    y = tmp_y;
    if(&J) J = tmp_J;
  }

  virtual uint dim_phi(const rai::KinematicWorld& G)
  {
    return dim_;
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("AxisDistance");
  }

private:
  static const uint dim_ = 1;
  std::string object_1_;
  std::string object_2_;
  const double bound_;
  const BoundType boundType_;
  const DistanceType distanceType_;
  const double k_;
  const std::size_t id_;
};
