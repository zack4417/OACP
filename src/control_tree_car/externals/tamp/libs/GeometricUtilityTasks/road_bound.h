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

#include <geom_utility.h>

using namespace std;

struct RoadBound:Feature{

  RoadBound( const std::string & object_name, double half_road_width, double vehicle_width, const rai::KinematicWorld& G )
    : object_index_(getFrameIndex(G, object_name))
    , max_y_(half_road_width - 0.5 * vehicle_width)
  {

  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G)
  {
    rai::Frame * object = G.frames(object_index_);//G.getFrameByName( object_.c_str() ); // avoid doing that all the time (save frma index)
    arr posObject, posJObject;
    G.kinematicsPos(posObject, posJObject, object, offset_);    // get function to minimize and its jacobian in state G
    //
    //posObject = G.q;
    //posJObject = diag(1, 3);
    //
    y.resize( dim_ );
    y( 0 ) = posObject( 1 ) > max_y_ ? posObject( 1 ) - max_y_ : 0; // left
    y( 1 ) = posObject( 1 ) < -max_y_ ? -posObject( 1 ) + max_y_ : 0; // right
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

      if(posObject( 1 ) > max_y_) // left
        J.setMatrixBlock( posJObject.row( 1 ), 0 , 0 ); // left
      else
        J.setMatrixBlock( zeros(1, J.d1), 0 , 0 );

      if(posObject( 1 ) < -max_y_) // right
        J.setMatrixBlock( -posJObject.row( 1 ), 1 , 0 ); // right
      else
        J.setMatrixBlock( zeros(1, J.d1), 1 , 0 );

      //J.setMatrixBlock( posJObject.row( 1 ), 2 , 0 ); // centerline
    }
  }

  virtual uint dim_phi(const rai::KinematicWorld& G)
  {
    return dim_;
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("RoadBound");
  }

private:
  static const uint dim_ = 2; //3; (with centerline)
  const double max_y_ = 1.5;
  const rai::Vector offset_{4.0, 0, 0}; // part of the vehicle which should stay within the boundaries (here car front)
  uint object_index_{0};
};
