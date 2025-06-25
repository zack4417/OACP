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
#include <Kin/kin.h>
#include <Kin/taskMaps.h>
#include <Kin/proxy.h>
#include <Kin/frame.h>

//===========================================================================

struct CarKinematic:Feature{

  CarKinematic( const std::string & object, const rai::KinematicWorld& G );

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("CarKinematic");
  }

  virtual void phi(arr& y, arr& J, const WorldL& Gs) override;

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) override;

  virtual uint dim_phi(const rai::KinematicWorld& K) override;

private:
  static const uint dim_ = 1;
  const uint object_index_;
};
