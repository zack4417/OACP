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

#include <task_planner.h>
#include <motion_planner.h>

struct TAMPlanningConfiguration
{
  uint maxIterations = 0;
  bool saveInformedPolicy = false;
  bool saveFinalPolicy = false;
  bool showFinalPolicy = false;
  uint showDurationSecs = 0;
};

class TAMPController
{
public:
  TAMPController( TaskPlanner & tp, MotionPlanner & mp )
    : tp_( tp )
    , mp_( mp )
  {

  }

  virtual Policy plan( const TAMPlanningConfiguration & ) = 0;

protected:
  TaskPlanner & tp_;
  MotionPlanner & mp_;
};
