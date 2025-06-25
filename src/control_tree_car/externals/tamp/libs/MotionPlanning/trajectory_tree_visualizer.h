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

#include <Kin/kinViewer.h>

namespace mp
{
// the visualizer can work as long as the kinmeatic frames it has to display exist!
class TrajectoryTreeVisualizer
{
public:           // for each terminal node, for each possible world, each frame
  TrajectoryTreeVisualizer( const rai::Array< rai::Array< rai::Array< rai::KinematicWorld > > > & frames, const std::string & name, uint stepsPerSecs );

private:
  std::vector< std::shared_ptr< KinPathViewer > > views_;
  uint stepsPerSecs_;
};

}
