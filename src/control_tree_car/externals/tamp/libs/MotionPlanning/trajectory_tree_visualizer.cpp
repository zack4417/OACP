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

#include <trajectory_tree_visualizer.h>

namespace mp
{

TrajectoryTreeVisualizer::TrajectoryTreeVisualizer( const rai::Array< rai::Array< rai::Array< rai::KinematicWorld > > > & frames, const std::string & name, uint stepsPerSecs )
  : stepsPerSecs_( stepsPerSecs )
{
  // get number of views
  uint n = 0;
  for( auto leaf = frames.begin(); leaf != frames.end(); ++leaf )
  {
    for( auto x : *leaf )
    {
      if( x.N > 0 )
      {
        n++;
      }
    }
  }

  views_.resize( n );

  // build each view
  uint index = 0;
  for( auto leaf = frames.begin(); leaf != frames.end(); ++leaf )
  {
    for( uint w = 0; w < leaf->N; ++w )
    {
      rai::Array< rai::KinematicWorld > & traj = (*leaf)( w );
      if( traj.N > 0 )
      {
        std::string windowName = name + std::string( "-world-" ) + std::to_string( w ) + " - " + std::to_string( index ) ;

        Var<WorldL> configs;
        configs.name() = windowName.c_str();
        views_[ index ] = std::make_shared< KinPathViewer >(configs ,  1.0 / stepsPerSecs_, -0 ); //0.05

        rai::Array< rai::KinematicWorld * > configurations( traj.N );

        for( uint s = 0; s < traj.N; ++s )
        {
          configurations( s ) = &traj( s ); //s
        }

        views_[ index ]->setConfigurations( configurations );

//            setHeightAngle(12.);
//            setPosition(10., -15., 8.);
//          //  setPosition(10., -4., 10.);
//            focus(0, 0, 1.);
//            upright();

        // for baxter
//        {
//          auto & kin = views_[ index ];
//          const double zf = 1.47;
//          const double s = 0.55;
//          kin->copy.gl().camera.setPosition(s * 10., s * 3., zf + s * ( 2.28 - zf ));
//          kin->copy.gl().camera.focus(0, 0, zf);
//          kin->copy.gl().camera.upright();
//        }

        index++;
      }
    }
  }
//  for( auto w = 0; w < policy->N(); ++w )
//  {
//    std::string windowName = name + std::string( "-world-" ) + std::to_string( w );
//    views_[ w ] = std::make_shared< OrsPathViewer >( windowName.c_str(),  0.1, -0 );
//    views_[ w ]->setConfigurations( policy->getTrajectory( w ) );
//  }


//  threadOpenModules( true );
}

}
