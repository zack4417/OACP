#include <Kin/kin.h>

#include <chrono>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include <KOMO/komo.h>
#include <Kin/taskMaps.h>

#include <math_utility.h>

namespace mp
{
std::pair< double, double > evaluate( rai::Array< rai::KinematicWorld > & kinFrames, double tau )
{
  std::vector< arr > framesY;

  TM_Transition tm( kinFrames.front() );
  tm.effectiveJointsOnly = false;
  tm.order = 2;
  arr y, J;
  WorldL Gs(3);

  // collect the y with task map
  arr lastPos; double length = 0;
  for( uint i = 0; i < kinFrames.size(); ++i )
  {
    auto kin = kinFrames.at(i);
    arr pos, J;
    auto frame = kin.getFrameByName( "baxterR" );
    kin.kinematicsPos( pos, J, frame );
    if( lastPos.N == 3 )
    {
      auto diff = pos - lastPos;
      length += norm2( diff );
    }
    lastPos = pos;


    if( i > 1 && i < kinFrames.size() - 1 )
    {
      Gs(0) = &kinFrames.at(i-1);
      Gs(1) = &kinFrames.at(i);
      Gs(2) = &kinFrames.at(i+1);

      tm.phi( y, J, Gs );

      framesY.push_back( y );
    }
  }

  // compute sum of square
  double cost = 0;
  for( auto y : framesY )
  {
    for( double a : y )
    {
      cost+= a*a;
    }
  }

  return std::make_pair( length, cost );
}
}
