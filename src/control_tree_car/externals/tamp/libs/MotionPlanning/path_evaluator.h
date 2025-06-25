#pragma once

#include <Kin/kin.h>

namespace mp
{

std::pair< double, double > evaluate( rai::Array< rai::KinematicWorld > & kinFrames, double tau );

}
