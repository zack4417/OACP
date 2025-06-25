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

#include "komo_factory.h"
#include <Kin/frame.h>
#include <Core/graph.h>

namespace mp
{

//==============KOMOFactory==============================================

void KOMOFactory::registerInit( const TreeInitGrounder & grounder )
{
  treeInitGrounder_ = grounder;
}

void KOMOFactory::registerTask( const std::string & type, const TreeSymbolGrounder & grounder )
{
  treeTasks_[ type ] = grounder;
}

std::shared_ptr< ExtensibleKOMO > KOMOFactory::createKomo() const
{
  auto komo = std::make_shared< ExtensibleKOMO >();
  komo->registerInit( treeInitGrounder_ );

  for ( const auto& task : treeTasks_ )
  {
    komo->registerTask( task.first, task.second );
  }

  return komo;
}


//==============ExtensibleKOMO==============================================

ExtensibleKOMO::ExtensibleKOMO()
  : KOMO_ext()
{

}

void ExtensibleKOMO::registerTask( const std::string & type, const TreeSymbolGrounder & grounder )
{
  treeTasks_[ type ] = grounder;
}

void ExtensibleKOMO::registerInit( const TreeInitGrounder & grounder )
{
  treeInitGrounder_ = grounder;
}

void ExtensibleKOMO::groundInit( int verbose )
{
  TreeBuilder tree(1.0, 0);
  groundInit( tree, verbose );
}

void ExtensibleKOMO::groundInit( const TreeBuilder &tb, int verbose )
{
  if( treeInitGrounder_ )
  {
    treeInitGrounder_( tb, this, verbose );
  }
}

void ExtensibleKOMO::groundTasks( double phase, const std::vector< std::string >& facts, int verbose )
{
  TreeBuilder tree(1.0, 0);
  Interval it;
  it.time.from = phase;
  it.time.to = phase + 1.0;
  it.edge.from = -1;
  it.edge.to = -1;

  groundTasks(it, tree, facts, verbose);
}

void ExtensibleKOMO::groundTasks( const Interval& interval, const TreeBuilder& tb, const std::vector< std::string >& facts, int verbose )
{
  if( facts.empty() )
  {
    return;
  }

  const auto& type = facts.front();

  std::vector< std::string >args;
  if( facts.size() > 1 )
  {
    args = std::vector< std::string > { facts.begin() + 1, facts.end() };
  }

  if( treeTasks_.find( type ) != treeTasks_.end() )
  {
    treeTasks_[ type ]( interval, tb, args, this, verbose ); // ground the symbol
  }
  else
  {
    HALT("UNKNOWN komo TAG: '" << type <<"'");
  }
}

void ExtensibleKOMO::applyRandomization( const std::vector< double > & randomVec )
{
  // apply random vector
  // initial position
  rai::KinematicWorld world;
  world.copy(this->world);
  world.calc_q();

  //randomVec={-1.0, -1.0};

  uint i = 0;
  for( const auto & f: world.frames )
  {
    if( f->ats["random_bounds"]  )
    {
      const auto& randomBounds = f->ats.get<arr>("random_bounds");

      for( uint j = 0; j < randomBounds.size(); ++j )
      {
        if( randomBounds(j) > 0 )
        {
          world.q(f->joint->qIndex + j) += randomBounds(j) * randomVec[i];
          ++i;
        }
      }
    }
  }

  world.calc_Q_from_q();
  world.calc_fwdPropagateFrames();

  this->setModel(world);
}

void ExtensibleKOMO::saveTrajectory( const std::string & suffix ) const
{
  std::string filename = ( "z.coordinates" + suffix ).c_str();

  ofstream fil( filename.c_str() );
  StringA jointNames = world.getJointNames();
  //first line: legend
  for(const auto& s:jointNames) fil <<s <<' ';
  fil <<endl;

  // positions
  auto xx = x;
  xx.reshape(T, world.q.N);

  // coordinate
  arr coordinates = zeros(T, world.q.N);

  for( auto t = 0; t < T; ++t )
  {
    const auto& x_t   = xx.row( t );

    coordinates.setMatrixBlock( x_t, t, 0 );
  }

  coordinates.write(fil, NULL, NULL, "  ");
  fil.close();
}

void ExtensibleKOMO::plotVelocity( const std::string & suffix ) const
{
  std::string filename = ( "z.velocities" + suffix ).c_str();
  std::string filenamePlt = ( "z.velocities" + suffix + ".plt" ).c_str();

  ofstream fil( filename.c_str() );
  StringA jointNames = world.getJointNames();
  //first line: legend
  for(const auto& s:jointNames) fil <<s <<' ';
  fil <<endl;

  // positions
  auto xx = x;
  xx.reshape(T, world.q.N);

  // speeds
  arr velocities = zeros(T-1, world.q.N);

  for( auto t = 0; t < T - 1; ++t )
  {
    const auto& x_t   = xx.row( t );
    const auto& x_t_1 = xx.row( t + 1 );
    const auto& v = ( x_t_1 - x_t ) / tau;

    velocities.setMatrixBlock( v, t, 0 );
  }

  velocities.write(fil, NULL, NULL, "  ");
  fil.close();

  ofstream fil2( filenamePlt.c_str() );
  fil2 <<"set key autotitle columnheader" <<endl;
  fil2 <<"set title 'velocities'" <<endl;
  fil2 <<"set term qt 2" <<endl;
  fil2 <<"plot '" << filename << "' \\" << endl;
  for(uint i=1;i<=jointNames.N;i++) fil2 <<(i>1?"  ,''":"     ") <<" u 0:"<<i<<" w l lw 3 lc " <<i <<" lt " <<1-((i/10)%2) <<" \\" <<endl;
  //    if(dualSolution.N) for(uint i=0;i<objectives.N;i++) fil <<"  ,'' u 0:"<<1+objectives.N+i<<" w l \\" <<endl;
  fil2 <<endl;
  fil2.close();

  // command
  std::string command = "load '" + filenamePlt + "'";
  gnuplot( command.c_str() );
}

arr ExtensibleKOMO::getCostsPerPhase()
{
  auto maxPhase = T / stepsPerPhase;

  bool wasRun = featureValues.N!=0;

  arr phi;
  ObjectiveTypeA tt;
  if(wasRun){
      phi.referTo( featureValues );
      tt.referTo( featureTypes );
  }

  //-- collect all task costs and constraints
  StringA name; name.resize(objectives.N);
  arr err=zeros(maxPhase);
  uint M=0;
  for(uint t=0; t<T; t++){
    uint p = std::floor( t / stepsPerPhase );
    for(uint i=0; i<objectives.N; i++) {
      Objective *task = objectives(i);
      if(task->isActive(t)){
        uint d=0;
        if(wasRun){
          d=task->map->__dim_phi(configurations({t,t+k_order}));
          for(uint j=0;j<d;j++) CHECK(tt(M+j)==task->type,"");
          if(d){
            if(task->type==OT_sos){
              for(uint j=0;j<d;j++) err(p) += rai::sqr(phi(M+j)); //sumOfSqr(phi.sub(M,M+d-1));
            }
            if(task->type==OT_ineq){
              for(uint j=0;j<d;j++) err(p) += rai::MAX(0., phi(M+j));
            }
            if(task->type==OT_eq){
              for(uint j=0;j<d;j++) err(p) += fabs(phi(M+j));
            }
            M += d;
          }
        }
      }
    }
  }
  CHECK_EQ(M , phi.N, "");

  return err;
}

}
