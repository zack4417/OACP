#pragma once

#include <skeleton.h>
#include <boost/filesystem.hpp>

namespace mp
{

static double updateValue( const Policy::GraphNodeType::ptr & node )
{
  double value = 0;

  for( const auto& c : node->children() )
  {
    value += c->data().p * ( c->data().markovianReturn + updateValue( c ) );
  }

  return value;
}

static void updateValues( Policy & policy )
{
  policy.setValue( updateValue( policy.root() ) );
}

static arr extractAgentQMask( const rai::KinematicWorld & G )  // retrieve agent joints
{
  uintA selectedBodies;

  for( const auto & f: G.frames )
  {
    if( f->ats["agent"] && f->ats.get<bool>("agent") )
    {
      selectedBodies.setAppend(f->ID);
    }
  }

  // build mask
  arr qmask = zeros( G.q.d0 );

  for( const auto& b : selectedBodies )
  {
    rai::Joint *j = G.frames.elem(b)->joint;

    CHECK( j, "incoherence, the joint should not be null since it has been retrieved before" );

    for( auto i = j->qIndex; i < j->qIndex + j->dim; ++i )
    {
      qmask( i ) = 1;
    }
  }

  return qmask;
}

class WorkingDirLock
{
public:
  WorkingDirLock()
    : working_dir_(boost::filesystem::current_path())
  {

  }

  ~WorkingDirLock()
  {
    boost::filesystem::current_path(working_dir_);
  }

private:
  boost::filesystem::path working_dir_;
};

static Graph loadKin(const std::string & kinDescription)
{
  WorkingDirLock lock;
  return Graph(kinDescription.c_str());
}

static std::shared_ptr< rai::KinematicWorld > createKin(const Graph & kinG )
{
  WorkingDirLock lock;
  auto kin = std::make_shared< rai::KinematicWorld >();
  kin->init( kinG );
  return kin;
}

static void freeKomo( ExtensibleKOMO::ptr komo )
{
  listDelete( komo->configurations );
  listDelete( komo->objectives );
  listDelete( komo->switches );
}

}
