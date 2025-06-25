#pragma one

#include "komo_planner.h"
#include <komo_wrapper.h>

#include <Kin/switch.h>
#include <Kin/TM_angVel.h>
#include <Kin/TM_default.h>
#include <Kin/TM_gravity.h>
#include <Kin/TM_InsideBox.h>
#include <Kin/TM_AboveBox.h>
#include <Kin/TM_qLimits.h>

#include <functional>
#include <gtest/gtest.h>

using namespace std;
using namespace mp;

/*
 *
 * Tests of various komo configurations komo with car examples
 *
 */

struct CarKOMOWrapper : public mp::KomoWrapper
{
  CarKOMOWrapper(KOMO_ext * komo)
    : mp::KomoWrapper(komo)
  {

  }

  void setVelocity(const Interval& it, const TreeBuilder& tree, const char* shape, ObjectiveType type, const arr& target_velocity, double scale)
  {
    addObjective( it, tree, new TM_Default(TMT_pos, komo_->world, shape), type, target_velocity, scale, 1);
  }

  //  void addObjective(const Interval& it, const TreeBuilder& tb, Feature* map, ObjectiveType type=OT_sos, const arr& target=NoArr, double scale=-1., int order=-1, int deltaFromStep=0, int deltaToStep=0);

  void setPosition(const Interval& it, const TreeBuilder& tree, const char* shape, const char* shapeRel, ObjectiveType type, const arr& target_position, double scale)
  {
    addObjective( it, tree, new TM_Default(TMT_pos, komo_->world, shape, NoVector, shapeRel, NoVector), type, target_position, scale, 0);
  }

  // komo->setPosition( t_start+1, -1, "car_ego", car_next.c_str(), OT_sos, {-0.7, 0, 0} );
};

using W = CarKOMOWrapper;

/////////////////Tasks////////////////////////
struct AxisBound:Feature{

  enum Axis
  {
    X = 0,
    Y,
    Z
  };

  enum BoundType
  {
    MIN = 0,
    MAX
  };

  AxisBound( const std::string & object, double bound, const enum Axis & axis, const enum BoundType & boundType, const double k = 1.0 )
    : object_( object )
    , bound_( bound )
    , boundType_( boundType )
    , k_( k )
  {
    if( axis == X ) id_ = 0;
    else if( axis == Y ) id_ = 1;
    else if( axis == Z ) id_ = 2;
  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G)
  {
    rai::Frame *object = G.getFrameByName( object_.c_str() );
    arr posObject, posJObject;
    G.kinematicsPos(posObject, posJObject, object);    // get function to minimize and its jacobian in state G

    const double sign = ( ( boundType_ == MIN ) ? 1 : -1 );

    arr tmp_y = zeros( dim_ );
    tmp_y( 0 ) = - k_ * sign * ( posObject( id_ ) - bound_ );

    arr tmp_J = zeros( dim_, posJObject.dim(1) );
    tmp_J.setMatrixBlock( - k_ * sign * posJObject.row( id_ ), 0 , 0 );    // jacobian

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
    return rai::String("AxisBound");
  }

private:
  static const uint dim_ = 1;
  std::string object_;
  const double bound_;
  const BoundType boundType_;
  const double k_;
  std::size_t id_= 0;
};


/////////////////Grounders////////////////////
class InitialGrounder
{
public:
  void init( const TreeBuilder& tree, KOMO_ext* komo, int verbose )
  {
    mp::Interval always{{0, -1}, {0, 1}};

    // road bounds
    W(komo).addObjective( always, tree, new AxisBound( "car_ego", -0.15, AxisBound::Y, AxisBound::MIN ), OT_ineq, NoArr, 1e2, 0 );
    W(komo).addObjective( always, tree, new AxisBound( "car_ego",  0.15, AxisBound::Y, AxisBound::MAX ), OT_ineq, NoArr, 1e2, 0 );

    // min speed
    W(komo).addObjective( always, tree, new AxisBound( "car_ego",  0.00, AxisBound::X, AxisBound::MIN ), OT_ineq, - arr{ 0.03 }, 1e2, 1 );

    // truck speed
    arr truck_speed{ 0.03, 0, 0 };
    truck_speed( 0 ) = 0.03;
    //komo->setVelocity( 0.0, -1, "truck", NULL, OT_eq, truck_speed );
    //W(komo).addObjective( always, tree, new TM_Default(TMT_pos, komo->world, "truck"), OT_eq, truck_speed, 1e2, 1);
    W(komo).setVelocity(always, tree, "truck", OT_eq, truck_speed, 1e2);

    // preference ego lane?
    //W(komo).addObjective( always, tree, new AxisBound( "car_ego", -0.1, AxisBound::Y, AxisBound::MAX ), OT_sos, NoArr, 1e0, 0 );
    // min speed
    W(komo).addObjective( always, tree, new AxisBound( "car_ego",  0.00, AxisBound::X, AxisBound::MIN ), OT_ineq, - arr{ 0.03 }, 1e2, 1 );

    // collision
    komo->activateCollisions( "car_ego", "truck" );
    komo->activateCollisions( "car_ego", "car_op" );
    komo->add_collision( true );
  }

  virtual void groundInitSingleAgent( const TreeBuilder& tree, KOMO_ext* komo, int verbose )
  {
    mp::Interval always{{0, -1}, {0, 1}};

    init( tree, komo, verbose );

    // opposite car speed
    arr op_speed{ -0.03, 0, 0 };
    //komo->setVelocity( 0.0, -1, "car_op", NULL, OT_eq, op_speed );
    //W(komo).addObjective( always, tree, new TM_Default(TMT_pos, komo->world, "car_op"), OT_eq, op_speed, 1e2, 1);
    W(komo).setVelocity( always, tree, "car_op", OT_eq, op_speed, 1e2);
  }

  virtual void groundInitDoubleAgent( const TreeBuilder& tree, KOMO_ext* komo, int verbose )
  {
    mp::Interval always{{0, -1}, {0, 1}};

    init( tree, komo, verbose );

    arr op_speed{ -0.04, 0, 0 };
    //komo->setVelocity( 0, 1.5, "car_op", NULL, OT_eq, op_speed );
    //W(komo).addObjective( always, tree, new TM_Default(TMT_pos, komo->world, "car_op"), OT_eq, op_speed, 1e2, 1);
    W(komo).setVelocity( always, tree, "car_op", OT_eq, op_speed, 1e2);
  }
};

class InitGrounderMock : public InitialGrounder
{
public:
  virtual void groundInitSingleAgent( const TreeBuilder& tree, KOMO_ext* komo, int verbose )
  {
    InitialGrounder::groundInitSingleAgent( tree, komo, verbose );

    nInitSingleAgent++;
  }

  virtual void groundInitDoubleAgent( const TreeBuilder& tree, KOMO_ext* komo, int verbose )
  {
    InitialGrounder::groundInitDoubleAgent( tree, komo, verbose );

    nInitDoubleAgent++;
  }

  uint nInitSingleAgent = 0;
  uint nInitDoubleAgent = 0;
};

void groundLook( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string > & args, KOMO_ext* komo, int verbose )
{
  //komo->addObjective( t_start + 0.9, t_end, new AxisBound( "car_ego", 0.0, AxisBound::Y, AxisBound::MIN ), OT_sos );
  mp::Interval it{{phase.time.from + 0.9, phase.time.to}, phase.edge};
  W(komo).addObjective(it, tree, new AxisBound( "car_ego", 0.0, AxisBound::Y, AxisBound::MIN ), OT_sos, NoArr, 1e2, 0);

  if( verbose > 0 )
  {
    std::cout << phase.time.from << "->" << phase.time.from << ": " << " look " << " at " << args[0] << std::endl;
  }
}

void groundOvertake( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string > & args, KOMO_ext* komo, int verbose )
{
  // overtake
  //komo->setPosition( t_end, -1, "car_ego", args[0].c_str(), OT_sos, { 0.45, 0, 0 } );
  mp::Interval future{{phase.time.to, -1}, phase.edge};
  W(komo).setPosition(future, tree, "car_ego", args[0].c_str(), OT_sos, { 0.45, 0, 0 }, 1e2);

  if( verbose > 0 )
  {
    std::cout << phase.time.from << "->" << phase.time.from << ": " << " overtake " << args[0] << std::endl;
  }
}

void groundFollow( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string > & args, KOMO_ext* komo, int verbose )
{
  // overtake
  //komo->setPosition( t_end, -1, "car_ego", "truck", OT_sos, { -0.7, 0, 0 } ); // -0.55
  mp::Interval future{{phase.time.to, -1}, phase.edge};
  W(komo).setPosition(future, tree, "car_ego", args[0].c_str(), OT_sos, { -0.7, 0, 0 }, 1e2);

  if( verbose > 0 )
  {
    std::cout << phase.time.from << "->" << phase.time.from << ": " << " follow " << args[0] << std::endl;
  }
}

void groundAccelerate( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string > & args, KOMO_ext* komo, int verbose )
{
  // opposite car speed
  arr op_speed{ -0.07, 0, 0 };
  //komo->setVelocity( t_start, -1, "car_op", NULL, OT_eq, op_speed );
  mp::Interval always{{phase.time.from, -1}, phase.edge};
  W(komo).setVelocity( always, tree, "car_op", OT_eq, op_speed, 1e2);
}

void groundContinue( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string > & args, KOMO_ext* komo, int verbose )
{
  // opposite car speed
  arr op_speed{ -0.04, 0, 0 };
  //komo->setVelocity( t_start, -1, "car_op", NULL, OT_eq, op_speed );
  mp::Interval always{{phase.time.from, -1}, phase.edge};
  W(komo).setVelocity( always, tree, "car_op", OT_eq, op_speed, 1e2);
}

void groundSlowDown( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string > & args, KOMO_ext* komo, int verbose )
{
  // opposite car speed
  arr op_speed{ -0.015, 0, 0 };
  //komo->setVelocity( t_start, -1, "car_op", NULL, OT_eq, op_speed );
  mp::Interval always{{phase.time.from, -1}, phase.edge};
  W(komo).setVelocity( always, tree, "car_op", OT_eq, op_speed, 1e2);
}

void groundMergeBetween( const mp::Interval& phase, const mp::TreeBuilder& tree, const std::vector< std::string >& facts, KOMO_ext* komo, int verbose )
{
  auto car_before = facts[0];
  auto car_next = facts[1];

//  komo->setPosition( t_start+1, -1, "car_ego", car_next.c_str(), OT_sos, {-0.7, 0, 0} );
//  komo->setPosition( t_start+1, -1, car_before.c_str(), "car_ego", OT_sos, {-0.7, 0, 0} );

  mp::Interval future{{phase.time.to, -1}, phase.edge};
  W(komo).setPosition(future, tree, "car_ego", car_next.c_str(), OT_sos, { -0.7, 0, 0 }, 1e2);
  W(komo).setPosition(future, tree, car_before.c_str(), "car_ego", OT_sos, { -0.7, 0, 0 }, 1e2);

  //setKeepDistanceTask( phase+1, -1, komo, car_successors );
  //std::cout << "merge between " << car_before << " and " << car_next << std::endl;
}

//////////////Fixture////////////////
struct KomoPlannerFixture : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    // register symbols
    planner.registerTask( "__AGENT_0__look"      , groundLook );
    planner.registerTask( "__AGENT_0__overtake"  , groundOvertake );
    planner.registerTask( "__AGENT_0__follow"    , groundFollow );

    planner.registerTask( "__AGENT_1__accelerate", groundAccelerate );
    planner.registerTask( "__AGENT_1__continue"  , groundContinue );
    planner.registerTask( "__AGENT_1__slow_down" , groundSlowDown );

    planner.registerTask( "merge_between" , groundMergeBetween );
  }

  virtual void TearDown()
  {

  }

  InitGrounderMock initGrounder;
  mp::KOMOPlanner planner;
};

struct KomoPlannerSingleAgentFixture : public KomoPlannerFixture
{
protected:
  virtual void SetUp()
  {
    KomoPlannerFixture::SetUp();

    using namespace std::placeholders;

    // register symbols
    planner.registerInit( std::bind( &InitialGrounder::groundInitSingleAgent, &initGrounder, _1, _2, _3 ) );
  }
};

struct KomoPlannerDoubleAgentFixture : public KomoPlannerFixture
{
protected:
  virtual void SetUp()
  {
    KomoPlannerFixture::SetUp();

    using namespace std::placeholders;

    // register symbols
    planner.registerInit( std::bind( &InitialGrounder::groundInitDoubleAgent, &initGrounder, _1, _2, _3 ) );
  }
};
