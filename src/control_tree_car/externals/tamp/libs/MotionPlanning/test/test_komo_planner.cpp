#include "komo_planner_fixture.h"

/////////////////////////////////////

TEST_F(KomoPlannerSingleAgentFixture, ParseKinFileDoesntThrow1w)
{
  EXPECT_NO_THROW( planner.setKin( "data/LGP-overtaking-kin.g" ) );
}

TEST_F(KomoPlannerSingleAgentFixture, ParseKinFileDoesntThrow2w)
{
  EXPECT_NO_THROW( planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" ) );
}

//TEST_F(KomoPlannerSingleAgentFixture, FreePrefixLinearSplit) // NOT EFFICIENT
//{
//  planner.setKin( "data/LGP-overtaking-kin.g" );

//  Policy policy;
//  policy.load( "data/LGP-overtaking-single-agent-1w.po" );

//  MotionPlanningParameters po( policy.id() );
//  po.setParam( "type", "ADMMCompressed" );
//  po.setParam( "decompositionStrategy", "LinearSplit" ); // ENABLE HERE AND PUT LINEAR TRAJ TO CONTINUE WORKING
//  po.setParam( "nJobs", "2" );

//  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
//  EXPECT_TRUE( initGrounder.nInitSingleAgent > 1 );
//}

TEST_F(KomoPlannerSingleAgentFixture, InitialGroundingIsCalledWithAtEachStageForMarkovianPaths)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "markovJointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( initGrounder.nInitSingleAgent, 6 );  // 3 (pose) + 3(markov)
}

TEST_F(KomoPlannerSingleAgentFixture, InitialGroundingIsCalledWithAtEachStageForJointPaths)
{
  planner.setKin(  "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( initGrounder.nInitSingleAgent, 10 );  // 6 (pose) + 2 (paths) + 2(joint paths)
}

TEST_F(KomoPlannerSingleAgentFixture, InitialGroundingIsCalledWithNoRandomVector)
{
  planner.setKin(  "data/LGP-overtaking-kin-2w_bis.g" );
  auto randomVec = planner.drawRandomVector();

  EXPECT_EQ( randomVec.size(), 0 );
}

TEST_F(KomoPlannerSingleAgentFixture, InitialGroundingIsCalledWithRandomVectorOfCorrectSize)
{
  planner.setKin( "data/LGP-merging-kin.g" );
  auto randomVec = planner.drawRandomVector();

  EXPECT_EQ( randomVec.size(), 2 );
}

TEST_F(KomoPlannerSingleAgentFixture, TestRandomVectorOverride)
{
  planner.setKin( "data/LGP-merging-kin.g" );
  std::vector<double> randomVec = planner.drawRandomVector({1.0, 0.5});

  EXPECT_EQ( randomVec, std::vector<double>({1.0, 0.5}) );
}

/////////////////////SINGLE AGENT OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent1WMarkovianPath)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "markovJointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent1WJointPath)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

/////////////////////SINGLE AGENT PARTIALLY OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent2WMarkovianPath)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "markovJointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent2WMarkovianPath_NodeSetAsPlanned)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "markovJointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );

  for(const auto & leaf: policy.leaves())
  {
    auto path = getPathTo( leaf );
    for( const auto & n: path )
    {
      if(n->data().markovianReturn != 0)
        EXPECT_EQ( n->data().status, PolicyNodeData::INFORMED );
      else
        EXPECT_EQ( n->data().status, PolicyNodeData::UNPLANNED );
    }
  }
}

TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent2WJointPath)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

/////////////////////TWO AGENTS FULLY OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents1WMarkovianPath)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "markovJointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents1WJointPath)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

/////////////////////TWO AGENTS PARTIALLY OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents2WMarkovianPath)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "markovJointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents2WJointPath)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
}

TEST_F(KomoPlannerSingleAgentFixture, QResultSetAfterJointPathPlanning1W)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
  EXPECT_EQ( policy.qresult().nWorlds(), 1 );
  EXPECT_GE( policy.qresult().nSteps(0), 1 );
}

TEST_F(KomoPlannerSingleAgentFixture, QResultSetAfterJointPathPlanning2W)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.solveAndInform( po, policy ) );
  EXPECT_EQ( policy.status(), Policy::INFORMED );
  EXPECT_EQ( policy.qresult().nWorlds(), 2 );
  EXPECT_GE( policy.qresult().nSteps(1), 1 );
}

////////////////////////////////
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
