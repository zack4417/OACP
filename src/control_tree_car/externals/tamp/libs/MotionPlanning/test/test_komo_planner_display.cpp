#include "komo_planner_fixture.h"

/////////////////////SINGLE AGENT OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent1WDisplay)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.display( policy, 3.0 ) );
}

/////////////////////SINGLE AGENT PARTIALLY OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerSingleAgentFixture, PlanSingleAgent2WDisplay)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-single-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.display( policy, 10.0 ) );
}

/////////////////////TWO AGENTS FULLY OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents1WDisplay)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-1w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.display( policy, 10.0 ) );
}

TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents1WTweakedDisplay)
{
  planner.setKin( "data/LGP-overtaking-kin.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-1w-tweaked.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.display( policy, 10.0 ) );
}

/////////////////////TWO AGENTS PARTIALLY OBSERVABLE/////////////////////////////
TEST_F(KomoPlannerDoubleAgentFixture, PlanTwoAgents2WDisplay)
{
  planner.setKin( "data/LGP-overtaking-kin-2w_bis.g" );

  Policy policy;
  policy.load( "data/LGP-overtaking-double-agent-2w.po" );

  MotionPlanningParameters po( policy.id() );
  po.setParam( "type", "jointPath" );

  EXPECT_NO_THROW( planner.display( policy, 10.0 ) );
}

////////////////////////////////
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}

