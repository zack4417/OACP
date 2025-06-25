#include <skeleton.h>

#include <gtest/gtest.h>

#include <fstream>

#include <boost/filesystem.hpp>

Policy build_3_nodes_Policy()
{
  PolicyNodeData rootData;
  //rootData.beliefState = { 1.0 };
  Policy::GraphNodeTypePtr root = Policy::GraphNodeType::root( rootData );

  // child 1
  PolicyNodeData childData1;
  //childData.beliefState = { 1.0 };
  childData1.leadingKomoArgs = { "komoAction", "X", "Y", "Z" };
  root->makeChild( childData1 );

  // child 1
  PolicyNodeData childData2;
  //childData.beliefState = { 1.0 };
  childData2.leadingKomoArgs = { "komoAction", "A", "B", "C" };
  root->makeChild( childData2 );

  // policy
  Policy p( root );
  p.setValue( 1.0 );

  return p;
}

// QResult
TEST(PolicyNodeData, DefaultConstruction) {
  PolicyNodeData data;

  ASSERT_EQ( data.status, PolicyNodeData::UNPLANNED );
}


// QResult
TEST(QResult, DefaultConstruction) {
  QResult result;

  ASSERT_EQ( result.stepsPerPhase(), 0.0 );
  ASSERT_EQ( result.nWorlds(), 0 );
}

TEST(QResult, Construction) {
  QResult result(2, {1, 0}, 20);

  ASSERT_EQ( result.stepsPerPhase(), 20 );
  ASSERT_EQ( result.nWorlds(), 2 );
  ASSERT_EQ( result.qmask(0), 1 );
}

TEST(QResult, CopyConstruction) {
  QResult result(2, {1, 0}, 20);
  QResult result2(result);

  ASSERT_EQ( result2.stepsPerPhase(), 20 );
  ASSERT_EQ( result2.nWorlds(), 2 );
}

TEST(QResult, Copy) {
  QResult result(2, {1, 0}, 20);
  QResult result2(3, {1, 0}, 20);

  result2 = result;

  ASSERT_EQ( result2.stepsPerPhase(), 20 );
  ASSERT_EQ( result2.nWorlds(), 2 );
}

TEST(QResult, setGetTrajectory) {
  QResult result(2, {1, 0}, 20);

  result.createTrajectory(0, 2);
  result.createTrajectory(1, 1);

  result.setQ(0, 0, {0.0, 0.0});
  result.setQ(0, 1, {0.1, 0.1});

  result.setQ(1, 0, {1.0, 1.0});

  ASSERT_EQ( result.q(0, 0), std::vector< double >({0.0, 0.0}) );
  ASSERT_EQ( result.q(0, 1), std::vector< double >({0.1, 0.1}) );
  ASSERT_EQ( result.qmask(0), 1 );
  ASSERT_EQ( result.qmask(1), 0 );
  ASSERT_EQ( result.nSteps(0), 2 );

  ASSERT_EQ( result.q(1, 0), std::vector< double >({1.0, 1.0}) );
  ASSERT_EQ( result.nSteps(1), 1 );
}

// GraphNode
TEST(Policy, SaveLoad) {
  Policy p;
  //
  const std::string filename( "policy" );
  const double value = 1.0;
  const Policy::StatusType status( Policy::INFORMED );
  QResult qr(2, {1, 0}, 0.1);
  qr.createTrajectory(0, 2);
  qr.createTrajectory(1, 1);

  qr.setQ(0, 0, {0.0, 0.0});
  qr.setQ(0, 1, {0.1, 0.1});
  //

  p.setValue( 1.0 );
  p.setQResult( qr );
  p.setStatus( status );
  p.save( filename );

  Policy pp;
  pp.load( filename );

  ASSERT_EQ( pp.value(), value );
  ASSERT_EQ( pp.qresult(), qr );
  ASSERT_EQ( pp.status(), status );
}

TEST(Policy, Equality) {
  // root
  PolicyNodeData rootData;
  Policy::GraphNodeTypePtr root = Policy::GraphNodeType::root( rootData );
  Policy p( root );
  //
  const std::string filename( "policy" );
  const double value = 1.0;
  const Policy::StatusType status( Policy::INFORMED );
  //

  p.setValue( 1.0 );
  p.setStatus( status );
  p.save( filename );

  Policy pp;
  pp.load( filename );

  ASSERT_EQ( pp, p );
}

TEST(Policy, NonEquality) {
  // root
  PolicyNodeData rootData;
  Policy::GraphNodeTypePtr root = Policy::GraphNodeType::root( rootData );
  Policy p( root );
  //
  const std::string filename( "policy" );
  const double value = 1.0;
  const Policy::StatusType status( Policy::INFORMED );
  //

  p.setValue( 1.0 );
  p.setStatus( status );
  p.save( filename );

  Policy pp;

  ASSERT_NE( pp, p );
}

TEST(Policy, HashingEquality) {
  // root
  PolicyNodeData rootData;
  Policy::GraphNodeTypePtr root = Policy::GraphNodeType::root( rootData );
  Policy p( root );
  //
  const std::string filename( "policy" );
  const double value = 1.0;
  const Policy::StatusType status( Policy::INFORMED );
  //

  p.setValue( 1.0 );
  p.setStatus( status );
  p.save( filename );

  Policy pp;
  pp.load( filename );

  ASSERT_EQ( pp.hash(), p.hash() );
}

TEST(Policy, HashingInequality) {
  // root
  PolicyNodeData rootData;
  Policy::GraphNodeTypePtr root = Policy::GraphNodeType::root( rootData );
  Policy p( root );
  //
  const std::string filename( "policy" );
  const double value = 1.0;
  const Policy::StatusType status( Policy::INFORMED );
  //

  p.setValue( 1.0 );
  p.setStatus( status );
  p.save( filename );

  Policy pp;

  ASSERT_NE( pp.hash(), p.hash() );
}

TEST(Policy, NumberOfNodes) {
  // root
  Policy p = build_3_nodes_Policy();

  ASSERT_EQ(p.nNodes(), 3);
}

// GraphNode
TEST(Policy, SaveToGraph) {

  const std::string filename( "policy_graph.gv" );

  Policy p = build_3_nodes_Policy();

  p.saveToGraphFile( "policy_graph.gv" );

  ASSERT_TRUE( boost::filesystem::exists( filename ) );
}

//
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
