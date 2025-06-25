#include <graph_node.h>

#include <gtest/gtest.h>

#include <fstream>

// GraphNode
TEST(GraphNode, RootNode) {
  auto root = GraphNode< double >::root( 0.0 );
  ASSERT_EQ( root->isRoot(), true );
}

TEST(GraphNode, ChildNodeNotRoot) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  ASSERT_EQ( ! child->isRoot(), true );
}

TEST(GraphNode, ChildId) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child1 = root->makeChild( 1.0 );
  auto child2 = root->makeChild( 1.0 );
  ASSERT_EQ( child2->id(), 2 );
}

TEST(GraphNode, ChildDepth) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child1 = root->makeChild( 1.0 );
  ASSERT_EQ( child1->depth(), 1 );
}

TEST(GraphNode, SingleChildHasNoSibling) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  ASSERT_EQ( child->siblings().size(), 0 );
}

TEST(GraphNode, DoubleChildHasOneSibling) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child1 = root->makeChild( 1.0 );
  auto child2 = root->makeChild( 1.0 );
  ASSERT_EQ( child1->siblings().size(), 1 );
  ASSERT_EQ( child2->siblings().size(), 1 );
}

TEST(GraphNode, DoubleChildSiblingIdentity) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child1 = root->makeChild( 1.0 );
  auto child2 = root->makeChild( 1.0 );
  ASSERT_EQ( child1->siblings().back(), child2 );
  ASSERT_EQ( child2->siblings().back(), child1 );
}

TEST(GraphNode, ChildNodeRegisteredInParent) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  ASSERT_EQ( root->children().size(), 1 );
}

TEST(GraphNode, ChildOfChild) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  auto childchild = child->makeChild( 1.0 );
  ASSERT_EQ( child->children().size(), 1 );
}

TEST(GraphNode, ChildOfChildId) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  auto childchild = child->makeChild( 1.0 );
  ASSERT_EQ( childchild->id(), 2 );
}

TEST(GraphNode, ReferenceCountOnRoot) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  ASSERT_EQ( root.use_count(), 1 );
}

TEST(GraphNode, GraphNodeData) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  ASSERT_EQ( child->data(), 1.0 );
}

TEST(GraphNode, AddExistingChild) { // for graph
  auto root = GraphNode< double >::root( 0.0 );
  auto child = root->makeChild( 1.0 );
  auto childchild = child->makeChild( 1.0 );
  childchild->addExistingChild( child );
  ASSERT_EQ( childchild->children().size(), 1 );
  ASSERT_EQ( child->parents().size(), 2 );
}

TEST(GraphNode, RemoveChild) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child1 = root->makeChild( 1.0 );
  auto child2 = root->makeChild( 1.0 );
  root->removeChild( child2 );
  ASSERT_EQ( root->children().size(), 1 );
  ASSERT_EQ( child2->parents().size(), 0 );
}

TEST(GraphNode, ClearChildren) {
  auto root = GraphNode< double >::root( 0.0 );
  auto child1 = root->makeChild( 1.0 );
  auto child2 = root->makeChild( 1.0 );
  root->clearChildren();
  ASSERT_EQ( root->children().size(), 0 );
  ASSERT_EQ( child1->parents().size(), 0 );
  ASSERT_EQ( child2->parents().size(), 0 );
}

TEST(GraphNode, SerializeSingleNode) {
  auto root = GraphNode< double >::root( 1.0 );

  const std::string filename( "s_root" );
  std::ofstream ofs( filename );

  // save data to archive
  {
    boost::archive::text_oarchive oa(ofs);
    oa << root;
  }

  // restore
  GraphNode< double >::ptr newRoot;
  {
    // create and open an archive for input
    std::ifstream ifs( filename );
    boost::archive::text_iarchive ia(ifs);
    ia >> newRoot;
  }

  ASSERT_EQ( root->data(), newRoot->data() );
}

TEST(GraphNode, SerializeHierarchy) {
  auto root = GraphNode< double >::root( 1.0 );
  auto child1 = root->makeChild( 2.0 );
  auto child2 = root->makeChild( 3.0 );

  auto childchild1 = child1->makeChild( 4.0 );

  const std::string filename( "s_hierarchy" );
  std::ofstream ofs( filename );

  // save data to archive
  {
    boost::archive::text_oarchive oa(ofs);
    oa << root;
  }

  // restore
  GraphNode< double >::ptr newRoot;
  {
    // create and open an archive for input
    std::ifstream ifs( filename );
    boost::archive::text_iarchive ia(ifs);
    ia >> newRoot;
  }

  ASSERT_EQ( root->data(), newRoot->data() );
  ASSERT_EQ( root->children().size(), newRoot->children().size() );
  ASSERT_EQ( root->children().front()->children().size(), newRoot->children().front()->children().size() );
  ASSERT_EQ( root->children().front()->children().front()->data(), newRoot->children().front()->children().front()->data() );
}


//
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
