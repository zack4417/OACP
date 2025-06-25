#include "komo_planner.h"
#include <gtest/gtest.h>
#include <tree_builder.h>
#include <test/trees.h>

using namespace std;
using namespace mp;

TEST(TreeBuilder, ClassCreation)
{
  auto tb = TreeBuilder(1.0, 0);
}

TEST(TreeBuilder, AddEdge)
{
  auto tb = TreeBuilder(1.0, 0);
  tb.add_edge(0, 1);

  EXPECT_EQ( tb.n_nodes(), 2 );
  EXPECT_EQ( tb.p(0,1), 1 );

  tb.add_edge(1, 2);

  EXPECT_EQ(tb.n_nodes(), 3);
  EXPECT_EQ(tb.p(0, 1), 1);
  EXPECT_EQ(tb.p(1, 2), 1);

  tb.add_edge(2, 3, 0.5);

  EXPECT_EQ(tb.n_nodes(), 4);
  EXPECT_EQ(tb.p(2, 3), 0.5);

  tb.add_edge(2, 4, 0.5);

  EXPECT_EQ(tb.n_nodes(), 5);
  EXPECT_EQ(tb.p(2, 4), 0.5);
}

TEST(TreeBuilder, GetLeafs)
{
  auto tb = build_simple_path_builder();
  EXPECT_EQ(std::vector<uint>({3, 4}), tb.get_leaves());
}

TEST(TreeBuilder, GetParents)
{
  auto tb = build_simple_path_builder();
  EXPECT_EQ(std::vector<uint>({}), tb.get_parents(0));
  EXPECT_EQ(std::vector<uint>({0}), tb.get_parents(1));
  EXPECT_EQ(std::vector<uint>({1}), tb.get_parents(2));
  EXPECT_EQ(std::vector<uint>({2}), tb.get_parents(3));
  EXPECT_EQ(std::vector<uint>({2}), tb.get_parents(4));
}

TEST(TreeBuilder, _GetBranch)
{
  auto tb = build_simple_path_builder();
  _Branch expected_branch_to_3;
  expected_branch_to_3.p = 0.5;
  expected_branch_to_3.leaf_id = 3;
  expected_branch_to_3.local_to_global.push_back(0);
  expected_branch_to_3.local_to_global.push_back(1);
  expected_branch_to_3.local_to_global.push_back(2);
  expected_branch_to_3.local_to_global.push_back(3);
  expected_branch_to_3.global_to_local.push_back(0);
  expected_branch_to_3.global_to_local.push_back(1);
  expected_branch_to_3.global_to_local.push_back(2);
  expected_branch_to_3.global_to_local.push_back(3);
  expected_branch_to_3.global_to_local.push_back(-1);
  EXPECT_EQ(expected_branch_to_3, tb._get_branch(3));

  _Branch expected_branch_to_4;
  expected_branch_to_4.p = 0.5;
  expected_branch_to_4.leaf_id = 4;
  expected_branch_to_4.local_to_global.push_back(0);
  expected_branch_to_4.local_to_global.push_back(1);
  expected_branch_to_4.local_to_global.push_back(2);
  expected_branch_to_4.local_to_global.push_back(4);
  expected_branch_to_4.global_to_local.push_back(0);
  expected_branch_to_4.global_to_local.push_back(1);
  expected_branch_to_4.global_to_local.push_back(2);
  expected_branch_to_4.global_to_local.push_back(-1);
  expected_branch_to_4.global_to_local.push_back(3);
  EXPECT_EQ(expected_branch_to_4, tb._get_branch(4));
}

TEST(TreeBuilder, GetBranchs)
{
  auto tb = build_simple_path_builder();
  EXPECT_EQ(2, tb.get_branches().size());
}

TEST(TreeBuilder, GetP)
{
  auto tree = build_3_edges_1_branching();

  EXPECT_EQ(tree.p(0, 3), 0.4);
}

TEST(TreeBuilder, GetPath)
{
  auto tree = build_3_edges_1_branching();

  auto path = tree.get_path(0, 3);

  EXPECT_EQ(path, std::vector<uint>({0, 1, 3}));
}

TEST(TreeBuilder, GetRoot)
{
  {
    auto tree = build_3_edges_1_branching();
    EXPECT_EQ(0, tree.get_root());
  }

  {
    TreeBuilder tree(1.0, 0);
    tree.add_edge(1, 2);
    EXPECT_EQ(1, tree.get_root());
  }
}

TEST(TreeBuilder, GetGrandChildren)
{
    auto tree = build_3_edges_1_branching();

    {
      auto cs = tree.get_grand_children(0, 2);
      EXPECT_EQ(cs, std::vector<uint>({2, 3}));
    }

    {
      auto cs = tree.get_grand_children(0, 3); // no problem if step too big
      EXPECT_EQ(cs, std::vector<uint>({}));
    }

    {
      auto cs = tree.get_grand_children(1, 0); // returns itslef if no step
      EXPECT_EQ(cs, std::vector<uint>({1}));
    }
}


TEST(TreeBuilder, GetVarsNSteps1)
{
  auto tb = build_simple_path_builder();

  // 0->1
  // order 0
  EXPECT_EQ(intA(1, 1, {0}), tb.get_vars({0, 1.0}, 3, 0));

  // order 1
  EXPECT_EQ(intA(1, 2, {-1, 0}), tb.get_vars({0, 1.0}, 3, 1));

  // order 2
  EXPECT_EQ(intA(1, 3, {-2, -1, 0}), tb.get_vars({0, 1.0}, 3, 2));

  // 1->2
  // order 0
  EXPECT_EQ(intA(1, 1, {1}), tb.get_vars({1.0, 2.0}, 3, 0));

  // order 1
  EXPECT_EQ(intA(1, 2, {0, 1}), tb.get_vars({1.0, 2.0}, 3, 1));

  // order 2
  EXPECT_EQ(intA(1, 3, {-1, 0, 1}), tb.get_vars({1.0, 2.0}, 3, 2));

  // 1->3
  // order 0
  EXPECT_EQ(intA(2, 1, {1, 2}), tb.get_vars({1.0, 3.0}, 3, 0));

  // order 1
  EXPECT_EQ(intA(2, 2, {0, 1,  1, 2}), tb.get_vars({1.0, 3.0}, 3, 1));

  // order 2
  EXPECT_EQ(intA(2, 3, {-1, 0, 1,  0, 1, 2}), tb.get_vars({1.0, 3.0}, 3, 2));

  // 1->4
  // order 0
  EXPECT_EQ(intA(2, 1, {1, 2}), tb.get_vars({1.0, 3.0}, 4, 0));

  // order 1
  EXPECT_EQ(intA(2, 2, {0, 1,  1, 2}), tb.get_vars({1.0, 3.0}, 4, 1));

  // order 2
  EXPECT_EQ(intA(2, 3, {-1, 0, 1,  0, 1, 2}), tb.get_vars({1.0, 3.0}, 4, 2));
}

TEST(TreeBuilder, GetVarsNSteps2)
{
  auto tb = build_simple_path_builder();
  uint steps = 2;
  // 0->1
  // order 0
  EXPECT_EQ(intA(2, 1, {0, 1}), tb.get_vars({0, 1.0}, 3, 0, steps));

  // order 1
  EXPECT_EQ(intA(2, 2, {-1, 0,  0, 1}), tb.get_vars({0, 1.0}, 3, 1, steps));

  // order 2
  EXPECT_EQ(intA(2, 3, {-2, -1, 0,  -1, 0, 1}), tb.get_vars({0, 1.0}, 3, 2, steps));

  // 1->3
  // order 0
  EXPECT_EQ(intA(4, 1, {2, 3, 4, 5}), tb.get_vars({1.0, 3.0}, 3, 0, steps));

  // order 1
  EXPECT_EQ(intA(4, 2, {1, 2,  2, 3,  3, 4,  4, 5}), tb.get_vars({1.0, 3.0}, 3, 1, steps));

  // order 2
  EXPECT_EQ(intA(4, 3, {0, 1, 2,  1, 2, 3,  2, 3, 4,  3, 4, 5}), tb.get_vars({1.0, 3.0}, 3, 2, steps));

  // 1->4
  // order 0
  EXPECT_EQ(intA(4, 1, {2, 3, 6, 7}), tb.get_vars({1.0, 3.0}, 4, 0, steps));
}

TEST(TreeBuilder, GetVarsNSteps10)
{
  auto tb = build_3_edges_1_branching();

  auto steps = 10;

  // order 1
  EXPECT_EQ(intA(10, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), tb.get_vars({0, 1.0}, 2, 0, steps));
  EXPECT_EQ(intA(10, 1, {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}), tb.get_vars({1.0, 2.0}, 3, 0, steps));
  EXPECT_EQ(intA(10, 2, {9, 10,  10, 11,  11, 12,  12, 13,  13, 14,  14, 15,  15, 16,  16, 17,  17, 18,  18, 19}), tb.get_vars({1.0, 2.0}, 2 , 1, steps));
  EXPECT_EQ(intA(10, 2, {9, 20,  20, 21,  21, 22,  22, 23,  23, 24,  24, 25,  25, 26,  26, 27,  27, 28,  28, 29}), tb.get_vars({1.0, 2.0}, 3, 1, steps));
}

TEST(TreeBuilder, GetVarsScalesNegativeTime)
{
  auto tree = build_3_edges_1_branching();

  auto stepss = {1, 10};
  auto orders = {0, 1, 2};

  for(const auto& steps: stepss)
  {
    for(const auto& order: orders)
    {
      auto a = tree.get_vars({0.0, 2.0}, 2, order, steps);
      auto b = tree.get_vars({0.0, -1.0}, 2, order, steps);

      EXPECT_EQ(a, b);
    }

    auto r = tree.get_scales({0.0, 2.0}, 2, steps);
    auto s = tree.get_scales({0.0, -1.0}, 2, steps);

    EXPECT_EQ(r, s);
  }
}

TEST(TreeBuilder, VarsScalesNegativeTimes)
{
  auto steps = 5;

  auto tb = build_3_edges_1_branching();

  auto vars = tb.get_vars({-2.0, -1.0}, 2, 0, steps);
  auto scales = tb.get_scales({-2.0, -1.0}, 2, steps);

  EXPECT_EQ(intA(), vars);
  EXPECT_EQ(arr(), scales);
}

TEST(TreeBuilder, GetVarsScalesOvertime)
{
  const auto steps = 5;

  TreeBuilder tree(1.0, 0);
  tree.add_edge(0, 1);

  auto a = tree.get_vars({2.0, 2.0}, 1, 2, steps);

  EXPECT_EQ(0, a.d0);
}

TEST(TreeBuilder, GetVarsNSteps102Branchings)
{
  auto tb = build_5_edges_2_branchings();

  auto steps = 10;

  // order 0 and 1
  EXPECT_EQ(intA(10, 1, {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}), tb.get_vars({2.0, 3.0}, 5, 0, steps));
  EXPECT_EQ(intA(10, 2, {29, 40,  40, 41,  41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49}), tb.get_vars({2.0, 3.0}, 5 , 1, steps));
}

TEST(TreeBuilder, GetVarsNStepsVarsConcatenations)
{
  auto tb = build_5_edges_2_branchings();

  auto steps = 10;
  auto orders = {0, 1, 2};

  for(const auto & order: orders)
  {
    // to 2
    auto vars01_to_2 = tb.get_vars({0, 1.0}, 2, order, 10);
    auto vars12_to_2 = tb.get_vars({1.0, 2.0}, 2, order, 10);

    auto vars02 = vars01_to_2;
    vars02.append(vars12_to_2);

    auto vars02_to_2 = tb.get_vars({0, 2.0}, 2, order, 10);

    EXPECT_EQ(vars02_to_2, vars02);

    // to 5
    auto vars01_to_5 = tb.get_vars({0, 1.0}, 5, order, 10);
    auto vars12_to_5 = tb.get_vars({1.0, 2.0}, 5, order, 10);
    auto vars23_to_5 = tb.get_vars({2.0, 3.0}, 5, order, 10);

    auto vars03 = vars01_to_5;
    vars03.append(vars12_to_5);
    vars03.append(vars23_to_5);

    auto vars03_to_5 = tb.get_vars({0, 3.0}, 5, order, 10);

    EXPECT_EQ(vars03_to_5, vars03);
  }
}

TEST(TreeBuilder, GetVarsNSteps5)
{
  auto steps = 5;

  auto tb = build_5_edges_1_branching();

  // order 0
  EXPECT_EQ(intA(15, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}), tb.get_vars({0, 3.0}, 3, 0, steps));
  EXPECT_EQ(intA(15, 1, {0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}), tb.get_vars({0, 3.0}, 5, 0, steps));
}

TEST(TreeBuilder, GetScalesVarsAtEnd)
{
  auto steps = 5;

  auto tb = build_3_edges_1_branching();

  EXPECT_EQ(intA(1, 1, {9}), tb.get_vars({2.0, 2.0}, 2, 0, steps));
  EXPECT_EQ(arr{0.6}, tb.get_scales({2.0, 2.0}, 2, steps));
  EXPECT_EQ(arr{0.6}, tb.get_scales({2.0, -1.0}, 2, steps));
  EXPECT_EQ(arr{0.6}, tb.get_scales({2.0, 5.0}, 2, steps));

  EXPECT_EQ(intA(1, 2, {8, 9}), tb.get_vars({2.0, 2.0}, 2, 1, steps));
  EXPECT_EQ(intA(1, 2, {8, 9}), tb.get_vars({2.0, -1.0}, 2, 1, steps));
  EXPECT_EQ(intA(1, 2, {8, 9}), tb.get_vars({2.0, 5.0}, 2, 1, steps));
}

TEST(TreeBuilder, PrefixOfUnCompressedSubtree)
{
  auto steps = 5;

  TreeBuilder tree(1.0, 0);

  tree.add_edge(1, 2);
  auto vars1 = tree.get_vars(TimeInterval{0, 1}, 2, 1, steps);

  EXPECT_EQ(intA(5, 2, {4, 5, 5, 6, 6, 7, 7, 8, 8, 9}), vars1);
}

// scales
TEST(TreeBuilder, GetScaleNSteps5)
{
  auto steps = 5;

  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1, 1.0);

  tb.add_edge(1, 2, 0.2);
  tb.add_edge(2, 3, 1.0);

  tb.add_edge(1, 4, 0.8);
  tb.add_edge(4, 5, 1.0);

  // order 0
  auto expect_near = [](const arr & arr_1, const arr & arr_2, double eps)
  {
    bool near = true;
    for(auto i = 0; i < arr_1.d0; ++i)
    {
      near = near && fabs(arr_1(i) - arr_2(i)) < eps;
    }

    EXPECT_TRUE(near);
  };

  expect_near(arr(15, {1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}), tb.get_scales({0, 3.0}, 3, steps), 0.001);
  expect_near(arr(15, {1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8}), tb.get_scales({0, 3.0}, 5, steps), 0.001);
  expect_near(arr(10, {1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8}), tb.get_scales({0, 2.0}, 5, steps), 0.001);
  expect_near(arr(10, {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}), tb.get_scales({1.0, 3.0}, 3, steps), 0.001);
}

// specs
TEST(TreeBuilder, SpecUntilLeaves)
{
  auto steps = 5;

  auto tb = build_3_edges_1_branching();

  auto spec0 = tb.get_spec({0.0, -1.0}, Edge{0, 1}, 0, steps);
  auto spec1 = tb.get_spec({0.0, -1.0}, Edge{0, 1}, 1, steps);
  TaskSpec expected_spec0{intA(15, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}),
                          arr{1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4}};
  TaskSpec expected_spec1{intA(15, 2, {-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 4, 10, 10, 11, 11, 12, 12, 13, 13, 14}),
                          arr{1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4}};
  EXPECT_EQ(expected_spec0, spec0);
  EXPECT_EQ(expected_spec1, spec1);
}

TEST(TreeBuilder, SpecInterval)
{
  auto steps = 5;

  auto tb = build_3_edges_1_branching();

  auto spec0 = tb.get_spec({0.5, 1.5}, Edge{0, 1}, 0, steps);
  auto spec1 = tb.get_spec({0.5, 1.5}, Edge{0, 1}, 1, steps);
  TaskSpec expected_spec0{intA(7, 1, {2, 3, 4, 5, 6, 10, 11}),
                                  arr{1.0, 1.0, 1.0, 0.6, 0.6, 0.4, 0.4}};
  TaskSpec expected_spec1{intA(7, 2, {1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 4, 10, 10, 11}),
                                   arr{1.0, 1.0, 1.0, 0.6, 0.6, 0.4, 0.4}};
  EXPECT_EQ(expected_spec0, spec0);
  EXPECT_EQ(expected_spec1, spec1);
}

TEST(TreeBuilder, ScalesWithP0)
{
  auto steps = 5;

  auto tb = build_3_edges_1_branching();
  //tb.set

}

TEST(TreeBuilder, Step)
{
  auto steps = 5;

  auto tb = build_3_edges_1_branching();

  EXPECT_EQ(1, tb.get_step(0.5, {0, 1}, 5)); // should be 2??
}

TEST(Vars, Step)
{
  Vars var;
  var.k_order = 2;
  var.microSteps = 5;
  var.order0 = intA(5, 1, {0, 1, 2, 3, 4});

  EXPECT_EQ(2, var.getPreviousStep(3));
}

TEST(TreeBuilder, GetBranch)
{
  auto tree = build_3_edges_1_branching();

  auto b1 = tree.get_branch(2);

  EXPECT_EQ(3, b1.n_nodes());
  EXPECT_EQ(1, b1.get_leaves().size());
  EXPECT_EQ(2, b1.get_leaves()[0]);

  auto b2 = tree.get_branch(3);

  EXPECT_EQ(3, b2.n_nodes());
  EXPECT_EQ(1, b2.get_leaves().size());
  EXPECT_EQ(3, b2.get_leaves()[0]);
}

TEST(TreeBuilder, HasNode)
{
  auto tree = build_3_edges_1_branching();
  auto b1 = tree.get_branch(2);

  EXPECT_TRUE(tree.has_node(0));
  EXPECT_FALSE(tree.has_node(5));
  EXPECT_FALSE(b1.has_node(3));
}

TEST(TreeBuilder, Print)
{
  auto tree = build_3_edges_1_branching();
  std::cout << tree << std::endl;
}

TEST(TreeBuilder, CompressedVar)
{
  auto tree = build_3_edges_1_branching();

  auto b1 = tree.get_branch(2);
  auto b2 = tree.get_branch(3);
  auto var = b2.get_vars({0, 2}, 3, 0, 2);

  Mapping mapping;
  auto c_b2 = b2.compressed(mapping);

  std::cout << b2 << std::endl;
  std::cout << c_b2 << std::endl;

  auto compressed_var = c_b2.get_vars({0, 2}, mapping.orig_to_compressed(3), 0, 2);

  std::cout << var << std::endl;
  std::cout << compressed_var << std::endl;

  EXPECT_EQ(intA(4, 1, {0, 1, 4, 5}), var);
  EXPECT_EQ(intA(4, 1, {0, 1, 2, 3}), compressed_var);
}

TEST(TreeBuilder, VariousTestsOnRealExample)
{
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2);
  tb.add_edge(2, 3);
  tb.add_edge(2, 8);
  tb.add_edge(8, 9);
  tb.add_edge(9, 10);
  tb.add_edge(10, 11);
  tb.add_edge(11, 12);
  tb.add_edge(3, 4);
  tb.add_edge(4, 5);
  tb.add_edge(5, 6);
  tb.add_edge(6, 7);

  auto b1 = tb.get_branch(7);
  auto b2 = tb.get_branch(12);

  EXPECT_EQ(8, b1.n_nodes());
  EXPECT_EQ(1, b1.get_leaves().size());
  EXPECT_EQ(7, b1.get_leaves()[0]);

  EXPECT_EQ(8, b2.n_nodes());
  EXPECT_EQ(1, b2.get_children(2).size());
  EXPECT_EQ(1, b2.get_leaves().size());
  EXPECT_EQ(12, b2.get_leaves()[0]);

  TimeInterval it;
  it.from = 6;
  it.to = 7;

  b2.get_vars(it, 12, 2, 20);

  Mapping mapping;
  auto c_b2 = b2.compressed(mapping);

  //std::cout << c_b2 << std::endl;

  auto compressed_var = c_b2.get_vars({0, 2}, mapping.orig_to_compressed(12), 0, 2);
}

////////////////////////////////
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}

