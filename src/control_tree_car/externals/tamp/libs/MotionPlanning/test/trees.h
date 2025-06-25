#include <tree_builder.h>

using namespace std;
using namespace mp;

TreeBuilder build_simple_path_builder()
{
  /*   0
   *   |
   *   1
   *   |
   *   2
   *  / \
   * 3   4
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2);
  tb.add_edge(2, 3, 0.5);
  tb.add_edge(2, 4, 0.5);
  return tb;
}

TreeBuilder build_5_edges_2_branchings()
{
  /*   0
   *   |
   *   1
   *  / \
   * 2   3
   *    / \
   *   4   5
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2, 0.4);
  tb.add_edge(1, 3, 0.6);
  tb.add_edge(3, 4, 0.5);
  tb.add_edge(3, 5, 0.5);
  return tb;
}

TreeBuilder build_5_edges_1_branching()
{
  /*   0
   *   |
   *   1
   *  / \
   * 2   4
   * |   |
   * 3   5
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2);
  tb.add_edge(2, 3);
  tb.add_edge(1, 4);
  tb.add_edge(4, 5);
  return tb;
}

TreeBuilder build_3_edges_1_branching()
{
  /*   0
   *   |
   *   1
   *  / \
   * 2   3
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2, 0.6);
  tb.add_edge(1, 3, 0.4);
  return tb;
}

TreeBuilder build_3_edges_1_branching_sub_1()
{
  /*   0
   *   |
   *   1
   *  /
   * 2
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2, 0.6);
  return tb;
}

TreeBuilder build_3_edges_1_branching_sub_2()
{
  /*
   *   1
   *    \
   *     3
   */
  TreeBuilder tb(1.0, 1);
  tb.add_edge(1, 3, 0.4);
  return tb;
}

TreeBuilder build_2_linear_edges()
{
  /*   0
   *   |
   *   1
   *   |
   *   2
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2);
  return tb;
}

TreeBuilder build_3_linear_edges()
{
  /*   0
   *   |
   *   1
   *   |
   *   2
   *   |
   *   3
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  tb.add_edge(1, 2);
  tb.add_edge(2, 3);
  return tb;
}

TreeBuilder build_2_linear_edges_sub_1()
{
  /*   0
   *   |
   *   1
   */
  TreeBuilder tb(1.0, 0);
  tb.add_edge(0, 1);
  return tb;
}

TreeBuilder build_2_linear_edges_sub_2()
{
  /*
   *
   *   1
   *   |
   *   2
   */
  TreeBuilder tb(1.0, 1);
  tb.add_edge(1, 2);
  return tb;
}
