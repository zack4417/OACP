#pragma once

#include <Core/array.h>
#include <tree_builder.h>
#include <stdlib.h>
#include <ostream>
#include <list>

namespace mp
{

// generates the edges of the full tree that can interact with the subtree (includes the path going to the subtree, and the branches after)
// the pair contains the edge in the ful tree, and its closest edge in the subtree
std::vector<std::pair<Edge, Edge>> interactingEdges(const TreeBuilder& tree, const TreeBuilder& subtree);

struct SubTreeGen
{
  virtual bool finished() const = 0;
  virtual TreeBuilder next() = 0;
};

struct BranchGen : public SubTreeGen
{
  BranchGen(const TreeBuilder& tree)
    : tree(tree)
  {
    leaves = tree.get_leaves();
  }

  bool finished() const override
  {
    return (index == leaves.size());
  }

  TreeBuilder next() override
  {
    CHECK(!finished(), "finished generator");

    return tree.get_branch(leaves[index++]);
  }

  const TreeBuilder& tree;
  std::vector<uint> leaves;
  uint index = 0;
};

struct SubTreesAfterFirstBranching : public SubTreeGen // common trunk
{
  SubTreesAfterFirstBranching(const TreeBuilder& tree);

  bool finished() const override;

  TreeBuilder next() override;

  const TreeBuilder& tree;
  std::vector<uint> path_to_source;
  std::vector<uint> sources;
  uint index = 0;
};

struct LinearSplit : public SubTreeGen // NOT EFFICIENT
{
  LinearSplit(const TreeBuilder& tree, uint n);

  bool finished() const override;

  TreeBuilder next() override;

  const TreeBuilder& tree;
  uint n;
  std::vector<std::pair<uint, uint>> splits;
  uint index{0};
};

struct GeneratorFactory
{
  /**
   * @brief create generator for decomposing the tree
   * @param name
   * @param n ideal number of subtrees
   * @param tree
   * @return
   */
  std::shared_ptr<SubTreeGen> create(const std::string& name, uint n, const TreeBuilder& tree) const
  {
    if(name == "BranchGen")
    {
      return std::make_shared<BranchGen>(tree);
    }
    else if(name == "SubTreesAfterFirstBranching")
    {
      return std::make_shared<SubTreesAfterFirstBranching>(tree);
    }
    else if(name == "LinearSplit")
    {
      return std::make_shared<LinearSplit>(tree, n);
    }
  }
};

Vars fuse(const std::vector<Vars> & vars);

std::vector< std::tuple< TreeBuilder, TreeBuilder, Mapping > > get_subproblems(const std::shared_ptr<SubTreeGen> & gen);

/**
 * @brief get_all_vars
 * @param subproblems
 * @param steps
 * @return vars (uncompressed and compressed, uncompressed-fused)
 */
std::tuple< std::vector<Vars>, std::vector<Vars>, Vars> get_all_vars(const std::vector< std::tuple< TreeBuilder, TreeBuilder, Mapping > > & subproblems, uint steps);

}

