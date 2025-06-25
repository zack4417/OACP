#pragma once

#include <stdlib.h>
#include <Core/array.h>
#include <ostream>
#include <list>

namespace mp
{
struct _Branch
{
    std::vector< int > local_to_global;
    std::vector< int > global_to_local;
    double p; // probability to reach the leaf
    uint leaf_id;

    static _Branch computeMicroStepBranch(const _Branch& a, int stepsPerPhase);
    static _Branch linearTrajectory(int T);
};

bool operator==(const _Branch& a, const _Branch& b);
bool operator<(const _Branch& a, const _Branch& b);

bool empty_row(const arr & m, uint i);
bool empty_col(const arr & m, uint j);

struct Vars // Branch
{
  intA order0;
  intA order1;
  intA order2;
  uint microSteps{0};
  uint k_order = 2;

  Vars() = default;
  Vars(const Vars&) = default;
  Vars(const intA & order0, const intA& order1, const intA& order2, uint microSteps)
    : order0(order0)
    , order1(order1)
    , order2(order2)
    , microSteps(microSteps)
  {

  }

  intA getVars(double from, double to, uint order) const
  {
    CHECK(from>=0.0, "invalid start time");
    CHECK(to<order0.d0*microSteps, "invalid end time");

    uint indexFrom = from * microSteps;
    uint indexTo = to > 0 ? to * microSteps : (*this)[order].d0;

    if(indexFrom > 1) indexFrom--;
    indexTo--;

    CHECK(indexFrom>=0, "invalid start index time");
    CHECK(indexTo<order0.d0, "invalid end index time");

    return (*this)[order].sub(indexFrom, indexTo, 0, -1);
  }

  int getStep(double time) const
  {
    //int step = (floor(time*double(microSteps) + .500001))-1;
    int step = time * microSteps - 1;
    return order0(step, 0);
  }

  int getPreviousStep(uint step) const
  {
    auto it = std::find(order0.begin(), order0.end(), step);

    if(it != order0.end())
      return *(--it);

    return -1;
  }

  const intA& operator[](std::size_t i) const
  {
    CHECK(i <= 2, "wrong order request!");
    switch(i)
    {
      case 0:
        return order0;
      case 1:
        return order1;
      case 2:
        return order2;
      default:
        break;
    }
  }

  intA& order(std::size_t i)
  {
    return const_cast<intA&>((*this)[i]);
  }
};

struct TimeInterval
{
  double from;
  double to;
};

struct Edge // edge in Tree
{
  uint from;
  uint to;
};

struct Interval
{
  TimeInterval time;
  Edge edge;
};

struct TaskSpec
{
  intA vars;
  arr scales;
};

bool operator==(const TaskSpec& a, const TaskSpec& b);

struct Mapping
{
  intA compressed_to_orig;
  intA orig_to_compressed;
};

class TreeBuilder
{
public:
  TreeBuilder(double p, uint d);

  arr adjacency_matrix() const {return adjacency_matrix_;}
  uint n_nodes() const;
  bool has_node(uint n) const;
  double p() const { return p_; }
  uint d() const { return d_; }
  double p(uint from, uint to) const;
  uint get_root() const; // typically 0 but can be something else in case of uncompressed subtree
  std::vector<uint> get_nodes() const;
  std::vector<uint> get_leaves() const;
  std::vector<uint> get_parents(uint node) const;
  std::vector<uint> get_children(uint node) const;
  std::vector<uint> get_grand_children(uint node, uint step) const;
  std::vector<uint> get_grand_children_with_backtracking(uint node, uint step) const;
  std::vector<uint> get_leaves_from(uint node) const;
  std::vector<uint> get_path(uint from, uint to) const;
  _Branch _get_branch(uint leaf) const;
  std::vector<_Branch> get_branches() const;
  TreeBuilder get_branch(uint leaf) const;
  TreeBuilder get_subtree_from(uint node) const;
  TreeBuilder compressed(Mapping & mapping) const; // remove non existing nodes, de-facto changing ids mapping is a mapping from compressed to initial tree
  intA get_vars0(const TimeInterval& interval, const _Branch& branch, uint steps=1) const;
  intA get_vars(const TimeInterval& interval, uint leaf, uint order=2, uint steps=1) const;
  arr get_scales(const TimeInterval& interval, uint leaf, uint steps=1) const;
  TaskSpec get_spec(uint order=2, uint steps=1) const;
  TaskSpec get_spec(const TimeInterval& interval, const Edge& start_edge, uint order=2, uint steps=1) const;
  TaskSpec get_spec(const TimeInterval& interval, const std::vector<uint> & leaves, uint order=2, uint steps=1) const;
  int get_step(double time, const Edge& edge, uint steps) const;

  void add_edge(uint from, uint to, double p = 1.0);

private:
  arr adjacency_matrix_;
  const double p_; // existence probability of the tree (makes sense when built out of a subtree)
  const uint d_; // depth offset
};

std::ostream& operator<<(std::ostream& os, const TreeBuilder & tree);
bool operator==(const Edge& a, const Edge& b);
bool operator==(const TreeBuilder& a, const TreeBuilder& b);

}

