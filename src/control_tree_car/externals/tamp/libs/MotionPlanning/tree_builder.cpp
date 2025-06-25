#include <tree_builder.h>
#include <list>

namespace mp
{
bool operator==(const _Branch& a, const _Branch& b)
{
  return (a.p == b.p) && (a.local_to_global == b.local_to_global) && (a.global_to_local == b.global_to_local) && (a.leaf_id == b.leaf_id);
}

bool operator<(const _Branch& a, const _Branch& b)
{
  return a.leaf_id < b.leaf_id;
}

bool operator==(const TaskSpec& a, const TaskSpec& b)
{
  return (a.vars == b.vars) && (a.scales == b.scales);
}

static TimeInterval normalize(const TimeInterval & interval, const _Branch & branch, uint steps)
{
  auto from = interval.from;
  auto to = interval.to;

  // handle the case of to == -1
  if(from > to && to < 0  || (to > branch.local_to_global.size() - 1))
  {
    to = branch.local_to_global.size() - 1;
  }

  // handle the case of from == to
  if(from == to)
  {
    from -= (1.0 - 0.0001)/ steps;
  }

  if(from < 0 && to > 0)
  {
    from = 0;
  }

  return {from, to};
}

bool empty_row(const arr & m, uint i)
{
  CHECK(i < m.d0, "wrong row dimensions");

  for(auto j = 0; j < m.d1; ++j)
  {
    if(m(i, j) != 0)
    {
      return false;
    }
  }

  return true;
};

bool empty_col(const arr & m, uint j)
{
  CHECK(j < m.d1, "wrong col dimensions");

  for(auto i = 0; i < m.d0; ++i)
  {
    if(m(i, j) != 0)
    {
      return false;
    }
  }

  return true;
};


TreeBuilder::TreeBuilder(double p, uint d)
  : adjacency_matrix_ ( arr(uint(0), uint(0)) )
  , p_(p)
  , d_(d)
{

}

uint TreeBuilder::n_nodes() const
{
  return get_nodes().size();
}

bool TreeBuilder::has_node(uint n) const
{
  //if(n == 0 && adjacency_matrix_.d1) return true;
  if(n >= adjacency_matrix_.d1) return false;

  return !empty_col(adjacency_matrix_, n) || !empty_row(adjacency_matrix_, n);
}

double TreeBuilder::p(uint from, uint to) const
{
  double p = 1.0;

  auto path = get_path(from, to);

  for(auto i = 1; i < path.size(); ++i)
  {
    p *= adjacency_matrix_(path[i-1], path[i]);
  }

  return p;
}

uint TreeBuilder::get_root() const
{
  std::vector<uint> roots;

  for(auto i = 0; i < adjacency_matrix_.d0; ++i)
  {
    if(!empty_row(adjacency_matrix_, i) && empty_col(adjacency_matrix_, i))
    {
      roots.push_back(i);
    }
  }

  CHECK_EQ(1, roots.size(), "tree should have exactly one root!");

  return roots.front();
}

std::vector<uint> TreeBuilder::get_nodes() const
{
  std::vector<uint> nodes;

  for(auto i = 0; i < adjacency_matrix_.d0; ++i)
  {
    if(!empty_row(adjacency_matrix_, i) || !empty_col(adjacency_matrix_, i))
    {
      nodes.push_back(i);
    }
  }

  return nodes;
}

std::vector<uint> TreeBuilder::get_leaves() const
{
  std::vector<uint> leafs;

  for(auto i = 0; i < adjacency_matrix_.d0; ++i)
  {
    if(empty_row(adjacency_matrix_, i) && !empty_col(adjacency_matrix_, i))
    {
      leafs.push_back(i);
    }
  }

  return leafs;
}

std::vector<uint> TreeBuilder::get_parents(uint node) const
{
  std::vector<uint> parents;

  auto col = adjacency_matrix_.col(node);

  for(uint i = 0; i < col.d0; ++i)
  {
    if(col(i, 0) != 0)
    {
      parents.push_back(i);
    }
  }

  return parents;
}

std::vector<uint> TreeBuilder::get_children(uint node) const
{
  std::vector<uint> children;

  auto row = adjacency_matrix_.row(node);

  for(uint i = 0; i < row.d1; ++i)
  {
    if(row(0, i) != 0)
    {
      children.push_back(i);
    }
  }

  return children;
}

std::vector<uint> TreeBuilder::get_grand_children(uint node, uint step) const
{
  std::vector<uint> parents;
  std::vector<uint> children;

  parents.push_back(node);

  for(auto s = 1; s <= step; ++s)
  {
    children.clear();
    for(auto n: parents)
    {
      auto _children = get_children(n);
      children.insert(children.begin(), _children.begin(), _children.end());
    }

    if(s == step)
    {
      return children;
    }
    else
    {
      parents = children;
    }
  }

  return std::vector<uint>({node});
}

std::vector<uint> TreeBuilder::get_grand_children_with_backtracking(uint node, uint step) const
{
  std::vector<uint> grand_children;
  {
    auto _step = step;
    grand_children = get_grand_children(node, _step);
    while(grand_children.empty() && _step > 1)
    {
      _step--;
      grand_children = get_grand_children(node, _step);
    }
  }
  return grand_children;
}

std::vector<uint> TreeBuilder::get_leaves_from(uint node) const
{
  std::vector<uint> leaves;
  std::list<uint> queue;
  queue.push_back(node);

  while(!queue.empty())
  {
    auto p = queue.back();
    queue.pop_back();

    auto children = get_children(p);
    if(children.size() == 0)
    {
      leaves.push_back(p);
    }
    else
    {
      for(const auto& q: children)
      {
        queue.push_back(q);
      }
    }
  }
  return leaves;
}

std::vector<uint> TreeBuilder::get_path(uint from, uint to) const
{
  std::vector<uint> path;

  uint current = to;

  while(current!=from)
  {
    path.push_back(current);

    auto parents = get_parents(current);

    CHECK(parents.size() <= 1, "graph not yet supported");
    //CHECK(parents.size() > 0, "path doesn't exist");

    if(parents.empty())
      return std::vector<uint>();

    current = parents.front();
  }

  path.push_back(from);

  std::reverse(path.begin(), path.end());

  return path;
}

_Branch TreeBuilder::_get_branch(uint leaf) const
{
  _Branch branch;
  auto current = leaf;
  auto parents = get_parents(current);

  CHECK(parents.size() > 0, "No parents for this leaf!");
  CHECK(parents.size() < 2, "Not implemented yet!, needs graph support!");

  branch.p = p(parents[0], leaf);
  branch.leaf_id = leaf;

  while(parents.size())
  {
    auto parent = parents[0];
    auto p = this->p(parent, current);
    branch.local_to_global.push_back(current);
    current = parent;
    parents = get_parents(current);
  }

  branch.local_to_global.push_back(get_root());
  std::reverse(branch.local_to_global.begin(), branch.local_to_global.end());

  branch.global_to_local = std::vector< int >(adjacency_matrix_.d0, -1);
  for( auto local = 0; local < branch.local_to_global.size(); ++local )
  {
    auto global = branch.local_to_global[local];
    branch.global_to_local[global] = local;
  }

  return branch;
}

TreeBuilder TreeBuilder::get_subtree_from(uint node) const
{
  TreeBuilder tree(p(0, node), 0);
  std::list<uint> queue;
  queue.push_back(node);

  while(!queue.empty())
  {
    auto p = queue.back();
    queue.pop_back();

    auto children = get_children(p);

    for(const auto& q: children)
    {
      tree.add_edge(p, q);
      queue.push_back(q);
    }
  }
  return tree;
}

TreeBuilder TreeBuilder::compressed(Mapping & mapping) const
{
  mapping.orig_to_compressed = intA(adjacency_matrix_.d0);
  for(auto i = 0; i < adjacency_matrix_.d0; ++i)
  {
    if(!empty_col(adjacency_matrix_, i) || !empty_row(adjacency_matrix_, i))
    {
      mapping.compressed_to_orig.append(i);
      mapping.orig_to_compressed(i) = mapping.compressed_to_orig.d0 - 1;
    }
    else
    {
      mapping.orig_to_compressed(i) = -1;
    }
  }

  arr adj = zeros(mapping.compressed_to_orig.d0, mapping.compressed_to_orig.d0);

  for(auto i = 0; i < adj.d0; ++i)
  {
    auto I = mapping.compressed_to_orig(i);
    for(auto j = 0; j < adj.d0; ++j)
    {
      auto J = mapping.compressed_to_orig(j);
      adj(i, j) = adjacency_matrix_(I, J);
    }
  }

  TreeBuilder compressed(p_, d_);
  compressed.adjacency_matrix_ = adj;

  return compressed;
}

std::vector<_Branch> TreeBuilder::get_branches() const
{
  std::vector<_Branch> branches;

  for(auto l : get_leaves())
  {
    branches.push_back(_get_branch(l));
  }

  return branches;
}

TreeBuilder TreeBuilder::get_branch(uint leaf) const
{
  TreeBuilder branch(p(0, leaf), 0);

  auto current = leaf;
  auto parents = get_parents(current);

  CHECK(parents.size() > 0, "No parents for this leaf!");
  CHECK(parents.size() < 2, "Not implemented yet!, needs graph support!");

  while(parents.size())
  {
    auto parent = parents[0];
    //auto p = this->p(parent, current);

    branch.add_edge(parent, current, 1.0); // could be p here eventually, but here we assume here that we are on the common part "knowing that we will branch"

    current = parent;
    parents = get_parents(current);
  }

  return branch;
}

intA TreeBuilder::get_vars0(const TimeInterval& interval, const _Branch& branch, uint steps) const
{
  auto from = interval.from;
  auto to = interval.to;

  const auto duration = to - from; //ceil(to - from);
  uint d0 = duration > 0 ? ceil(duration * steps) : 0;
  int from_step = from * steps;

  intA vars(d0);

  for(auto t=0; t < d0; ++t)
  {
    CHECK(t < vars.d0, "bug");

    int k = from_step + t;

    if(k < 0) // prefix handling (we don't branch during the prefix)
    {
      int from_node = branch.local_to_global[0];
      int to_node   = branch.local_to_global[1];
      vars(t) = from_node * steps + k;
    }
    else
    {
      auto i = floor(k / double(steps));
      auto j = ceil(k / double(steps) + 0.00001);

      CHECK(i >= 0 && i < branch.local_to_global.size(), "bug");
      CHECK(j >= 0 && j < branch.local_to_global.size(), "bug");

      int from_node = branch.local_to_global[i];
      int to_node   = branch.local_to_global[j];
      int r = k % steps;

      vars(t) = (steps > 1 ? to_node - 1 : from_node) * steps + r; // in new branched phase
    }
  }

  return vars;
}

intA TreeBuilder::get_vars(const TimeInterval& interval, uint leaf, uint order, uint steps) const
{
  auto branch = _get_branch(leaf);
  auto it = normalize(interval, branch, steps);
  auto from = it.from;
  auto to = it.to;

  // if frm and to are negative, return early
  if(from < 0 && to <= 0)
  {
    return intA();
  }

  std::vector<intA> splitted_vars(order+1);// = get_vars0(from, to, leaf, steps);
  for(auto j = 0; j < order+1; ++j)
  {
    auto delta = double(j) / steps;
    splitted_vars[j] = get_vars0({from - delta, to - delta}, branch, steps);
  }

  auto d0 = splitted_vars.front().d0;
  intA vars(d0, order + 1);
  for(auto i = 0; i < d0; ++i)
  {
    for(auto j = 0; j < order+1; ++j)
    {
      vars(i, order - j) = splitted_vars[j](i);
    }
  }

  return vars;
}

arr TreeBuilder::get_scales(const TimeInterval& interval, uint leaf, uint steps) const
{
  auto branch = _get_branch(leaf);

  auto it = normalize(interval, branch, steps);
  const auto& from = it.from;
  const auto& to = it.to;

  // if frm and to are negative, return early
  if(from < 0 && to <= 0)
  {
    return arr();
  }

  const auto duration = to - from;
  uint d0 = duration > 0 ? ceil(duration * steps) : 0;
  uint from_step = from * steps;

  arr full_scale = arr((branch.local_to_global.size() - 1) * steps);

  double p = 1.0;
  for(auto i = 0; i < branch.local_to_global.size() - 1; ++i)
  {
    auto global_i = branch.local_to_global[i];
    auto global_j = branch.local_to_global[i+1];
    p *= adjacency_matrix_(global_i, global_j);

    for(auto s = 0; s < steps; ++s)
    {
      full_scale(steps * i + s) = p;
    }
  }

  arr scale(d0);
  for(auto i = 0; i < d0; ++i)
  {
    scale(i) = full_scale(i + from_step);
  }

  return scale;
}

TaskSpec TreeBuilder::get_spec(uint order, uint steps) const
{
  // get leaves fron start_edge
  std::vector<uint> leaves = get_leaves();
  std::sort(leaves.begin(), leaves.end()); // unnecessary but easier to debug

  return get_spec({0, -1.0}, leaves, order, steps);
}

TaskSpec TreeBuilder::get_spec(const TimeInterval& interval, const Edge& start_edge, uint order, uint steps) const
{
  // get leaves fron start_edge
  std::vector<uint> leaves = get_leaves_from(start_edge.to);
  std::sort(leaves.begin(), leaves.end()); // unnecessary but easier to debug

  return get_spec(interval, leaves, order, steps);
}

TaskSpec TreeBuilder::get_spec(const TimeInterval& interval, const std::vector<uint> & leaves, uint order, uint steps) const
{
  // get vars for each leaf
  std::vector<std::vector<intA>> slitted_varss(leaves.size());
  std::vector<arr> scaless(leaves.size());
  for(auto i = 0; i < leaves.size(); ++i)
  {
    auto vars = get_vars(interval, leaves[i], order, steps);
    auto scales = get_scales(interval, leaves[i], steps);
    std::vector<intA> splitted_vars = std::vector<intA>(vars.size() / (order+1));
    for(auto s = 0; s < vars.size() / (order+1); ++s)
    {
      auto steps = intA(order+1, 1);
      for(auto j = 0; j < order + 1; ++j)
      {
        steps(j, 0) = vars(s, j);
      }
      splitted_vars[s] = steps;
    }
    slitted_varss[i] = splitted_vars;
    scaless[i] = scales;
  }

  // remove doubles
  std::vector<intA> splitted_no_doubles_vars;
  arr no_doubles_scales;
  for(auto i = 0; i < slitted_varss.size(); ++i)
  {
    for(auto s = 0; s < slitted_varss[i].size(); ++s)
    {
      const auto & steps = slitted_varss[i][s];
      double scale = scaless[i](s);

      if(std::find(splitted_no_doubles_vars.begin(), splitted_no_doubles_vars.end(), steps) == splitted_no_doubles_vars.end())
      {
        splitted_no_doubles_vars.push_back(steps);
        no_doubles_scales.append(scale);
      }
    }
  }

  // flatten
  intA vars(splitted_no_doubles_vars.size(), (order + 1));

  for(auto s = 0; s < splitted_no_doubles_vars.size(); ++s)
  {
    for(auto j = 0; j < order + 1; ++j)
    {
      vars(s, j) = splitted_no_doubles_vars[s](j, 0);
    }
  }

  CHECK(vars.d0 == no_doubles_scales.d0, "size corruption");

  return TaskSpec{std::move(vars), std::move(no_doubles_scales)};
}

int TreeBuilder::get_step(double time, const Edge& edge, uint steps) const
{
  //int step = time * microSteps - 1;
  //return order0(step, 0);

  auto spec = get_spec({time, time}, edge, 0, steps);

  if(spec.vars.d0 == 0) // out of the time interval
    return -1;

  CHECK(spec.vars.d0 == 1, "wrong spec!");

  return spec.vars.front();
}

void TreeBuilder::add_edge(uint from, uint to, double p)
{
  uint max = std::max(from, to);
  auto size = max+1;

  if(adjacency_matrix_.d0 < size)
  {
    auto old_adjacency_matrix = adjacency_matrix_;
    const auto& old_size = old_adjacency_matrix.d0;
    auto adjacency_matrix = arr(size, size);
    adjacency_matrix.setMatrixBlock(old_adjacency_matrix, 0, 0);
    adjacency_matrix_ = adjacency_matrix;
  }

  adjacency_matrix_(from, to) = p;
}

std::ostream& operator<<(std::ostream& os, const TreeBuilder & tree)
{
  std::list<uint> queue;
  queue.push_back(0);

  while(!queue.empty())
  {
    auto p = queue.back();
    queue.pop_back();

    auto children = tree.get_children(p);
    for(const auto& q: children)
    {
      std::cout << p << "->" << q << std::endl;

      queue.push_back(q);
    }

    if(children.empty() && p < tree.adjacency_matrix().d0 - 1)
      queue.push_back(p+1);
  }

  return os;
}

bool operator==(const Edge& a, const Edge& b)
{
  return a.from == b.from && a.to == b.to;
}

bool operator==(const TreeBuilder& a, const TreeBuilder& b)
{
  return a.adjacency_matrix() == b.adjacency_matrix() && a.p() == b.p() && a.d() == b.d();
}
}
