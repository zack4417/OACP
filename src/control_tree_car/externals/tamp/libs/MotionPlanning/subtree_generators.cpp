#include <subtree_generators.h>
#include <unordered_set>

namespace mp
{

std::vector<std::pair<Edge, Edge>> interactingEdges(const TreeBuilder& tree, const TreeBuilder& subtree)
{
  std::vector<std::pair<Edge, Edge>> edges;

  auto subleaves = subtree.get_leaves();
  std::vector<std::pair<uint, uint>> leaves; // global, local

  for(auto l: subleaves)
  {
    auto extended = tree.get_leaves_from(l);
    for(const auto& e: extended)
    {
      leaves.push_back(std::pair<uint, uint>(e, l));
    }
    //leaves.insert(leaves.begin(), extended.begin(), extended.end());
  }

  std::list<std::pair<uint, uint>> lifo;
  std::unordered_set<uint> visited;
  for(const auto& el: leaves)
  {
    lifo.push_back(el);
  }

  while(!lifo.empty())
  {
    auto ql = lifo.back();
    lifo.pop_back();

    auto q = ql.first;
    auto l = ql.second;

    if(visited.find(q) != visited.end())
      continue;

    auto ps = tree.get_parents(q);

    for(const auto p: ps)
    {
      visited.insert(q);
      lifo.push_back(std::pair<uint, uint>(p, l));

      const auto e = Edge({p, q});
      if(subtree.has_node(p) && subtree.has_node(q)) // closest edge is on tree
        edges.push_back(std::pair<Edge, Edge>(e, e));
      else if(tree.get_path(l, q).size()) // after the local leaf
      {
        auto ps = subtree.get_parents(l);
        CHECK(ps.size() ==  1, "should have exactly one parent!");
        const auto f = Edge({ps.front(), l});
        edges.push_back(std::pair<Edge, Edge>(e, f));
      }
      else
      {
        auto cs = subtree.get_children(subtree.get_root());
        CHECK(cs.size() ==  1, "should have exactly one child!");
        const auto f = Edge({subtree.get_root(), cs.front()});
        edges.push_back(std::pair<Edge, Edge>(e, f));
      }
    }
  }

  std::reverse(edges.begin(), edges.end());

  return edges;
}

/// SUBTREES AFTER FIRST BRANCHING (QMDP)
SubTreesAfterFirstBranching::SubTreesAfterFirstBranching(const TreeBuilder& tree)
  : tree(tree)
{
  uint branching_node = 0;

  std::list<uint> queue;
  queue.push_back(0);

  while(!queue.empty())
  {
    auto p = queue.back();
    queue.pop_back();

    path_to_source.push_back(p);

    auto children = tree.get_children(p);

    if(children.size() > 1)
    {
      sources = children;
      break;
    }

    for(const auto& c: children)
    {
      queue.push_back(c);
    }
  }

  CHECK(sources.size() > 0, "No first branching!, check that the policy is not linear!");
}

bool SubTreesAfterFirstBranching::finished() const
{
  return (index == sources.size());
}

TreeBuilder SubTreesAfterFirstBranching::next()
{
  CHECK(!finished(), "finished generator");

  auto sub = tree.get_subtree_from(sources[index]);

  for(auto i = 1; i < path_to_source.size(); ++i)
  {
    sub.add_edge(path_to_source[i-1], path_to_source[i]);
  }

  sub.add_edge(path_to_source.back(), sources[index++]);

  return sub;
}

/// LINEAR SPLIT -> NOT EFFICIENT
LinearSplit::LinearSplit(const TreeBuilder& tree, uint n)
  : tree(tree)
  , n(n)
{
  std::list<uint> queue;
  queue.push_back(0);

  const auto N = tree.n_nodes();
  uint step = uint(N / n + 0.5000001);
  if(!step) step = 1;

  while(!queue.empty())
  {
    auto p = queue.back();
    queue.pop_back();

    for(const auto& q: tree.get_grand_children_with_backtracking(p, step))
    {
      splits.push_back(std::pair<uint, uint>(p, q));
      queue.push_back(q);
    }
  }
}

bool LinearSplit::finished() const
{
  return (index == splits.size());
}

TreeBuilder LinearSplit::next()
{
  CHECK(!finished(), "invalid call to next");

  auto pq = splits[index++];
  auto from = pq.first;
  auto to = pq.second;
  auto p = tree.p(0, to);
  auto d = tree.get_path(0, from).size() - 1;

  TreeBuilder sub(p, d);
  auto path = tree.get_path(from, to);
  for(auto i = 1; i < path.size(); ++i)
  {
    sub.add_edge(path[i-1], path[i]);
  }
  return sub;
}

Vars fuse(const std::vector<Vars> & vars)
{
  // n log n
  Vars fused;
  fused.microSteps = vars.front().microSteps;
  fused.k_order = vars.front().k_order;

  std::unordered_set<uint> visited;

  for(auto i = 0; i < vars.size(); ++i)
  {
    for(auto j = 0; j < vars[i].order0.d0; ++j)
    {
      auto k = vars[i].order0(j, 0);

      if(std::find(visited.begin(), visited.end(), k) == visited.end())
      {
        for(auto order = 0; order <= vars.front().k_order; ++order)
        {
          intA var(1, order+1);
          for(auto l = 0; l <= order; ++l)
          {
            var(0, l) = vars[i][order](j, l);
          }

          fused.order(order).append(var);
        }
        visited.insert(k);
      }
    }
  }

  return fused;
}

std::vector< std::tuple< TreeBuilder, TreeBuilder, Mapping > > get_subproblems(const std::shared_ptr<SubTreeGen> & gen)
{
  std::vector< std::tuple< TreeBuilder, TreeBuilder, Mapping > > subproblems; // uncompressed pb, compressed pb, mapping

  while(!gen->finished())
  {
    Mapping mapping; // from global opt variable to local and vice versa
    auto uncompressed = gen->next(); // extract subtree (here a branch)
    auto compressed = uncompressed.compressed(mapping); // compress so that it has its own opt variable
    subproblems.push_back(std::tuple< TreeBuilder, TreeBuilder, Mapping >{uncompressed, compressed, mapping});

    std::cout << "uncompressed:\n" << uncompressed << std::endl;
  }

  return subproblems;
}

std::tuple< std::vector<Vars>, std::vector<Vars>, Vars> get_all_vars(const std::vector< std::tuple< TreeBuilder, TreeBuilder, Mapping > > & subproblems, uint steps)
{
  std::tuple< std::vector<Vars>, std::vector<Vars>, Vars > allVars; // uncompressed, compressed
  std::get<0>(allVars).reserve(subproblems.size());
  std::get<1>(allVars).reserve(subproblems.size());

  for(const auto & sub: subproblems)
  {
    {
    auto uncompressed = std::get<0>(sub);
    auto vars0 = uncompressed.get_spec(0, steps).vars;
    auto vars1 = uncompressed.get_spec(1, steps).vars;
    auto vars2 = uncompressed.get_spec(2, steps).vars;
    Vars var{vars0, vars1, vars2, steps};
    std::get<0>(allVars).push_back(var);
    }
    {
    auto compressed = std::get<1>(sub);
    auto vars0 = compressed.get_spec(0, steps).vars;
    auto vars1 = compressed.get_spec(1, steps).vars;
    auto vars2 = compressed.get_spec(2, steps).vars;
    Vars var{vars0, vars1, vars2, steps};
    std::get<1>(allVars).push_back(var);
    }
  }

  std::get<2>(allVars) = fuse(std::get<0>(allVars));

  return allVars;
}

}
