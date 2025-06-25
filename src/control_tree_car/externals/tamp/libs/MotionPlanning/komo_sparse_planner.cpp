#include <komo_sparse_planner.h>
#include <tree_builder.h>
#include <subtree_generators.h>
#include <komo_wrapper.h>
#include <trajectory_tree_visualizer.h>

#include <Optimization/utils.h>
#include <Optimization/decentralized_optimizer.h>

#include <Kin/kin.h>
#include <Kin/switch.h>
#include <Kin/TM_transition.h>
#include <Kin/TM_FlagConstraints.h>
#include <Kin/TM_FixSwitchedObjects.h>

#include <unordered_set>

using T = ConstrainedProblem;
using U = AverageUpdater;

namespace mp
{
/// COMMON: KOMOSparsePlanner
TreeBuilder KOMOSparsePlanner::buildTree( Policy & policy ) const
{
  TreeBuilder treeBuilder(1.0, 0);

  std::list< Policy::GraphNodeTypePtr > fifo;
  fifo.push_back( policy.root() );

  while( ! fifo.empty() )
  {
    auto b = fifo.back();
    fifo.pop_back();

    const auto& a = b->parent();

    if(a)
    {
      const auto& p = b->data().p;
      treeBuilder.add_edge(a->id(), b->id(), p);
    }

    for(const auto&c : b->children())
    {
      fifo.push_back(c);
    }
  }

  return treeBuilder;
}

std::shared_ptr< ExtensibleKOMO > KOMOSparsePlanner::intializeKOMO( const TreeBuilder & tree, const std::shared_ptr< const rai::KinematicWorld > & startKinematic ) const
{
  auto komo = komoFactory_.createKomo();
  komo->setModel(*startKinematic);
  komo->sparseOptimization = true;

  CHECK(tree.d() == 0, "support for free prefix komo deactivated!");
  //komo->freePrefix = !(tree.d() == 0); // free prefix is used when decomposing the trajectory in several pieces and optimizing them independantly, this is NOT efficient

  const auto nPhases = tree.n_nodes() - 1;
  komo->setTiming(nPhases, config_.microSteps_, config_.secPerPhase_, 2);

  return komo;
}

std::vector<Vars> KOMOSparsePlanner::getSubProblems( const TreeBuilder & tree, Policy & policy ) const
{
  std::vector<Vars> allVars;
  allVars.reserve(tree.get_leaves().size());

  for(const auto& l: policy.sleaves())
  {
    auto vars0 = tree.get_vars({0, 1.0 * l->depth()}, l->id(), 0, config_.microSteps_);
    auto vars1 = tree.get_vars({0, 1.0 * l->depth()}, l->id(), 1, config_.microSteps_);
    auto vars2 = tree.get_vars({0, 1.0 * l->depth()}, l->id(), 2, config_.microSteps_);
    Vars branch{vars0, vars1, vars2, config_.microSteps_};
    allVars.push_back(branch);
  }

  return allVars;
}

std::vector<intA> KOMOSparsePlanner::getSubProblemMasks( const std::vector<Vars> & allVars, uint T ) const
{
  std::vector<intA> masks(allVars.size());
  for(auto w = 0; w < allVars.size(); ++w)
  {
    auto & mask = masks[w];
    auto & vars = allVars[w][0];

    mask = intA(T);

    for(auto i: vars)
    {
      mask(i) = 1;
    }
  }

  return masks;
}

void KOMOSparsePlanner::groundPolicyActionsJoint( const TreeBuilder & tree,
                               Policy & policy,
                               const std::shared_ptr< ExtensibleKOMO > & komo ) const
{
  // traverse tree and ground symbols
  std::unordered_set<uint> visited;

  komo->groundInit(tree);

  for(const auto& l: policy.sleaves())
  {
    auto q = l;
    auto p = q->parent();

    while(p)
    {
      if(visited.find(q->id()) == visited.end())
      {
        double start = p->depth();
        double end = q->depth();

        Interval interval;
        interval.time = {start, start + 1.0};
        interval.edge = {p->id(), q->id()};

        // square acc
        W(komo.get()).addObjective(interval, tree, new TM_Transition(komo->world), OT_sos, NoArr, 1.0, 2);

        // ground other tasks
        komo->groundTasks(interval, tree, q->data().leadingKomoArgs, 1);

        visited.insert(q->id());
      }
      q = p;
      p = q->parent();
    }
  }
}

void KOMOSparsePlanner::watch( const std::shared_ptr< ExtensibleKOMO > & komo ) const
{
  //komo->displayTrajectory(0.1, true, false);
  Var<WorldL> configs;
  auto v = std::make_shared<KinPathViewer>(configs,  0.2, -0 );
  v->setConfigurations(komo->configurations);
  rai::wait();
}

/// JOINT
void JointPlanner::optimize( Policy & policy, const rai::Array< std::shared_ptr< const rai::KinematicWorld > > & startKinematics ) const
{
  using W = KomoWrapper;

  // build tree
  auto tree = buildTree(policy);

  // prepare komo
  auto komo = intializeKOMO(tree, startKinematics.front());

  // ground policy actions
  komo->groundInit(tree);
  auto allVars = getSubProblems(tree, policy);
  groundPolicyActionsJoint(tree, policy, komo);

  // run optimization
  komo->verbose = 1;
  W(komo.get()).reset(allVars);

  auto start = std::chrono::high_resolution_clock::now();

  komo->run();

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double optimizationTime=std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.0;

  // LOGS
  auto log = true;
//  if(log) {
//    cout <<"** Hessian size.[" << komo->opt->newton.Hx.d0 << "] sparsity=" << sparsity(komo->opt->newton.Hx);
//    cout <<endl;
//  }

  if(log) {
    cout <<"** optimization time=" << optimizationTime
         <<" setJointStateCount=" << rai::KinematicWorld::setJointStateCount <<endl;
    cout <<" (kin:" << komo->timeKinematics <<" coll:" << komo->timeCollisions << " feat:" << komo->timeFeatures <<" newton: " << komo->timeNewton <<")" << std::endl;
  }

  //
  //komo->getReport(true);
  //for(auto c: komo->configurations) std::cout << c->q.N << std::endl;

  watch(komo);
}

/// ADMM SPARSE
void ADMMSParsePlanner::optimize( Policy & policy, const rai::Array< std::shared_ptr< const rai::KinematicWorld > > & startKinematics ) const
{
  // build tree
  auto tree = buildTree(policy);

  // prepare komos
  std::vector< std::shared_ptr< ExtensibleKOMO > > komos;
  for(auto w = 0; w < policy.leaves().size(); ++w)
  {
    auto komo = intializeKOMO(tree, startKinematics.front());
    komos.push_back(komo);
  }

  // ground each komo
  for(uint w = 0; w < policy.leaves().size(); ++w)
  {
    groundPolicyActionsJoint(tree, policy, komos[w]);
  }

  // reset each komo
  auto allVars = getSubProblems(tree, policy);   // get subproblems
  auto allTMasks = getSubProblemMasks(allVars, komos.front()->T);

  for(auto & komo: komos)
  {
    W(komo.get()).reset(allVars, 0);
  }

  // ADMM
  std::vector<std::shared_ptr<GraphProblem>> converters;
  std::vector<std::shared_ptr<ConstrainedProblem>> constrained_problems;
  std::vector<arr> xmasks;
  converters.reserve(policy.leaves().size());
  constrained_problems.reserve(policy.leaves().size());
  xmasks.reserve(policy.leaves().size());

  for(auto w = 0; w < policy.leaves().size(); ++w)
  {
    auto& komo = *komos[w];
    auto& tmask = allTMasks[w];

    auto gp = std::make_shared<ADMM_MotionProblem_GraphProblem>(komo);
    gp->setSubProblem(tmask);
    arr xmask;
    gp->getXMask(xmask);//, false);

    auto pb = std::make_shared<Conv_Graph_ConstrainedProblem>(*gp, komo.logFile);

    converters.emplace_back(gp);
    constrained_problems.push_back(pb);
    xmasks.push_back(xmask);
  }

  // RUN
  auto start = std::chrono::high_resolution_clock::now();

  auto x = komos.front()->x;

  DecOptConstrained<T, U> opt(x, constrained_problems, xmasks, U(), DecOptConfig(PARALLEL, false));
  opt.run();

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double optimizationTime=std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.0;


  // LOGS
  if(true) {
    cout <<"** optimization time=" << optimizationTime
         <<" setJointStateCount=" << rai::KinematicWorld::setJointStateCount <<endl;
  }

  auto & komo = komos.back();
  komo->set_x(x);
  watch(komo);
}

/// ADMM COMPRESSED
void ADMMCompressedPlanner::setDecompositionStrategy( const std::string& strategy, const std::string& nJobs )
{
  decompositionStrategy_ = strategy;

  try
  {
    nJobs_ = std::stoi(nJobs);
  }
  catch (const std::invalid_argument& ia)
  {
    CHECK(false, "wrong argument for the number of jobs");
  }
}

void ADMMCompressedPlanner::groundPolicyActionsCompressed( const TreeBuilder & fullTree,
                                                           const TreeBuilder & uncompressed,
                                                           const TreeBuilder & compressed,
                                                           const Mapping & mapping,
                                                           Policy & policy,
                                                           const std::shared_ptr< ExtensibleKOMO > & komo ) const
{
  komo->groundInit(compressed);

  auto edges = interactingEdges(fullTree, uncompressed); // interacting edges and the closest edge on subtree
  // interating edges are all the edges on which groundings have an influence on the subtree

  // traverse tree and ground symbols
  std::unordered_set<uint> visited;
  for(const auto& l: policy.sleaves())
  {
    auto q = l;
    auto p = q->parent();

    while(p)
    {
      // check if p-q is part of the interacting edges
      auto it = std::find_if(edges.begin(), edges.end(), [&](const std::pair<Edge, Edge> & es)
      {
        return es.first == Edge({p->id(), q->id()});
      });
      if(visited.find(q->id()) == visited.end()
         && it != edges.end())
      {
        double start = p->depth();
        double start_offset = uncompressed.d();
        double end = q->depth();

        const auto& local_edge = it->second;

        auto pid = mapping.orig_to_compressed(local_edge.from);
        auto qid = mapping.orig_to_compressed(local_edge.to);

        Interval interval;
        interval.time = {start - start_offset, start - start_offset + 1.0};
        interval.edge = {pid, qid};

        // square acc
        W(komo.get()).addObjective(interval, compressed, new TM_Transition(komo->world), OT_sos, NoArr, 1.0, 2);

        // ground other tasks
        komo->groundTasks(interval, compressed, q->data().leadingKomoArgs, 1);

        visited.insert(q->id());
      }
      q = p;
      p = q->parent();
    }
  }
}

void ADMMCompressedPlanner::optimize( Policy & policy, const rai::Array< std::shared_ptr< const rai::KinematicWorld > > & startKinematics ) const
{
  using W = KomoWrapper;

  // build tree
  auto tree = buildTree(policy);

  // 0 - SPLIT INTO SUBTREES
  auto gen = generatorFactory_.create(decompositionStrategy_, nJobs_, tree);
  auto subproblems = get_subproblems(gen); // uncompressed pb, compressed pb, mapping
  auto allVars = get_all_vars(subproblems, config_.microSteps_);   // get subproblem vars (local)

  // 1 - PREPARE KOMOS
  auto witness = intializeKOMO(tree, startKinematics.front());
  groundPolicyActionsJoint(tree, policy, witness);

  // 1.1 - init and ground each komo
  std::vector< std::shared_ptr< ExtensibleKOMO > > komos;
  for(auto w = 0; w < subproblems.size(); ++w)
  {
    std::cout << "GROUND " << w << std::endl;

    const auto& sub = subproblems[w];
    const auto& uncompressed = std::get<0>(sub);
    const auto& compressed = std::get<1>(sub);
    const auto& mapping = std::get<2>(sub);

    auto komo = intializeKOMO(compressed, startKinematics(0));
    komos.push_back(komo);
    groundPolicyActionsCompressed(tree, uncompressed, compressed, mapping, policy, komos[w]);
  }

  // 1.2 - reset komos
  const auto& witnessVars = std::get<2>(allVars); // fused
  W(witness.get()).reset(std::get<0>(allVars), 0); // uncompressed -> used fused!
  for(auto w = 0; w < komos.size(); ++w)
  {
    const std::vector<Vars>& uncompressed{std::get<0>(allVars)[w]};
    const std::vector<Vars>& compressed{std::get<1>(allVars)[w]};

    const auto firstIndex = uncompressed.front().order0.front();

    if(firstIndex>0)
    {
      const auto prev = witnessVars.getPreviousStep(firstIndex);
      komos[w]->world.copy(*witness->configurations(prev + witness->k_order));
    }
    //else // if first index == 0, the start configuration (world) is already well configured from the komo init

    W(komos[w].get()).reset(compressed);
  }

  // 2 - CREATE SUB-OPTIMIZATION-PROBLEMS
  auto uncompressedTMasks = getSubProblemMasks(std::get<0>(allVars), witness->T); // use uncompressed

  // 2.1 - create xmasks
  std::vector<arr> xmasks;
  xmasks.reserve(policy.leaves().size());
  auto gp = std::make_shared<ADMM_MotionProblem_GraphProblem>(*witness);
  for(auto w = 0; w < komos.size(); ++w)
  {
    auto& tmask = uncompressedTMasks[w];
    gp->setSubProblem(tmask);
    arr xmask;
    gp->getXMask(xmask);//, komos[w]->freePrefix);
    xmasks.push_back(xmask);
  }

  // 2.2 - create sub-opt-problems
  std::vector<std::shared_ptr<GraphProblem>> converters;
  std::vector<std::shared_ptr<ConstrainedProblem>> constrained_problems;
  converters.reserve(policy.leaves().size());
  constrained_problems.reserve(policy.leaves().size());
  for(auto w = 0; w < komos.size(); ++w)
  {
    auto& komo = *komos[w];
    auto gp = std::make_shared<KOMO::Conv_MotionProblem_GraphProblem>(komo);
    auto pb = std::make_shared<Conv_Graph_ConstrainedProblem>(*gp, komo.logFile);

    converters.push_back(gp);
    constrained_problems.push_back(pb);
  }

  // 3 - RUN
  auto start = std::chrono::high_resolution_clock::now();

  auto x = witness->x;

  OptOptions options;
  options.verbose = 1;
//  options.stopTolerance = 0; // avoid ADMM to return too early if it seems stuck
//  options.stopEvals = 20000; // avoid ADMM and optconstraintd solver to stop too early
//  options.stopIters = 20000; // ..
//  options.stopOuters = 20000; // ..

  DecOptConfig decOptConfig(PARALLEL, true, options, false);
  decOptConfig.scheduling = PARALLEL;
  decOptConfig.compressed = true;
  decOptConfig.checkGradients = false;

  DecOptConstrained<T, U> opt(x, constrained_problems, xmasks, U(), decOptConfig);
  opt.run();

  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double optimizationTime=std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.0;

  //4 - LOGS + PRINTS
  if(true) {
    cout <<"** optimization time=" << optimizationTime
         <<" setJointStateCount=" << rai::KinematicWorld::setJointStateCount <<endl;

    for(auto w = 0; w < opt.newtons.size(); ++w)
    {
      const auto& komo = *komos[w];
      auto timeNewton = opt.newtons[w]->timeNewton;
      std::cout <<" (kin:" << komo.timeKinematics <<" coll:" << komo.timeCollisions << " feat:" << komo.timeFeatures <<" newton: " << timeNewton <<")" << std::endl;
    }
  }
  //
  witness->set_x(x);
  witness->x = x;
  watch(witness);

  //watch(komos.back());
  //witness->plotVelocity();
  //rai::wait( 30, true );
}

// for checking the dimension consistency
//auto nz = [](const arr& x)
//{
//  uint n = 0;
//  for(auto i = 0; i < x.d0; ++i)
//  {
//    if(x(i)!=0.0)
//      n++;
//  }
//  return n;
//};

//uint nx = 0;
//uint nkx = 0;
//for(auto w = 0; w < komos.size(); ++w)
//{
//  const auto& x = xmasks[w];
//  const auto& kx = komos[w]->x;

//  const auto n = nz(x);
//  //CHECK_EQ(n, kx.d0, "corrupted dimensions");
//  //if(n != kx.d0)
//  std::cout << n << " vs. " << kx.d0 << std::endl;

//  nkx += kx.d0;
//  nx += n;
//}

//std::cout << nkx << " VS. " << nx << " VS. " << witness->x.d0 << std::endl;

// debug
//{
//  auto gen2 = generatorFactory_.create("BranchGen", nJobs_, tree);
//  auto subproblems2 = get_subproblems(gen2); // uncompressed pb, compressed pb, mapping
//  auto allVars2 = get_all_vars(subproblems2, config_.microSteps_);   // get subproblem vars (local)

//  for(auto w = 0; w < subproblems.size(); ++w)
//  {
//    const auto& sub = subproblems[w];
//    const auto& uncompressed = std::get<0>(sub);
//    const auto& compressed = std::get<1>(sub);
//    const auto& mapping = std::get<2>(sub);

//    const auto& sub2 = subproblems2[w];
//    const auto& uncompressed2 = std::get<0>(sub2);
//    const auto& compressed2 = std::get<1>(sub2);
//    const auto& mapping2 = std::get<2>(sub2);

//    std::cout << uncompressed.adjacency_matrix() << std::endl << std::endl;
//    std::cout << uncompressed2.adjacency_matrix() << std::endl << std::endl;


//    std::cout << compressed.adjacency_matrix() << std::endl << std::endl;
//    std::cout << compressed2.adjacency_matrix() << std::endl << std::endl;

//    //CHECK_EQ(uncompressed, uncompressed2, "");
//    //CHECK_EQ(compressed, compressed2, "");

//  }
//}


}
