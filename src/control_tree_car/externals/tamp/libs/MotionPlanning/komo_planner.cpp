#include <komo_planner.h>

#include <kin_equality_task.h>
#include <trajectory_tree_visualizer.h>

#include <Kin/kin.h>
#include <Kin/switch.h>
#include <Kin/TM_transition.h>
#include <Kin/TM_FlagConstraints.h>
#include <Kin/TM_FixSwitchedObjects.h>

#include <belief_state.h>
#include <komo_planner_utils.h>
#include <komo_sparse_planner.h>
#include <Core/util.h>

#include <thread>
#include <future>
#include <list>
#include <chrono>


namespace mp
{
static constexpr double eps = std::numeric_limits< double >::epsilon();

//--------Motion Planner--------------//

void KOMOPlanner::setKin( const std::string & kinDescription )
{
  Graph G = loadKin(kinDescription);

  if( G[ config_.beliefStateTag_ ] == nullptr )
  {
    auto kin = std::make_shared< rai::KinematicWorld >();
    kin->init( kinDescription.c_str() );
    computeMeshNormals( kin->frames );
    kin->calc_fwdPropagateFrames();
    //kin->watch(/*true*/);

    startKinematics_.append( kin );
  }
  else
  {
    const auto& bsGraph = &G.get<Graph>(config_.beliefStateTag_ );
    const uint nWorlds = bsGraph->d0;

    // build the different worlds
    for( uint w = 0; w < nWorlds; w++ )
    {
      Graph kinG = loadKin(kinDescription);

      // copy unobservable facts
      auto n = bsGraph->elem(w);

      for( const auto& nn : n->graph() )
      {
        nn->newClone( kinG );
      }

      const auto& bsNode = kinG.getNode( config_.beliefStateTag_ );
      kinG.removeValue(bsNode);

      auto kin = createKin(kinG);
      computeMeshNormals( kin->frames );
      kin->calc_fwdPropagateFrames();
      //
      //if( w == 5 )
      //kin->watch( true );
      //
      startKinematics_.append( kin );
    }
  }

  computeQMask();
}

std::vector< double > KOMOPlanner::drawRandomVector( const std::vector< double > & override )
{
  if( startKinematics_.size() == 0 )
  {
    return std::vector< double >();
  }

  if( override.size() > 0 )
  {
    randomVec_ = override;
    return randomVec_;
  }

  const auto& world = startKinematics_(0);

  // get size of random Vector
  uint randomVecSize = 0;
  for( const auto& f : world->frames )
  {
    if( f->ats["random_bounds"]  )
    {
      const auto& randomBounds = f->ats.get<arr>("random_bounds");

      for( const auto& b : randomBounds )
      {
        if( b > 0 )
        {
          randomVecSize++;
        }
      }
    }
  }

  // draw it
  randomVec_ = std::vector< double >( randomVecSize );

  for( auto i = 0; i < randomVecSize; ++i )
  {
    auto v = rnd.uni(-1.0, 1.0);
    randomVec_[i] = v;
  }

  return randomVec_;
}

void KOMOPlanner::solveAndInform( const MotionPlanningParameters & po, Policy & policy )
{
  CHECK( startKinematics_.d0 == policy.N(), "consitency problem, the belief state size of the policy differs from the belief state size of the kinematics" );
  CHECK( po.policyId() == policy.id(), "id of the policy and the planning orders are not consistent" );

  //po.getParam( "type" );
  clearLastNonMarkovianResults();

  // solve on pose level
  optimizePoses( policy );

  /// EARLY STOPPING, detect if pose level not possible
  bool poseOptimizationFailed = false;
  // if a node has a constraint which is not satisfied, we set the node to infeasible i.e. infinite cost!
  savePoseOptimizationResults(policy, poseOptimizationFailed);

  if( poseOptimizationFailed )  // early stopping
    return;

  /// PATH OPTI
  if( po.getParam( "type" ) == "markovJointPath" )
  {
    optimizeMarkovianPath( policy );
    saveMarkovianPathOptimizationResults( policy );
  }
  else if( po.getParam( "type" ) == "jointPath" )
  {
    // solve on path level
    optimizePath( policy );

    // solve on joint path level
    if( policy.N() > 1 )
    {
      optimizeJointPath( policy );
    }

    saveJointPathOptimizationResults( policy );
  }
  else if( po.getParam( "type" ) == "jointSparse" )
  {
    JointPlanner planner(config_, komoFactory_);
    planner.optimize(policy, startKinematics_);
    //optimizeJointSparse( policy );
  }
  else if( po.getParam( "type" ) == "ADMMSparse" )
  {
    ADMMSParsePlanner planner(config_, komoFactory_);
    planner.optimize(policy, startKinematics_);
    //optimizeADMMSparse( policy );
  }
  else if( po.getParam( "type" ) == "ADMMCompressed" )
  {
    ADMMCompressedPlanner planner(config_, komoFactory_);
    planner.setDecompositionStrategy(po.getParam("decompositionStrategy"), po.getParam("nJobs"));
    planner.optimize(policy, startKinematics_);
  }
  else
  {
    CHECK( false, "not implemented yet!" );
  }
  //CHECK( checkPolicyIntegrity( policy ), "Policy is corrupted" );
}

void KOMOPlanner::display( const Policy & policy, double sec )
{
  Policy tmp( policy );
  MotionPlanningParameters po( policy.id() );

  po.setParam( "type", "jointPath" );
  // resolve since this planner doesn't store paths
  //
  auto start = std::chrono::high_resolution_clock::now();
  //

  solveAndInform( po, tmp );

  //
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout << "motion planning time (ms):" << ms << std::endl;
  //

  // retrieve trajectories
  rai::Array< rai::Array< rai::Array< rai::KinematicWorld > > > frames;

  const auto & kinFrames = jointPathKinFrames_.size() > 1 ? jointPathKinFrames_ : pathKinFrames_;
  for( const auto& leafWorldKinFramesPair : kinFrames )
  {
    frames.append( leafWorldKinFramesPair.second );
  }

  // display
  if( sec > 0 )
  {
    TrajectoryTreeVisualizer viz( frames, "policy", config_.microSteps_ * config_.secPerPhase_ );

    rai::wait( sec, true );
  }
}
std::pair< double, double > KOMOPlanner::evaluateLastSolution()
{
  // retrieve trajectories
  rai::Array< rai::Array< rai::Array< rai::KinematicWorld > > > frames;

  //CHECK( jointPathKinFrames_.size() == 0, "not supported yet if branching!" );

  const auto & kinFrames = jointPathKinFrames_.size() > 1 ? jointPathKinFrames_ : pathKinFrames_;
  for( const auto& leafWorldKinFramesPair : kinFrames )
  {
    frames.append( leafWorldKinFramesPair.second );
  }
  // evaluation
  for( auto k = 0; k < frames.N; ++k )
  {
    for( auto l = 0; l < frames.at(k).N; ++l )
    {
      auto eval = evaluate( frames.at(k).at(l), config_.secPerPhase_ / config_.microSteps_ );

      auto length = eval.first;
      auto acc_cost = eval.second;

      return std::make_pair( length, acc_cost );
    }
  }
}

void KOMOPlanner::registerInit( const TreeInitGrounder & grounder )
{
  komoFactory_.registerInit( grounder );
}

void KOMOPlanner::registerTask( const std::string & type, const TreeSymbolGrounder & grounder )
{
  komoFactory_.registerTask( type, grounder );
}

void KOMOPlanner::computeQMask()
{
  qmask_ = extractAgentQMask( *startKinematics_( 0 ) );

  // sanity check
  for( uint w = 1; w < startKinematics_.size(); ++w )
  {
    CHECK( qmask_ == extractAgentQMask( *startKinematics_( 1 ) ), "corruption in agent joint definition" );
  }
}

///MARKOVIAN

void KOMOPlanner::optimizePoses( Policy & policy )
{
  std::cout << "optimizing poses.." << std::endl;

  optimizePosesFrom( policy.root() );
}

void KOMOPlanner::optimizePosesFrom( const Policy::GraphNodeTypePtr & node )
{
  //std::cout << "optimizing pose for:" << node->id() << std::endl;

  bool feasible = true;

  const auto N = node->data().beliefState.size();
  //
  effKinematics_  [ node->data().decisionGraphNodeId ] = rai::Array< rai::KinematicWorld >( N );
  poseCosts_      [ node->data().decisionGraphNodeId ] = arr( N );
  poseConstraints_[ node->data().decisionGraphNodeId ] = arr( N );
  //
  for( auto w = 0; w < N; ++w )
  {
    if( node->data().beliefState[ w ] > eps )
    {
      rai::KinematicWorld kin = node->isRoot() ? *( startKinematics_( w ) ) : ( effKinematics_.find( node->parent()->data().decisionGraphNodeId )->second( w ) );

      // create komo
      auto komo = komoFactory_.createKomo();

      // set-up komo
      komo->setModel( kin, true/*, false, true, false, false */);

      komo->setTiming( 1., 2, 5., 1/*, true*/ );
      //      komo->setHoming( -1., -1., 1e-1 ); //gradient bug??
      //komo->setSquaredQVelocities();
      komo->setSquaredQVelocities();
      //komo->setFixSwitchedObjects( -1., -1., 1e3 );

      komo->groundInit();
      komo->groundTasks( 0., node->data().leadingKomoArgs );

      if( node->isRoot() ) komo->applyRandomization( randomVec_ );
      komo->reset(); //huge

      try{
        komo->run();
      } catch( const char* msg ){
        cout << "KOMO FAILED: " << msg <<endl;
      }
      //komo->checkGradients();
//      if( node->id() == 3 )
//      {
//        komo->displayTrajectory();

//        rai::wait();
//      }
      // save results
      //    DEBUG( komo->MP->reportFeatures(true, FILE("z.problem")); )

      Graph result = komo->getReport();

      double cost = result.get<double>( { "total","sqrCosts" } );
      double constraints = result.get<double>( { "total","constraints" } );

      poseCosts_[ node->data().decisionGraphNodeId ]( w )       = cost;
      poseConstraints_[ node->data().decisionGraphNodeId ]( w ) = constraints;

      // what to do with the cost and constraints here??
      if( constraints >= config_.maxConstraint_ )
      {
        feasible = false;
      }

      // update effective kinematic
      effKinematics_[ node->data().decisionGraphNodeId ]( w ) = *komo->configurations.last();

      // update switch
      for( rai::KinematicSwitch *sw: komo->switches )
      {
        //    CHECK_EQ(sw->timeOfApplication, 1, "need to do this before the optimization..");
        if( sw->timeOfApplication>=2 ) sw->apply( effKinematics_[ node->data().decisionGraphNodeId ]( w ) );
      }
      effKinematics_[ node->data().decisionGraphNodeId ]( w ).getJointState();

      // free
      freeKomo( komo );
    }
  }

  // solve for next nodes if this one was feasible
  if( feasible )
  {
    for( const auto& c : node->children() )
    {
      optimizePosesFrom( c );
    }
  }
}

void KOMOPlanner::savePoseOptimizationResults( Policy & policy, bool & poseOptimizationFailed ) const
{
  std::list< Policy::GraphNodeTypePtr > fifo;
  fifo.push_back( policy.root() );

  while( ! fifo.empty()  )
  {
    auto node = fifo.back();
    fifo.pop_back();

    double maxConstraint = 0;
    auto cIt = poseConstraints_.find(node->data().decisionGraphNodeId);

    CHECK(cIt != poseConstraints_.end(), "the result of the pose optimization should be contained in the map");
    for( const auto& constraint : cIt->second )
    {
      maxConstraint = std::max( constraint, maxConstraint );
    }

    if( maxConstraint >= config_.maxConstraint_ )
    {
      std::cout << "Pose Optimization failed on node " << node->id() << " max constraint:" << maxConstraint << std::endl;
      std::cout << "action: " << std::endl;

      for(const auto & arg: node->data().leadingKomoArgs)
        std::cout << arg << " ";
      std::cout << std::endl;

      node->data().markovianReturn = std::numeric_limits< double >::lowest();
      node->data().status = PolicyNodeData::INFORMED;

      poseOptimizationFailed = true;
      policy.setValue( std::numeric_limits< double >::lowest() );
    }
    else
    {
      // push children on list
      for( const auto& c : node->children() )
      {
        fifo.push_back( c );
      }
    }
  }

  policy.setStatus( Policy::INFORMED );
}

// markovian path
void KOMOPlanner::optimizeMarkovianPath( Policy & policy )
{
  std::cout << "optimizing markovian paths.." << std::endl;

  optimizeMarkovianPathFrom( policy.root() );
}

void KOMOPlanner::optimizeMarkovianPathFrom( const Policy::GraphNodeTypePtr & node )
{
  //std::cout << "optimizing markovian path for:" << node->id() << std::endl;

  bool feasible = true;

  if( markovianPathCosts_.find( node->data().decisionGraphNodeId ) == markovianPathCosts_.end() )
  {
    const auto N = node->data().beliefState.size();
    effMarkovianPathKinematics_[ node->data().decisionGraphNodeId ] = rai::Array< rai::KinematicWorld >( N );

    markovianPathCosts_      [ node->data().decisionGraphNodeId ] = 0;
    markovianPathConstraints_[ node->data().decisionGraphNodeId ] = 0;

    for( auto w = 0; w < N; ++w )
    {
      if( node->data().beliefState[ w ] > eps )
      {
        if(!node->isRoot()) CHECK( effMarkovianPathKinematics_.find( node->parent()->data().decisionGraphNodeId ) != effMarkovianPathKinematics_.end(),"no parent effective kinematic!" );

        rai::KinematicWorld kin = node->isRoot() ? *( startKinematics_( w ) ) : ( effMarkovianPathKinematics_.find( node->parent()->data().decisionGraphNodeId )->second( w ) );

        //kin.calc_q(); // this line is necessary to have the assert being valid (since rai version)
        //CHECK( kin.q.size() > 0, "wrong start configuration!");

        // create komo
        auto komo = komoFactory_.createKomo();

        // set-up komo
        komo->setModel( kin, true/*, false, true, false, false*/ );
        komo->setTiming( 1.0, config_.microSteps_, config_.secPerPhase_, 2 );
        komo->setSquaredQAccelerations();

        komo->groundInit();
        komo->groundTasks(0, node->data().leadingKomoArgs );

        if( node->isRoot() ) komo->applyRandomization( randomVec_ );
        komo->reset(); //huge

        try{
          komo->run();
        } catch( const char* msg ){
          cout << "KOMO FAILED: " << msg <<endl;
        }

//        if( node->id() == 1 )
//        {
//            komo->displayTrajectory();
//            komo->saveTrajectory( std::to_string( node->id() ) );
//            komo->plotVelocity( std::to_string( node->id() ) );
//           //rai::wait();
//        }

        Graph result = komo->getReport();

        double cost = result.get<double>( { "total","sqrCosts" } );
        double constraints = result.get<double>( { "total","constraints" } );

        markovianPathCosts_      [ node->data().decisionGraphNodeId ] += node->data().beliefState[ w ] * cost;
        markovianPathConstraints_[ node->data().decisionGraphNodeId ] += node->data().beliefState[ w ] * constraints;

        // what to do with the cost and constraints here??
        if( constraints >= config_.maxConstraint_ )
        {
          feasible = false;
        }

        // update effective kinematic
        effMarkovianPathKinematics_[ node->data().decisionGraphNodeId ]( w ) = *komo->configurations.last();

        // update switch
        for( rai::KinematicSwitch * sw: komo->switches )
        {
          //    CHECK_EQ(sw->timeOfApplication, 1, "need to do this before the optimization..");
          if( sw->timeOfApplication>=2 ) sw->apply( effMarkovianPathKinematics_[ node->data().decisionGraphNodeId ]( w ) );
        }
        //effKinematics_[ node ]( w ).topSort();
        effMarkovianPathKinematics_[ node->data().decisionGraphNodeId ]( w ).getJointState();

        // free
        freeKomo( komo );
      }
    }
  }
  // solve for next nodes if this one was feasible
  if( feasible )
  {
    for( const auto& c : node->children() )
    {
      optimizeMarkovianPathFrom( c );
    }
  }
}

void KOMOPlanner::saveMarkovianPathOptimizationResults( Policy & policy ) const
{
  std::list< Policy::GraphNodeTypePtr > fifo;
  fifo.push_back( policy.root() );

  while( ! fifo.empty()  )
  {
    auto node = fifo.back();
    fifo.pop_back();

    auto kIt = markovianPathConstraints_.find(node->data().decisionGraphNodeId);
    auto cIt = markovianPathCosts_.find(node->data().decisionGraphNodeId);
    CHECK(kIt != markovianPathConstraints_.end(), "map should contain optimization results");

    double constraint = kIt->second;
    double cost = cIt->second;

    if( constraint >= config_.maxConstraint_ )
    {
      std::cout << "Markovian Optimization failed on node " << node->id() << " constraint:" << constraint << std::endl;

      node->data().markovianReturn = std::numeric_limits< double >::lowest();
      //node->setValue( std::numeric_limits< double >::lowest() );
      //node->setStatus( PolicyNode::INFORMED );

      policy.setValue( std::numeric_limits< double >::lowest() );
    }
    else
    {
      node->data().markovianReturn =  -( config_.minMarkovianCost_ + cost );
      node->data().status = PolicyNodeData::INFORMED;

      // push children on list
      for( const auto& c : node->children() )
      {
        fifo.push_back( c );
      }
    }
  }

  /// UPDATE VALUES
  updateValues( policy );
  policy.setStatus( Policy::INFORMED );
}

///NON MARKOVIAN
void KOMOPlanner::clearLastNonMarkovianResults()
{
  // path
  for( auto& pair : pathKinFrames_ )
  {
    pair.second.clear();
  }
  pathKinFrames_.clear(); // maps each leaf to its path

  // joint path
  for( auto& pair : jointPathKinFrames_ )
  {
    pair.second.clear();
  }

  jointPathKinFrames_.clear(); // maps each leaf to its path
}
// path
void KOMOPlanner::optimizePath( Policy & policy )
{
  std::cout << "optimizing full path.." << std::endl;

  bsToLeafs_             = rai::Array< PolicyNodePtr > ( policy.N() );

  std::list<std::future<void>> futures;
  auto leaves = policy.leaves();
  for( const auto& l : leaves )
  {
     auto future = std::async(config_.executionPolicy_,
                              [&]{
                                  optimizePathTo( l );
                                });

     futures.push_back(std::move(future));
  }

  for(auto &future: futures)
  {
    future.get();
  }

  computePathQResult(policy);
}

void KOMOPlanner::optimizePathTo( const PolicyNodePtr & leaf )
{
  const auto N = leaf->data().beliefState.size();

  pathKinFrames_[ leaf->id() ] = rai::Array< rai::Array< rai::KinematicWorld > >( N );
  pathXSolution_[ leaf->id() ] = rai::Array< arr                               >( N );
  pathCostsPerPhase_[ leaf->id() ] = rai::Array< arr >( N );

  //-- collect 'path nodes'
  auto treepath = getPathTo( leaf );

  // solve problem for all ( relevant ) worlds
  for( auto w = 0; w < N; ++w )
  {
    if( leaf->data().beliefState[ w ] > eps )
    {
      // indicate this leaf as terminal for this node, this is used during the joint optimization..
      bsToLeafs_( w ) = leaf;

      // create komo
      auto komo = komoFactory_.createKomo();

      // set-up komo
      auto leafTime = leaf->depth();
      komo->setModel( *startKinematics_( w ), true/*, false, true, false, false*/ );
      komo->setTiming( leafTime, config_.microSteps_, config_.secPerPhase_, 2 );
      komo->setSquaredQAccelerations();

      komo->groundInit();

      for( const auto& node:treepath )
      {
        auto time = ( node->parent() ? node->parent()->depth(): 0. );     // get parent time
        komo->groundTasks( time, node->data().leadingKomoArgs ); // ground parent action (included in the initial state)
      }

      komo->applyRandomization( randomVec_ );
      komo->reset();
      //komo->verbose = 3;
      try{
        komo->run();
      } catch(const char* msg){
        cout << "KOMO FAILED: " << msg <<endl;
      }

      // all the komo lead to the same agent trajectory, its ok to use one of it for the rest
//      if( leaf->id() == 10 )
//      {
//        //      komo->plotTrajectory();
//        komo->displayTrajectory( 0.02, true );
//        komo->saveTrajectory( "-j-" + std::to_string( w ) );
//          komo->plotVelocity();// "-j-"   + std::to_string( w ) );
//      }

      auto costs = komo->getCostsPerPhase();
      Graph result = komo->getReport();
      double cost        = result.get<double>( {"total","sqrCosts"} );
      double constraints = result.get<double>( {"total","constraints"} );

      pathCostsPerPhase_[ leaf->id() ]( w ) = costs;

      for( auto s = 0; s < komo->configurations.N; ++s )
      {
        rai::KinematicWorld kin( *komo->configurations( s ) );
        pathKinFrames_[ leaf->id() ]( w ).append( kin );
      }

      pathXSolution_[ leaf->id() ]( w ) = komo->x;

      // free
      freeKomo( komo );
    }
  }
}

void KOMOPlanner::computePathQResult( const Policy& policy )
{
  pathQResult_ = QResult( policy.N(), qmask_, config_.microSteps_ );
  for( uint w = 0; w < bsToLeafs_.size(); ++w )
  {
    const PolicyNodePtr leaf = bsToLeafs_.at(w);
    const auto& trajForW = pathKinFrames_.at(leaf->id()).at(w);
    const uint nSteps = trajForW.size();

    pathQResult_.createTrajectory(w, nSteps);

    for( uint s = 0; s < nSteps; ++s )
    {
      pathQResult_.setQ( w, s, trajForW.at(s).q );
    }
  }
}

void KOMOPlanner::optimizeJointPath( Policy & policy )
{
  std::cout << "optimizing full joint path.." << std::endl;

  std::list<std::future<void>> futures;
  auto leaves = policy.leaves();
  for( const auto& l : leaves )
  {
    auto future = std::async(config_.executionPolicy_,
                             [&]{
      optimizeJointPathTo( l );
    });
    futures.push_back(std::move(future));
  }

  for(auto &future: futures)
  {
    future.get();
  }

  computeJointPathQResult( policy );
}

void KOMOPlanner::optimizeJointPathTo( const PolicyNodePtr & leaf )
{
  const auto N = leaf->data().beliefState.size();

  jointPathKinFrames_  [ leaf->id() ] = rai::Array< rai::Array< rai::KinematicWorld > >( N );
  jointPathCostsPerPhase_[ leaf->id() ] = rai::Array< arr >( N );

  //-- collect 'path nodes'
  auto treepath = getPathTo( leaf );

  // solve problem for all ( relevant ) worlds
  for( auto w = 0; w < N; ++w )
  {
    if( leaf->data().beliefState[ w ] > eps )
    {
      // create komo
      auto komo = komoFactory_.createKomo();

      // set-up komo
      auto leafTime = leaf->depth();
      komo->setModel( *startKinematics_( w ), true/*, false, true, false, false*/ );
      komo->setTiming( leafTime, config_.microSteps_, config_.secPerPhase_, 2 );
      komo->setSquaredQAccelerations();

      komo->groundInit();

      for( const auto& node:treepath )
      {
        // set task
        auto start = ( node->parent() ? node->parent()->depth(): 0. );     // get parent time
        komo->groundTasks( start, node->data().leadingKomoArgs );          // ground parent action (included in the initial state)

        if( node->depth() > 0 )
        {
          for( auto s = 1; s < komo->stepsPerPhase; ++s )
          {
            uint stepsPerPhase = komo->stepsPerPhase; // get number of steps per phases
            uint nodeSlice = stepsPerPhase * node->depth() - s;
            arr q = zeros( pathKinFrames_[ leaf->id() ]( w )( nodeSlice ).q.N );

            // set constraints enforcing the path equality among worlds
            uint nSupport = 0;
            for( auto x = 0; x < N; ++x )
            {
              if( node->data().beliefState[ x ] > 0 )
              {
                CHECK( bsToLeafs_( x ) != nullptr, "no leaf for this state!!?" );

                const auto& terminalLeafx = bsToLeafs_( x );

                CHECK( pathKinFrames_[ terminalLeafx->id() ]( x ).N > 0, "one node along the solution path doesn't have a path solution already!" );

                const auto & pathLeafx     = pathKinFrames_[ terminalLeafx->id() ]( x );

                CHECK_EQ( q.N, pathLeafx( nodeSlice ).q.N, "wrong q dimensions!" );

                q += node->data().beliefState[ x ] * pathLeafx( nodeSlice ).q;

                nSupport++;
              }
            }

            if( nSupport > 1 )  // enforce kin equality between at least two worlds, useless with just one world!
            {
              AgentKinEquality * task = new AgentKinEquality( node->id(), q, qmask_ );  // tmp camille, think to delete it, or komo does it?
              double slice_t = node->depth() - double( s ) / stepsPerPhase;
              komo->addObjective( slice_t, slice_t, task, OT_eq, NoArr, config_.kinEqualityWeight_ );

              //
              //std::cout << "depth:" << node->depth() << " slice:" << slice_t << " has kin equality, q size = " << qmask.size() << std::endl;
              //
            }
          }
        }
      }

      komo->applyRandomization( randomVec_ );
      komo->set_x( pathXSolution_[ leaf->id() ]( w ) );
      komo->reset();

      try{
        komo->run();
      } catch(const char* msg){
        cout << "KOMO FAILED: " << msg <<endl;
      }

      // all the komo lead to the same agent trajectory, its ok to use one of it for the rest
      //komo->displayTrajectory();
//     if( leaf->id() == 7 )
//     {
//  //      komo->plotTrajectory();
//        komo->displayTrajectory( 0.02, true );
//        komo->saveTrajectory( "-j-" + std::to_string( w ) );
//        komo->plotVelocity( "-j-"   + std::to_string( w ) );
//     }

      auto costs = komo->getCostsPerPhase();
      const auto& result = komo->getReport();

      double cost = result.get<double>({"total","sqrCosts"});
      double constraints = result.get<double>({"total","constraints"});

      // store costs
      jointPathCostsPerPhase_[ leaf->id() ]( w ) = costs;

      // store result
      for( auto s = 0; s < komo->configurations.N; ++s )
      {
        const rai::KinematicWorld& kin( *komo->configurations( s ) );
        jointPathKinFrames_[ leaf->id() ]( w ).append( kin );
      }

      // free
      freeKomo( komo );
    }
  }
}

void KOMOPlanner::computeJointPathQResult( const Policy& policy )
{
  jointPathQResult_ = QResult( policy.N(), qmask_, config_.microSteps_ );
  for( uint w = 0; w < bsToLeafs_.size(); ++w )
  {
    const PolicyNodePtr leaf = bsToLeafs_.at(w);
    const auto& trajForW = jointPathKinFrames_.at(leaf->id()).at(w);
    const uint nSteps = trajForW.size(); //  .at(w)->size();

    jointPathQResult_.createTrajectory(w, nSteps);

    for( uint s = 0; s < nSteps; ++s )
    {
      jointPathQResult_.setQ( w, s, trajForW.at(s).q );
    }
  }
}

void KOMOPlanner::saveJointPathOptimizationResults( Policy & policy ) const
{
  /// INFORM POLICY NODES
  std::list< Policy::GraphNodeTypePtr > fifo;
  fifo.push_back( policy.root() );

  while( ! fifo.empty()  )
  {
    auto node = fifo.back();
    fifo.pop_back();

    uint phase = node->depth();

    double cost = 0;

    // get the right world
    auto & pathCostsPerPhase = jointPathKinFrames_.size() > 1 ? jointPathCostsPerPhase_ : pathCostsPerPhase_;
    for( auto w = 0; w < node->data().beliefState.size(); ++w )
    {
      if( node->data().beliefState[ w ] > 0 )
      {
        const auto& leaf = bsToLeafs_( w );
        auto tIt = pathCostsPerPhase.find(leaf->id());

        CHECK( pathCostsPerPhase.find( leaf->id() ) != pathCostsPerPhase.end(), "corruption in datastructure" );
        CHECK(tIt!=pathCostsPerPhase.end(), "optimization results should be in the map");

        const auto& trajCosts = tIt->second( w );
        const auto& wcost = trajCosts( phase - 1 );

        cost += node->data().beliefState[ w ] * wcost;
        //std::cout << "cost of phase:" << cost << " phase:" << phase << std::endl;
      }
    }

    // push children on list
    for( const auto& c : node->children() )
    {
      c->data().markovianReturn = - cost;

      fifo.push_back( c );
    }
  }

  /// UPDATE VALUES AND STATUS
  updateValues( policy );
  policy.setQResult(policy.N()>1 ? jointPathQResult_ : pathQResult_);
  policy.setStatus( Policy::INFORMED );
}

}
