#include <komo_wrapper.h>
#include <Kin/switch.h>
#include <Kin/TM_angVel.h>
#include <Kin/TM_default.h>
#include <Kin/TM_gravity.h>
#include <Kin/TM_InsideBox.h>
#include <Kin/TM_AboveBox.h>

using namespace rai;

namespace mp
{
void KomoWrapper::reset(const std::vector<Vars>& branches, double initNoise)
{
  CHECK(komo_->sparseOptimization, "valid only in sparse mode!");

  if(!komo_->configurations.N) setupConfigurations(branches);
  komo_->x = komo_->getPath_decisionVariable();
  komo_->dual.clear();
  komo_->featureValues.clear();
  komo_->featureTypes.clear();
  komo_->komo_problem.clear();
  komo_->dense_problem.clear();
  if(initNoise>0.)
    rndGauss(komo_->x, initNoise, true); //don't initialize at a singular config
  if(komo_->splineB.N) {
    komo_->z = pseudoInverse(komo_->splineB) * komo_->x;
  }
}

void KomoWrapper::setupConfigurations(const std::vector<Vars>& branches)
{
  //IMPORTANT: The configurations need to include the k prefix configurations!
  //Therefore configurations(0) is for time=-k and configurations(k+t) is for time=t
  CHECK(!komo_->configurations.N, "why setup again?");
//    listDelete(configurations);

  computeMeshNormals(komo_->world.frames, true);

  komo_->configurations.resize(komo_->T + komo_->k_order);
  for(auto s = 0; s < komo_->configurations.d0; ++s) komo_->configurations(s) = new KinematicWorld();

  //komo_->configurations.append(new KinematicWorld())->copy(komo_->world, true);
  komo_->configurations(0)->copy(komo_->world, true);
  komo_->configurations(0)->setTimes(komo_->tau); //(-tau*k_order);
  komo_->configurations(0)->calc_q();
  komo_->configurations(0)->checkConsistency();
  for(KinematicSwitch *sw:komo_->switches) {
    if(sw->timeOfApplication+(int)komo_->k_order<=0) {
      sw->apply(*komo_->configurations(0));
    }
  }

  komo_->configurations(1)->copy(*komo_->configurations(0), true);
  komo_->configurations(2)->copy(*komo_->configurations(1), true);
  std::vector<int> visited(komo_->T+komo_->k_order, 0);

  uint i = 0;
  for(const auto& branch: branches)
  {
    std::cout << "setup configurations branch " << i << std::endl; ++i;

    for(uint s=0; s<branch.order0.d0; s++)
    {
      auto s_global = branch.order1(s,1);
      auto s_m_1_global = branch.order1(s,0);

      if(visited[s_global + komo_->k_order]) continue;
      visited[s_global + komo_->k_order] = 1;

      komo_->configurations(s_global + komo_->k_order)->copy(*komo_->configurations(s_m_1_global + komo_->k_order), true);
      rai::KinematicWorld& K = *komo_->configurations(s_global + komo_->k_order);
      CHECK(K.frames.d0>0, "Copied wrong element");
      K.setTimes(komo_->tau); //(tau*(int(s)-int(k_order)));
      K.checkConsistency();

      //apply potential graph switches
      for(KinematicSwitch *sw:komo_->switches) {
        if(sw->timeOfApplication == s_global) {
          sw->apply(K);
          //std::cout << "  apply switch at" << s_global << std::endl;
        }
      }
      //apply potential PERSISTENT flags
      for(Flag *fl:komo_->flags) {
        if(fl->persist && fl->stepOfApplication == s_global) {
          fl->apply(K);
        }
      }
      K.calc_q();
      K.checkConsistency();
      //    {
      //      cout <<"CONFIGURATION s-k_order=" <<int(s)-k_order <<endl;
      //      K.glAnimate();
      //      rai::wait();
      ////      K.glClose();
      //    }
    }
    //++i;
  }

  //now apply NON-PERSISTENT flags
  //for(uint s=1; s<komo_->k_order+komo_->T; s++) {
  for(const auto& branch: branches)
  for(uint s=1; s<branch.order0.d0; s++) {
    auto s_global = branch.order0(s,0);
    for(Flag *fl:komo_->flags) {
      if(!fl->persist && fl->stepOfApplication==s_global) {
        fl->apply(*komo_->configurations(s_global+komo_->k_order));
      }
    }
  }
}

void KomoWrapper::addObjective(const Interval& it, const TreeBuilder& tree, Feature* map, ObjectiveType type, const arr& target, double scale, int order, int deltaFromStep, int deltaToStep)
{
  if(tree.n_nodes())
  {
    CHECK(komo_->sparseOptimization, "should be there in sparse mode only")
    CHECK(scale != -1, "please put a meaningful scale");
    CHECK(order != -1, "please put a meaningful task order");

    auto spec = tree.get_spec(it.time, it.edge, order, komo_->stepsPerPhase);
    auto obj = komo_->addObjective(-123., -123., map, type, target, scale, order, deltaFromStep, deltaToStep);

    obj->vars = spec.vars;
    obj->scales = spec.scales;
  }
  else
  {
    komo_->addObjective(it.time.from, it.time.to, map, type, target, scale, order, deltaFromStep, deltaToStep);
  }
}

void KomoWrapper::addSwitch(const Interval& it, const TreeBuilder& tree, KinematicSwitch * sw)
{
  CHECK(it.time.from == it.time.to, "Wrong interval for a switch");

  if(tree.n_nodes())
  {
    CHECK(komo_->sparseOptimization, "should be there in sparse mode only")
    sw->timeOfApplication = tree.get_step(it.time.to, it.edge, komo_->stepsPerPhase);
    komo_->switches.append(sw);
  }
  else
  {
    komo_->addSwitch(it.time.from, true, sw);
  }
}

void ADMM_MotionProblem_GraphProblem::getStructure(uintA& variableDimensions, intAA& featureVariables, ObjectiveTypeA& featureTypes)
{
  CHECK_EQ(komo.configurations.N, komo.k_order+komo.T, "configurations are not setup yet: use komo.reset()");
  if(!!variableDimensions) {
    variableDimensions.resize(komo.T);
    for(uint t=0; t<komo.T; t++)
      variableDimensions(t) = komo.configurations(t+komo.k_order)->getJointStateDimension();
  }

  if(!!featureVariables) featureVariables.clear();
  if(!!featureTypes) featureTypes.clear();
  uint M=0;
  for(Objective *ob:komo.objectives) {
    CHECK_EQ(ob->vars.nd, 2, "in sparse mode, vars need to be tuples of variables");
    for(uint t=0;t<ob->vars.d0;t++)
      if(tmask(ob->vars(t, -1)))
      {
        WorldL Ktuple = komo.configurations.sub(convert<uint,int>(ob->vars[t]+(int)komo.k_order));
        uint m = ob->map->__dim_phi(Ktuple); //dimensionality of this task
        if(!!featureVariables) featureVariables.append(ob->vars[t], m);
        if(!!featureTypes) featureTypes.append(ob->type, m);
        M += m;
      }
  }

  // Add admm terms




  //

  if(!!featureTypes) komo.featureTypes = featureTypes;

  dimPhi = M;
}

void ADMM_MotionProblem_GraphProblem::phi(arr& phi, arrA& J, arrA& H, const arr& x)
{
  //-- set the trajectory
  komo.set_x(x);

//  if(!dimPhi) getStructure();
  CHECK(dimPhi,"getStructure must be called first");
  phi.resize(dimPhi);
  if(!!J) J.resize(dimPhi);

//  uintA x_index = getKtupleDim(komo.configurations({komo.k_order,-1}));
//  x_index.prepend(0);

  rai::timerStart();
  arr y, Jy;
  uint M=0;
  for(Objective *ob:komo.objectives) {
    CHECK_EQ(ob->vars.nd, 2, "in sparse mode, vars need to be tuples of variables");
    for(uint t=0;t<ob->vars.d0;t++)
    if(tmask(ob->vars(t, -1)))
    {
      const auto scale = ob->scales.d0 ? ob->scales(t) : 1.0;
      WorldL Ktuple = komo.configurations.sub(convert<uint,int>(ob->vars[t]+(int)komo.k_order));
      uintA kdim = getKtupleDim(Ktuple);
      kdim.prepend(0);

      //query the task map and check dimensionalities of returns
      ob->map->__phi(y, (!!J?Jy:NoArr), Ktuple);

      if(!!J) CHECK_EQ(y.N, Jy.d0, "");
      if(!!J) CHECK_EQ(Jy.nd, 2, "");
      if(!!J) CHECK_EQ(Jy.d1, kdim.last(), "");
      if(!y.N) continue;
      if(absMax(y)>1e10) RAI_MSG("WARNING y=" <<y);

      //write into phi and J
      phi.setVectorBlock(scale * y, M); // appy scale on y

      if(!!J) {
        for(uint j=ob->vars.d1;j--;){
          if(ob->vars(t,j)<0){
            Jy.delColumns(kdim(j),kdim(j+1)-kdim(j)); //delete the columns that correspond to the prefix!!
          }
        }
        for(uint i=0; i<y.N; i++) J(M+i) = scale * Jy[i]; // appy scale on J
      }

      //counter for features phi
      M += y.N;
    }
  }

  komo.timeFeatures += rai::timerRead(true);

  CHECK_EQ(M, dimPhi, "");
  //  if(!!lambda) CHECK_EQ(prevLambda, lambda, ""); //this ASSERT only holds is none of the tasks is variable dim!
  komo.featureValues = phi;
  komo.featureDense = true;
}

//void ADMM_MotionProblem_GraphProblem::getXMask(arr & xmask, bool withPrefix) const
void ADMM_MotionProblem_GraphProblem::getXMask(arr & xmask) const
{
  CHECK(komo.x.d0 > 0, "should reset first!");

  xmask = zeros(komo.x.d0);

  // translate step time to
  //uint s = komo.freePrefix ? komo.k_order * komo.world.getJointStateDimension() : 0; // freePrefix deactivated
  uint s = 0;
  std::vector<std::pair<uint, uint>> t_to_x_interval(komo.T);
  for(uint t=0; t<komo.T; t++)// x.append(configurations(t+k_order)->getJointState());
  {
    uint n = komo.configurations(t+komo.k_order)->getJointStateDimension();
    t_to_x_interval[t] = std::pair<uint, uint>(s, s+n);
    s+=n;
  }

  // fill mask based on objective
  uint start_t = komo.T; // will be updated in the loop
  for(Objective *ob:komo.objectives)
  {
    for(uint t=0;t<ob->vars.d0;t++)
    {
      auto global = ob->vars(t, -1);
      if(tmask(global))
      {
        if(global < start_t) start_t = global;
        const auto& xinterval = t_to_x_interval[global];
        for(auto i = xinterval.first; i < xinterval.second; ++i)
        {
          xmask(i) = 1.0;
        }
      }
    }
  }

  // special handling for prefix (add the prefix in the optimization variable)
//  if(withPrefix)
//  {
//    uint start_x = t_to_x_interval[start_t].first;
//    uint n = 0;
//    for(auto o = 0; o < komo.k_order; ++o)
//      n += komo.configurations(start_t - o + komo.k_order)->getJointStateDimension();
//    //uint nm1 = komo.configurations(start_t + komo.k_order)->getJointStateDimension();
//    //uint nm2 = komo.configurations(start_t - 1 + komo.k_order)->getJointStateDimension();
//    for(auto i = start_x - n; i < start_x; ++i)
//    {
//      xmask(i) = 1;
//    }
//  }
}

}
