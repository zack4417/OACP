/*  ------------------------------------------------------------------
    Copyright (c) 2017 Marc Toussaint
    email: marc.toussaint@informatik.uni-stuttgart.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#include "komo.h"
#include "komo-ext.h"
#include <Algo/spline.h>
#include <iomanip>
#include <Kin/frame.h>
#include <Kin/switch.h>
#include <Kin/kin_swift.h>
#include <Kin/taskMaps.h>
#include <Gui/opengl.h>
#include <Kin/TM_FixSwitchedObjects.h>
#include <Kin/TM_QuaternionNorms.h>
#include <Kin/TM_AboveBox.h>
#include <Kin/TM_InsideBox.h>
#include <Kin/TM_AlignStacking.h>
#include <Kin/TM_linTrans.h>
#include <Kin/TM_StaticStability.h>
#include <Kin/TM_Max.h>
#include <Kin/TM_ImpulseExchange.h>
#include <Kin/TM_ContactConstraints.h>
#include <Kin/TM_FlagConstraints.h>
#include <Kin/TM_energy.h>
#include <Kin/TM_gravity.h>
#include <Kin/contact.h>
#include <Optim/optimization.h>
#include <Optim/convert.h>
#include <Optim/primalDual.h>
#include <Optim/GraphOptim.h>
#include <Kin/kin_physx.h>
#include <Kin/TM_time.h>
#include <Kin/TM_NewtonEuler.h>
#include <Kin/TM_angVel.h>
#include <Kin/F_pushed.h>

#ifdef RAI_GL
#  include <GL/gl.h>
#endif

using namespace rai;

//===========================================================================

double shapeSize(const KinematicWorld& K, const char* name, uint i=2);

Shape *getShape(const KinematicWorld& K, const char* name) {
  Frame *f = K.getFrameByName(name);
  Shape *s = f->shape;
  if(!s) {
    for(Frame *b:f->parentOf) if(b->name==name && b->shape) { s=b->shape; break; }
  }
  return s;
}

KOMO::KOMO() : useSwift(true), useSwitches(true), verbose(1), komo_problem(*this), dense_problem(*this), graph_problem(*this) {
  verbose = getParameter<int>("KOMO/verbose",1);
}

KOMO::KOMO(const KinematicWorld& K, bool )
  : KOMO() {
  setModel(K, true);
  world.optimizeTree();
  world.calc_q();
}

KOMO::~KOMO() {
  gl.reset();
  if(opt) delete opt;
  if(logFile) delete logFile;
  listDelete(objectives);
  listDelete(flags);
  listDelete(switches);
  listDelete(configurations);
}

void KOMO::setModel(const KinematicWorld& K, bool _useSwift) {
  if(&K!=&world) world.copy(K, _useSwift);
  useSwift = _useSwift;
  if(useSwift) world.swift();
  world.calc_q();
}

void KOMO_ext::useJointGroups(const StringA& groupNames, bool OnlyTheseOrNotThese) {
  world.selectJointsByGroup(groupNames, OnlyTheseOrNotThese, false);
  
  world.reset_q();
  world.optimizeTree();
  world.getJointState();
  
//  world.meldFixedJoints();
//  world.removeUselessBodies();

//  FILE("z.komo.model") <<world;
}

void KOMO::setTiming(double _phases, uint _stepsPerPhase, double durationPerPhase, uint _k_order) {
  double maxPhase = _phases;
  stepsPerPhase = _stepsPerPhase;
  if(stepsPerPhase>=0) {
    T = ceil(stepsPerPhase*maxPhase);
    CHECK(T, "using T=0 to indicate inverse kinematics is deprecated.");
    tau = durationPerPhase/double(stepsPerPhase);
  }
//    setTiming(stepsPerPhase*maxPhase, durationPerPhase*maxPhase);
  k_order = _k_order;
}

void KOMO::setPairedTimes() {
  CHECK_EQ(k_order, 1, "NIY");
  for(uint s=1; s<T; s+=2) {
    configurations(s)  ->setTimes(1.98*tau); //(tau*(.98+int(s+1)-int(k_order)));
    configurations(s+1)->setTimes(0.02*tau); //(tau*(int(s)-int(k_order)));
  }
}

void KOMO::activateCollisions(const char* s1, const char* s2) {
  if(!useSwift) return;
  Frame *sh1 = world.getFrameByName(s1);
  Frame *sh2 = world.getFrameByName(s2);
  if(sh1 && sh2) world.swift().activate(sh1, sh2);
}

void KOMO::deactivateCollisions(const char* s1, const char* s2) {
  if(!useSwift) return;
  Frame *sh1 = world.getFrameByName(s1);
  Frame *sh2 = world.getFrameByName(s2);
  if(sh1 && sh2) world.swift().deactivate(sh1, sh2);
  else LOG(-1) <<"not found:" <<s1 <<' ' <<s2;
}

void KOMO::setTimeOptimization(){
  world.addTimeJoint();
  Objective* o = addObjective(0., -1., new TM_Time(), OT_eq, {}, 1e2, 1); //smooth time evolution
  //break the constraint at phase switches:
  CHECK(o->vars.nd==1 && o->vars.N==T, "");
  CHECK_GE(stepsPerPhase, 10, "NIY")
  for(uint t=1;t<o->vars.N; t+=stepsPerPhase) o->vars(t)=0;

  addObjective(0., -1., new TM_Time(), OT_sos, {tau}, 1e-1); //prior on timing
//  addObjective(0., -1., new TM_Time(), OT_ineq, {tau}, -1e1); //lower bound on timing
}

//===========================================================================
//
// task specs
//

void KOMO::clearObjectives() {
  listDelete(objectives);
  listDelete(switches);
  listDelete(flags);
}


Objective *KOMO::addObjective(double startTime, double endTime,
                              const ptr<Feature>& map, ObjectiveType type,
                              const arr& target, double scale, int order,
                              int deltaFromStep, int deltaToStep) {
  if(!!target) map->target = target;
  if(scale>0. && scale!=1.) map->scale = ARR(scale);
  if(order>=0) map->order = order;
  CHECK_GE(k_order, map->order, "task requires larger k-order: " <<map->shortTag(world));
  Objective *task = new Objective(map, type);
  task->name = map->shortTag(world);
  objectives.append(task);
  if(startTime!=-123. && endTime!=-123.){ //very special case!: Only when KOMO::addObjective calls (see below) we don't set the variables
    task->setCostSpecs(startTime, endTime, stepsPerPhase, T, deltaFromStep, deltaToStep, denseOptimization || sparseOptimization);
    if(denseOptimization || sparseOptimization){
      CHECK_EQ(task->vars.nd, 2, "");
    }
  }
  return task;
}

Objective* KOMO::addObjective(double startTime, double endTime, Feature* map, ObjectiveType type, const arr& target, double scale, int order, int deltaFromStep, int deltaToStep){
  return addObjective(startTime, endTime, ptr<Feature>(map), type, target, scale, order, deltaFromStep, deltaToStep);
}

Objective* KOMO::addObjective(const arr& times, ObjectiveType type, const FeatureSymbol& feat, const StringA& frames, const arr& scale, const arr& target, int order){
  ptr<Feature> f = symbols2feature(feat, frames, world, scale, target, order);

  Objective *task = addObjective(-123.,-123., f, type);

  if(!denseOptimization && !sparseOptimization){
    if(!times.N){
      task->setCostSpecs(0, T-1);
    }else if(times.N==1){
      task->setCostSpecs(times(0), times(0), stepsPerPhase, T);
    }else{
      CHECK_EQ(times.N, 2, "");
      task->setCostSpecs(times(0), times(1), stepsPerPhase, T);
    }
  }else{
    intA vars = convert<int,double>(times);
    if(!vars.N){
      vars.resize(T, f->order+1);
      for(uint t=0;t<vars.d0;t++)
        for(uint i=0;i<vars.d1;i++) vars(t,i) = t+i-int(f->order);
      task->vars = vars;
    }else{
      uint order = vars.N-1;
      CHECK_GE(k_order, order, "task requires larger k-order: " <<task->map->shortTag(world));
      task->map->order = order;
      task->vars = vars;
      task->vars.reshape(1,vars.N);
    }
    CHECK_EQ(task->vars.nd, 2, "");
  }
  return task;
}

void KOMO::addFlag(double time, Flag *fl, int deltaStep) {
  if(time<0.) time=0.;
  fl->stepOfApplication = conv_time2step(time, stepsPerPhase) + deltaStep;
  flags.append(fl);
}

void KOMO::addSwitch(double time, bool before, KinematicSwitch *sw) {
  if(time<0.) time=0.;
  sw->setTimeOfApplication(time, before, stepsPerPhase, T);
  switches.append(sw);
}

void KOMO::addSwitch(double time, bool before, const char* type, const char* ref1, const char* ref2, const Transformation& jFrom) {
  KinematicSwitch *sw = KinematicSwitch::newSwitch(type, ref1, ref2, world, 0/*con_time2step(time, stepsPerPhase)+(before?0:1)*/, jFrom);
  addSwitch(time, before, sw);
}

void KOMO::addSwitch_mode(SkeletonSymbol prevMode, SkeletonSymbol newMode, double time, double endTime, const char* prevFrom, const char* from, const char* to){
  if(newMode==SY_stable || newMode==SY_stableOn){
    if(!useSwitches){
      if(prevMode==SY_initial){
        addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_free, world.frames.first()->name, to, world, SWInit_copy));
        addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
      }
//      addObjective({time, endTime}, OT_eq, FS_poseRel, {from, to}, {1e2}, {}, 1);
      addObjective(time, endTime, symbols2feature(FS_poseRel, {from, to}, world), OT_eq, NoArr, 1e2, 1, +1, 0);
      if(k_order>1) addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e2, 2, +0, +1);
    }else{
      if(newMode==SY_stable){
        addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_free, from, to, world, SWInit_copy));
      }else{  //SY_stableOn
        Transformation rel = 0;
        rel.pos.set(0,0, .5*(shapeSize(world, from) + shapeSize(world, to)));
        addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_transXYPhi, from, to, world, SWInit_copy, 0, rel));
      }

      //-- DOF-is-constant constraint
      if((endTime<0. && stepsPerPhase*time<T) || stepsPerPhase*endTime>stepsPerPhase*time+1){
        addObjective(time, endTime, new TM_ZeroQVel(world, to), OT_eq, NoArr, 1e1, 1, +1, -1);
        //      addObjective(time, endTime, symbols2feature(FS_poseRel, {from, to}, world), OT_eq, NoArr, 1e1, 1, +1, -1);
      }

      //-- no relative jump at end
      //    if(endTime>0.) addObjective(endTime, endTime, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);

      if(prevMode==SY_initial || prevMode==SY_stable || prevMode==SY_stableOn){
        //-- no acceleration at start: +1 EXCLUDES (x-2, x-1, x0), ASSUMPTION: this is a placement that can excert impact
        if(k_order>1) addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e2, 2, +0, +1);
        else addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
      }else{
        //-- no acceleration at start: +1 EXCLUDES (x-2, x-1, x0), ASSUMPTION: this is a placement that can excert impact
        if(k_order>1) addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e2, 2, +1, +1);
        else addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
      }
    }
  }

  if(newMode==SY_dynamic){
    addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_free, from, to, world, SWInit_copy));
    addObjective(time, endTime, new TM_NewtonEuler(world, to), OT_eq, NoArr, 1e0, k_order, +0, -1);
  //  addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e2, 2); //this should be implicit in the NE equations!
  }

  if(newMode==SY_dynamicTrans){
    addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_trans3, from, to, world, SWInit_copy));
  #if 0
    addObjective(time, endTime, new TM_Gravity2(world, to), OT_eq, NoArr, 3e1, k_order, +1, -1);
  #else
    addObjective(time, endTime, new TM_NewtonEuler(world, to, true), OT_eq, NoArr, 3e1, k_order, +0, -1);
  #endif
  }

  if(newMode==SY_dynamicOn){
    Transformation rel = 0;
    rel.pos.set(0,0, .5*(shapeSize(world, from) + shapeSize(world, to)));
    addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transXYPhi, from, to, world, SWInit_copy, 0, rel));
    if(k_order>=2) addObjective(time, endTime, new TM_ZeroAcc(world, to), OT_eq, NoArr, 3e1, k_order, +0, -1);
  //  else addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
  }
}


void KOMO::addSwitch_stable(double time, double endTime, const char* from, const char* to) {
  addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_free, from, to, world));
  //-- DOF-is-constant constraint
  if(endTime<0. || stepsPerPhase*endTime>stepsPerPhase*time+1)
    addObjective(time, endTime, new TM_ZeroQVel(world, to), OT_eq, NoArr, 3e1, 1, +1, -1);
  //-- no relative jump at end
  if(endTime>0.) addObjective(endTime, endTime, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
  //-- no object acceleration at start: +0 include (x-2, x-1, x0), which enforces a SMOOTH pickup
  if(k_order>1) addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e1, 2, +0, +1);
  else addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
}

void KOMO::addSwitch_stableOn(double time, double endTime, const char *from, const char* to) {
  Transformation rel = 0;
  rel.pos.set(0,0, .5*(shapeSize(world, from) + shapeSize(world, to)));
  addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_transXYPhi, from, to, world, SWInit_zero, 0, rel));
  //-- DOF-is-constant constraint
  if(endTime<0. || stepsPerPhase*endTime>stepsPerPhase*time+1)
    addObjective(time, endTime, new TM_ZeroQVel(world, to), OT_eq, NoArr, 3e1, 1, +1, -1);
  //-- no relative jump at end
  if(endTime>0.) addObjective(endTime, endTime, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
  //-- no acceleration at start: +1 EXCLUDES (x-2, x-1, x0), ASSUMPTION: this is a placement that can excert impact
  if(k_order>1) addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e1, 2, +1, +1);
//  else addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
}

void KOMO::addSwitch_dynamic(double time, double endTime, const char* from, const char* to) {
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_free, from, to, world, SWInit_copy));
  addObjective(time, endTime, new TM_NewtonEuler(world, to), OT_eq, NoArr, 1e0, k_order, +0, -1);
//  addObjective(time, time, new TM_LinAngVel(world, to), OT_eq, NoArr, 1e2, 2); //this should be implicit in the NE equations!
}

void KOMO::addSwitch_dynamicTrans(double time, double endTime, const char* from, const char* to) {
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_trans3, from, to, world, SWInit_copy));
#if 0
  addObjective(time, endTime, new TM_Gravity2(world, to), OT_eq, NoArr, 3e1, k_order, +1, -1);
#else
  addObjective(time, endTime, new TM_NewtonEuler(world, to, true), OT_eq, NoArr, 3e1, k_order, +0, -1);
#endif
}

void KOMO::addSwitch_dynamicOn(double time, double endTime, const char *from, const char* to) {
  Transformation rel = 0;
  rel.pos.set(0,0, .5*(shapeSize(world, from) + shapeSize(world, to)));
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transXYPhi, from, to, world, SWInit_zero, 0, rel));
  if(k_order>=2) addObjective(time, endTime, new TM_ZeroAcc(world, to), OT_eq, NoArr, 3e1, k_order, +0, -1);
//  else addObjective(time, time, new TM_NoJumpFromParent(world, to), OT_eq, NoArr, 1e2, 1, 0, 0);
}

void KOMO::addSwitch_dynamicOnNewton(double time, double endTime, const char *from, const char* to) {
  Transformation rel = 0;
  rel.pos.set(0,0, .5*(shapeSize(world, from) + shapeSize(world, to)));
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transXYPhi, from, to, world, SWInit_zero, 0, rel));
  if(k_order>=2) addObjective(time, endTime, new TM_NewtonEuler(world, to), OT_eq, NoArr, 1e0, k_order, +0, -1);
}


void KOMO::addSwitch_magic(double time, double endTime, const char* from, const char* to, double sqrAccCost) {
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_free, from, to, world, SWInit_copy));
  if(sqrAccCost>0.){
    if(k_order>=2)
      addObjective(time, endTime, new TM_LinAngVel(world, to), OT_sos, NoArr, sqrAccCost, 2);
    else
      addObjective(time, endTime, new TM_LinAngVel(world, to), OT_sos, NoArr, sqrAccCost, 1);
  }
}

void KOMO::addSwitch_magicTrans(double time, double endTime, const char* from, const char* to, double sqrAccCost) {
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transZ, from, to, world, SWInit_copy));
  if(sqrAccCost>0.){
    addObjective(time, endTime, new TM_LinAngVel(world, to), OT_eq, NoArr, sqrAccCost, 2);
  }
}

void KOMO::addSwitch_on(double time, const char *from, const char* to) {
  Transformation rel = 0;
  rel.pos.set(0,0, .5*(shapeSize(world, from) + shapeSize(world, to)));
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transXYPhi, from, to, world, SWInit_zero, 0, rel));
}

void KOMO::addContact_slide(double startTime, double endTime, const char *from, const char* to) {
  addSwitch(startTime, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, from, to, world) );
  if(endTime>0.) addSwitch(endTime, false, new rai::KinematicSwitch(rai::SW_delContact, rai::JT_none, from, to, world) );

  addObjective(startTime, endTime, new TM_Contact_ForceIsNormal(world, from, to), OT_eq, NoArr, 3e1);
  addObjective(startTime, endTime, new TM_Contact_POAisInIntersection_InEq(world, from, to), OT_ineq, NoArr, 1e1);
  addObjective(startTime, endTime, new TM_Contact_ForceRegularization(world, from, to), OT_sos, NoArr, 1e-4);
  addObjective(startTime, endTime, new TM_PairCollision(world, from, to, TM_PairCollision::_negScalar, false), OT_eq, NoArr, 1e1);
}

void KOMO::addContact_stick(double startTime, double endTime, const char *from, const char* to) {
  addSwitch(startTime, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, from, to, world) );
  if(endTime>0.) addSwitch(endTime, false, new rai::KinematicSwitch(rai::SW_delContact, rai::JT_none, from, to, world) );

  addObjective(startTime, endTime, new TM_Contact_POAisInIntersection_InEq(world, from, to), OT_ineq, NoArr, 1e1);
  addObjective(startTime, endTime, new TM_Contact_POAmovesContinuously(world, from, to), OT_sos, NoArr, 1e-1, 1, +1);
  addObjective(startTime, endTime, new TM_Contact_ForceRegularization(world, from, to), OT_sos, NoArr, 1e-4);
  addObjective(startTime, endTime, new TM_PairCollision(world, from, to, TM_PairCollision::_negScalar, false), OT_eq, NoArr, 1e1);
  addObjective(startTime, endTime, new TM_Contact_POAzeroRelVel(world, from, to), OT_eq, NoArr, 1e0, 1, +1, +1);
}

void KOMO::addContact_Complementary(double startTime, double endTime, const char* from, const char* to){
  addSwitch(startTime, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, from, to, world) );
  addObjective(startTime, endTime, new TM_Contact_ForceIsNormal(world, from, to), OT_eq, NoArr, 3e1);
  addObjective(startTime, endTime, new TM_Contact_ForceIsComplementary(world, from, to), OT_eq, NoArr, 1e2);
  addObjective(startTime, endTime, new TM_Contact_POAmovesContinuously(world, from, to), OT_sos, NoArr, 1e0, 1, +1);
  addObjective(startTime, endTime, new TM_Contact_ElasticVelIsComplementary(world, from, to, 0., 0.), OT_eq, NoArr, 1e1);

//  addObjective(startTime, endTime, new TM_ContactConstraints_Vel(world, from, to), OT_eq, NoArr, 1e1);
//  addObjective(startTime, endTime, new TM_Contact_POAisInIntersection_InEq(world, from, to), OT_ineq, NoArr, 1e1);
  addObjective(startTime, endTime, new TM_Contact_ForceRegularization(world, from, to), OT_sos, NoArr, 1e-3);
  addObjective(startTime, endTime, new TM_PairCollision(world, from, to, TM_PairCollision::_negScalar, false), OT_ineq, NoArr, 1e1);
  if(endTime>0.){
    addSwitch(endTime, false, new rai::KinematicSwitch(rai::SW_delContact, rai::JT_none, from, to, world) );
  }
}

void KOMO::addContact_noFriction(double startTime, double endTime, const char *from, const char* to) {
  addSwitch(startTime, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, from, to, world) );
  if(endTime>0.) addSwitch(endTime, false, new rai::KinematicSwitch(rai::SW_delContact, rai::JT_none, from, to, world) );

  addObjective(startTime, endTime, new TM_Contact_ForceIsNormal(world, from, to), OT_eq, NoArr, 3e1);
  addObjective(startTime, endTime, new TM_Contact_POAisInIntersection_InEq(world, from, to), OT_ineq, NoArr, 1e1);
  addObjective(startTime, endTime, new TM_Contact_POAmovesContinuously(world, from, to), OT_sos, NoArr, 1e0, 1, +1);
  addObjective(startTime, endTime, new TM_Contact_ForceRegularization(world, from, to), OT_sos, NoArr, 1e-4);
  addObjective(startTime, endTime, new TM_PairCollision(world, from, to, TM_PairCollision::_negScalar, false), OT_eq, NoArr, 1e1);
}

void KOMO::addContact_elasticBounce(double time, const char *from, const char* to, double elasticity, double stickiness) {
  addSwitch(time, true,  new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, from, to, world) );
  addSwitch(time, false, new rai::KinematicSwitch(rai::SW_delContact, rai::JT_none, from, to, world) );

  if(stickiness<=0.) addObjective(time, time, new TM_Contact_ForceIsNormal(world, from, to), OT_eq, NoArr, 3e1);
  addObjective(time, time, new TM_Contact_POAisInIntersection_InEq(world, from, to), OT_ineq, NoArr, 1e1);
  addObjective(time, time, new TM_Contact_ForceRegularization(world, from, to), OT_sos, NoArr, 1e-4);
  addObjective(time, time, new TM_PairCollision(world, from, to, TM_PairCollision::_negScalar, false), OT_eq, NoArr, 1e1);

  if(!elasticity && stickiness>=1.){
    addObjective(time, time, new TM_Contact_POAzeroRelVel(world, from, to), OT_eq, NoArr, 1e1, 2, +1, +1);
  }else{
    addObjective(time, time, new TM_Contact_ElasticVel(world, from, to, elasticity, stickiness), OT_eq, NoArr, 1e1, 2, +1, +1);
  }
}

void KOMO::addContact_staticPush(double startTime, double endTime, const char *from, const char* to) {
  addSwitch(startTime, true, new rai::KinematicSwitch(rai::SW_addContact, rai::JT_none, from, to, world) );
  if(endTime>0.) addSwitch(endTime, false, new rai::KinematicSwitch(rai::SW_delContact, rai::JT_none, from, to, world) );

  addObjective(startTime, endTime, new TM_Contact_POAisInIntersection_InEq(world, from, to), OT_ineq, NoArr, 1e1);
  addObjective(startTime, endTime, new TM_Contact_POAzeroRelVel(world, from, to), OT_eq, NoArr, 1e1, 1, +1, +0);
  addObjective(startTime, endTime, new TM_Contact_POAmovesContinuously(world, from, to), OT_sos, NoArr, 1e0, 1, +1, +0);
//  addObjective(time, time, new F_pushed(world, to), OT_eq, NoArr, 1e1, 1, +1, +0);
}

void KOMO::setKS_slider(double time, double endTime, bool before, const char* obj, const char* slider, const char* table) {
  //disconnect object from grasp ref
//  setKinematicSwitch(time, before, "delete", NULL, obj);

  //the two slider objects
  String slidera = STRING(slider <<'a');
  String sliderb = STRING(slider <<'b');
  
  Transformation rel = 0;
  rel.addRelativeTranslation(0., 0., .5*(shapeSize(world, obj) + shapeSize(world, table)));
  
//  setKinematicSwitch(time, true, "transXYPhiZero", table, slidera, rel);
//  setKinematicSwitch(time, true, "hingeZZero", sliderb, obj);
  addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_transXYPhi, table, slidera, world, SWInit_zero, 0, rel));
  addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_hingeZ, sliderb, obj, world));

  addObjective(time, endTime, new TM_ZeroQVel(world, slidera), OT_eq, NoArr, 3e1, 1, +1, +0);
  addObjective(time, endTime, new TM_ZeroQVel(world, obj), OT_eq, NoArr, 3e1, 1, +1, -1);
  addObjective(time, time, new TM_LinAngVel(world, obj), OT_eq, NoArr, 1e2, 1);

//  setKinematicSwitch(time, before, "sliderMechanism", table, obj, rel );

//  if(!actuated)
//    setKinematicSwitch(time, before, "hingeZZero", slider, obj, rel );
//  else
  //    setKinematicSwitch(time, before, "transXActuated", slider, obj, rel );
}

void KOMO::setHoming(double startTime, double endTime, double prec, const char* keyword) {
  uintA bodies;
  Joint *j;
  for(Frame *f:world.frames) if((j=f->joint) && !j->constrainToZeroVel && j->qDim()>0 && (!keyword || f->ats[keyword])) bodies.append(f->ID);
//  cout <<"HOMING: "; for(uint i:bodies) cout <<' ' <<world.frames(i)->name;  cout <<endl;
  addObjective(startTime, endTime, new TM_qItself(bodies, true), OT_sos, NoArr, prec); //world.q, prec);
}

void KOMO::setSquaredQAccelerations(double startTime, double endTime, double prec) {
  CHECK_GE(k_order, 2,"");
  addObjective(startTime, endTime, new TM_Transition(world), OT_sos, NoArr, prec);
}

void KOMO::setSquaredQAccVelHoming(double startTime, double endTime, double accPrec, double velPrec, double homingPrec) {

  uintA selectedBodies;
  arr scale;
  for(rai::Frame *f:world.frames) if(f->joint && f->joint->dim>0 && f->joint->dim<7 && f->joint->type!=JT_time && f->joint->active && f->joint->H>0.){
    CHECK(!f->joint->mimic, "")
    selectedBodies.append(TUP(f->ID, f->parent->ID));
    scale.append(f->joint->H, f->joint->dim);
  }
  selectedBodies.reshape(selectedBodies.N/2,2);
//  cout <<scale <<endl <<world.getHmetric() <<endl;
  scale *= sqrt(tau);
  if(accPrec){
    //sqr accel
    CHECK_GE(k_order, 2, "");
    Objective *o = addObjective(startTime, endTime, make_shared<TM_qItself>(selectedBodies), OT_sos, NoArr, accPrec, 2);
    o->map->scale = accPrec*scale;
  }
  if(velPrec){
    //sqr vel
    CHECK_GE(k_order, 1, "");
    Objective *o = addObjective(startTime, endTime, make_shared<TM_qItself>(selectedBodies), OT_sos, NoArr, velPrec, 1);
    o->map->scale = velPrec*scale;
  }
  if(homingPrec){
    //sqr homing
    homingPrec *= sqrt(tau);
    addObjective(startTime, endTime, make_shared<TM_qItself>(selectedBodies, true), OT_sos, NoArr, homingPrec, 0);
  }
}

void KOMO::setSquaredQVelocities(double startTime, double endTime, double prec) {
  auto *map = new TM_Transition(world);
  map->velCoeff = 1.;
  map->accCoeff = 0.;
  addObjective(startTime, endTime, map, OT_sos, NoArr, prec, 1);
}

void KOMO::setFixEffectiveJoints(double startTime, double endTime, double prec) {
//  setTask(startTime, endTime, new TM_Transition(world, true), OT_eq, NoArr, prec, 1); //NOTE: order=1!!
  addObjective(startTime, endTime, new TM_FlagConstraints(), OT_eq, NoArr, prec, k_order);
  addObjective(startTime, endTime, new TM_FlagCosts(), OT_sos, NoArr, 1., k_order);
}

void KOMO::setFixSwitchedObjects(double startTime, double endTime, double prec) {
  addObjective(startTime, endTime, new TM_FixSwichedObjects(), OT_eq, NoArr, prec, k_order);
}

void KOMO::setSquaredQuaternionNorms(double startTime, double endTime, double prec) {
  addObjective(startTime, endTime, new TM_QuaternionNorms(), OT_eq, NoArr, prec);
}

void KOMO::setHoldStill(double startTime, double endTime, const char* shape, double prec) {
  Frame *s = world.getFrameByName(shape);
  addObjective(startTime, endTime, new TM_qItself(TUP(s->ID)), OT_sos, NoArr, prec, 1);
}

void KOMO_ext::setPosition(double startTime, double endTime, const char* shape, const char* shapeRel, ObjectiveType type, const arr& target, double prec) {
  addObjective(startTime, endTime, new TM_Default(TMT_pos, world, shape, NoVector, shapeRel, NoVector), type, target, prec);
}

void KOMO_ext::setOrientation(double startTime, double endTime, const char* shape, const char* shapeRel, ObjectiveType type, const arr& target, double prec) {
//  setTask(startTime, endTime, new TM_Align(world, shape, shapeRel), type, target, prec);
  addObjective(startTime, endTime, new TM_Default(TMT_quatDiff, world, shape, NoVector, shapeRel, NoVector), type, target, prec);
}

void KOMO_ext::setVelocity(double startTime, double endTime, const char* shape, const char* shapeRel, ObjectiveType type, const arr& target, double prec) {
  addObjective(startTime, endTime, new TM_Default(TMT_pos, world, shape, NoVector, shapeRel, NoVector), type, target, prec, 1);
}

void KOMO_ext::setLastTaskToBeVelocity() {
  objectives.last()->map->order = 1; //set to be velocity!
}

void KOMO_ext::setImpact(double time, const char *a, const char *b) {
  add_touch(time, time, a, b);
  add_impulse(time, a, b);
}

void KOMO_ext::setOverTheEdge(double time, const char *object, const char *from, double margin) {
  double negMargin = margin + .5*shapeSize(world, object, 0); //how much outside the bounding box?
  addObjective(time, time+.5,
          new TM_Max(new TM_AboveBox(world, object, from, -negMargin), true), //this is the max selection -- only one of the four numbers need to be outside the BB
          OT_ineq, {}, 3e0); //NOTE: usually this is an inequality constraint <0; here we say this should be zero for a negative margin (->outside support)
}

void KOMO_ext::setInertialMotion(double startTime, double endTime, const char *object, const char *base, double g, double c) {
  addSwitch(startTime, true, new KinematicSwitch(SW_actJoint, JT_trans3, base, object, world));
//  setFlag(time, new Flag(FT_gravityAcc, world[object]->ID, 0, true),+1); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
//  setFlag(startTime, new Flag(FL_noQControlCosts, world[object]->ID, 0, true), +2);
//  setFlag(endTime, new Flag(FL_noQControlCosts, world[object]->ID, 0, true, false), +1);
  if(k_order>=2) {
    NIY;
//    setTask(startTime, endTime, new TM_InertialMotion(world, object, g, c), OT_sos, {}, 1e1, 2);
  }
}

void KOMO_ext::setFreeGravity(double time, const char *object, const char *base) {
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_trans3, base, object, world));
  addFlag(time, new Flag(FL_gravityAcc, world[object]->ID, 0, true),+1); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
}

/// a standard pick up: lower-attached-lift; centered, from top
void KOMO_ext::setGrasp(double time, const char* endeffRef, const char* object, int verbose, double weightFromTop, double timeToLift) {
  if(verbose>0) cout <<"KOMO_setGrasp t=" <<time <<" endeff=" <<endeffRef <<" obj=" <<object <<endl;
  //  String& endeffRef = world.getFrameByName(graspRef)->body->inLinks.first()->from->shapes.first()->name;
  
  //-- position the hand & graspRef
  //hand upright
  //  setTask(time, time, new TM_Default(TMT_vec, world, endeffRef, Vector_z), OT_sos, {0.,0.,1.}, weightFromTop);
  
  //hand center at object center (could be replaced by touch)
//  setTask(time, time, new TM_Default(TMT_posDiff, world, endeffRef, NoVector, object, NoVector), OT_eq, NoArr, 3e1);

  //hand grip axis orthogonal to object length axis
//  setTask(time, time, new TM_Default(TMT_vecAlign, world, endeffRef, Vector_x, object, Vector_x), OT_sos, NoArr, 3e0);
  //hand grip axis orthogonal to object length axis
//  setTask(time, time, new TM_Default(TMT_vecAlign, world, endeffRef, Vector_y, object, Vector_x), OT_sos, {-1.}, 3e0);

  //hand touches object
//  Shape *endeffShape = world.getFrameByName(endeffRef)->body->shapes.first();
//  setTask(time, time, new TM_GJK(endeffShape, world.getFrameByName(object), false), OT_eq, NoArr, 3e1);

  //disconnect object from table
//  setKinematicSwitch(time, true, "delete", NULL, object);
  //connect graspRef with object
#if 0
  setKinematicSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_quatBall, endeffRef, object, world));
  setKinematicSwitch(time, true, new KinematicSwitch(SW_insertEffJoint, JT_trans3, NULL, object, world));
  setTask(time, time, new TM_InsideBox(world, endeffRef, NoVector, object), OT_ineq, NoArr, 1e1);
#else
//  addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_free, endeffRef, object, world));
  addSwitch_stable(time, -1., endeffRef, object);
  addObjective(time, time, new TM_InsideBox(world, endeffRef, NoVector, object), OT_ineq, NoArr, 1e1);
//  setTouch(time, time, endeffRef, object);
#endif
  
//  if(stepsPerPhase>2 && timeToLift>0.){ //velocities down and up
//    setTask(time-timeToLift, time-2.*timeToLift/3, new TM_Default(TMT_pos, world, endeffRef), OT_sos, {0.,0.,-.1}, 1e0, 1); //move down
//    setTask(time-timeToLift/3,  time+timeToLift/3, new TM_Default(TMT_pos, world, endeffRef), OT_sos, {0.,0.,0.}, 3e0, 1); //move down
//    setTask(time+2.*timeToLift/3, time+timeToLift, new TM_Default(TMT_pos, world, endeffRef), OT_sos, {0.,0.,.1}, 1e0, 1); // move up
//  }

//  setFlag(time, new Flag(FL_clear, world[object]->ID, 0, true));
//  setFlag(time, new Flag(FL_zeroQVel, world[object]->ID, 0, true));
//  setFlag(time, new Flag(FL_kinematic, world[object]->getUpwardLink()->ID, 0, true));
}

/// a standard pick up: lower-attached-lift; centered, from top
void KOMO_ext::setGraspStick(double time, const char* endeffRef, const char* object, int verbose, double weightFromTop, double timeToLift) {
  if(verbose>0) cout <<"KOMO_setGraspStick t=" <<time <<" endeff=" <<endeffRef <<" obj=" <<object <<endl;
  
  //disconnect object from table
//  setKinematicSwitch(time, true, "delete", NULL, object);

  //connect graspRef with object
  addSwitch(time, true, "ballZero", endeffRef, object);
  addSwitch(time, true, "insert_transX", NULL, object);
//  setTask(time, time,
//          new TM_LinTrans(
//              new TM_Default(TMT_posDiff, world, endeffRef, NoVector, object, NoVector),
//              arr(2,3,{0,1,0,0,0,1}), {}),
//          OT_eq, NoArr, 3e1);
  addObjective(time, time, new TM_InsideBox(world, endeffRef, NoVector, object), OT_ineq, NoArr, 1e1);
  
  if(stepsPerPhase>2) { //velocities down and up
    addObjective(time-timeToLift, time, new TM_Default(TMT_pos, world, endeffRef), OT_sos, {0.,0.,-.1}, 3e0, 1); //move down
    addObjective(time, time+timeToLift, new TM_Default(TMT_pos, world, object), OT_sos, {0.,0.,.1}, 3e0, 1); // move up
  }
}

/// standard place on a table
void KOMO_ext::setPlace(double time, const char* endeff, const char* object, const char* placeRef, int verbose) {
  if(verbose>0) cout <<"KOMO_setPlace t=" <<time <<" obj=" <<object <<" place=" <<placeRef <<endl;
  
//  if(stepsPerPhase>2){ //velocities down and up
//    if(endeff){
//      setTask(time-.15, time-.10, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,-.1}, 3e0, 1); //move down
//      setTask(time-.05, time+.05, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,0. }, 1e1, 1); //hold still
//      setTask(time+.10, time+.15, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,+.1}, 3e0, 1); //move up
//    }else{
////      setTask(time-.15, time, new TM_Default(TMT_pos, world, object), OT_sos, {0.,0.,-.1}, 3e0, 1); //move down
//    }
//  }

  //place upright
//  setTask(time-.02, time, new TM_Default(TMT_vec, world, object, Vector_z), OT_sos, {0.,0.,1.}, 1e1);

  //place inside box support
//  setTask(time, time, new TM_StaticStability(world, placeRef, .01), OT_ineq);
  addObjective(time, time, new TM_AboveBox(world, object, placeRef), OT_ineq, NoArr, 1e1);
  
  //connect object to placeRef
#if 0
  Transformation rel = 0;
  rel.pos.set(0,0, .5*(shapeSize(world, object) + shapeSize(world, placeRef)));
//  setKinematicSwitch(time, true, "transXYPhiZero", placeRef, object, rel );
  addSwitch(time, true, new KinematicSwitch(SW_effJoint, JT_transXYPhi, placeRef, object, world, SWInit_zero, 0, rel));
  
  addFlag(time, new Flag(FL_clear, world[object]->ID, 0, true));
  addFlag(time, new Flag(FL_zeroQVel, world[object]->ID, 0, true));
#else
  addSwitch_stableOn(time, -1., placeRef, object);
#endif
}

/// place with a specific relative pose -> no effective DOFs!
void KOMO_ext::setPlaceFixed(double time, const char* endeff, const char* object, const char* placeRef, const Transformation& relPose, int verbose) {
  if(verbose>0) cout <<"KOMO_setPlace t=" <<time <<" endeff=" <<endeff <<" obj=" <<object <<" place=" <<placeRef <<endl;
  
  //connect object to table
  addSwitch(time, true, "rigidZero", placeRef, object, relPose);
  
  if(stepsPerPhase>2) { //velocities down and up
    if(endeff) {
      addObjective(time-.15, time-.10, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,-.1}, 3e0, 1); //move down
      addObjective(time-.05, time+.05, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,0. }, 1e1, 1); //hold still
      addObjective(time+.10, time+.15, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,+.1}, 3e0, 1); //move up
    }
  }
}

/// switch attachemend (-> ball eDOF)
void KOMO_ext::setHandover(double time, const char* oldHolder, const char* object, const char* newHolder, int verbose) {
#if 1
  setGrasp(time, newHolder, object, verbose, -1., -1.);
#else
  if(verbose>0) cout <<"KOMO_setHandover t=" <<time <<" oldHolder=" <<oldHolder <<" obj=" <<object <<" newHolder=" <<newHolder <<endl;
  
  //hand center at object center (could be replaced by touch)
  
  //disconnect object from table
//  setKinematicSwitch(time, true, "delete", oldHolder, object);
  //connect graspRef with object
#if 0
  setKinematicSwitch(time, true, "ballZero", newHolder, object); //why does this sometimes lead to worse motions?
#else
  setKinematicSwitch(time, true, "freeZero", newHolder, object);
  setTask(time, time, new TM_Default(TMT_posDiff, world, newHolder, NoVector, object, NoVector), OT_eq, NoArr, 3e1);
#endif
  
  if(stepsPerPhase>2) { //velocities: no motion
    setTask(time-.15, time+.15, new TM_Default(TMT_pos, world, object), OT_sos, {0.,0.,0.}, 3e0, 1); // no motion
  }
#endif
}

void KOMO::setPush(double startTime, double endTime, const char* stick, const char* object, const char* table, int verbose) {
  if(verbose>0) cout <<"KOMO_setPush t=" <<startTime <<" stick=" <<stick <<" object=" <<object <<" table=" <<table <<endl;
  
#if 1
  //stick normal alignes with slider direction
  addObjective(startTime, endTime, new TM_Default(TMT_vecAlign, world, stick, -Vector_y, "slider1b", Vector_x), OT_sos, {1.}, 1e1);
  //stick horizontal is orthogonal to world vertical
//  setTask(startTime, endTime, new TM_Default(TMT_vecAlign, world, stick, Vector_x, NULL, Vector_z), OT_sos, {0.}, 1e1);
  add_touch(startTime, endTime, stick, table);
  
  double dist = .05; //.5*shapeSize(world, object, 0)+.01;
  addObjective(startTime, endTime, new TM_InsideBox(world, "slider1b", {dist, .0, .0}, stick), OT_ineq);
//  setTask(startTime, endTime, new TM_Default(TMT_posDiff, world, stick, NoVector, "slider1b", {dist, .0, .0}), OT_sos, {}, 1e1);
#else
  setTouch(startTime, endTime, stick, object);
#endif

  setKS_slider(startTime, endTime, true, object, "slider1", table);
  
  addObjective(startTime, endTime-.1, new TM_AboveBox(world, object, table), OT_ineq, NoArr, 1e1);
  
#if 0
  //connect object to placeRef
  Transformation rel = 0;
  rel.pos.set(0,0, .5*(shapeSize(world, object) + shapeSize(world, table)));
  addSwitch(endTime, true, "transXYPhiZero", table, object, rel);
//  auto *o = addObjective(startTime, endTime, new TM_ZeroQVel(world, object), OT_eq, NoArr, 3e1, 1, +1);
//  o->prec(-1)=o->prec(-2)=0.;
#endif
  
  if(stepsPerPhase>2) { //velocities down and up
    addObjective(startTime-.3, startTime-.1, new TM_Default(TMT_pos, world, stick), OT_sos, {0.,0., -.1}, 3e0, 1); //move down
    addObjective(startTime-.05, startTime-.0, new TM_Default(TMT_pos, world, stick), OT_sos, {0.,0., 0}, 3e0, 1); //hold still
    addObjective(endTime+.0, endTime+.05, new TM_Default(TMT_pos, world, stick), OT_sos, {0.,0., 0}, 3e0, 1); //hold still
    addObjective(endTime+.1, endTime+.3, new TM_Default(TMT_pos, world, stick), OT_sos, {0.,0., .1}, 3e0, 1); // move up
  }
}

void KOMO::setGraspSlide(double time, const char* endeff, const char* object, const char* placeRef, int verbose) {

  double startTime = time;
  double endTime = time+1.;
  
  if(verbose>0) cout <<"KOMO_setSlide t=" <<startTime <<" endeff=" <<endeff <<" obj=" <<object <<endl;
  
  //-- grasp part
  //hand upright
//  setTask(startTime, startTime, new TM_Default(TMT_vec, world, endeff, Vector_z), OT_sos, {0.,0.,1.}, 1e-2);

  //disconnect object from table
//  setKinematicSwitch(startTime, true, "delete", placeRef, object);
  //connect graspRef with object
//  setKinematicSwitch(startTime, true, "ballZero", endeff, object);
//  setKinematicSwitch(time, true, "insert_trans3", NULL, object);
//  setTask(time, time, new TM_InsideBox(world, endeff, NoVector, object), OT_ineq, NoArr, 1e1);

//  addSwitch_stable(startTime, endTime+1., endeff, object);
  addSwitch(startTime, true, new KinematicSwitch(SW_effJoint, JT_free, endeff, object, world));
  addObjective(time, endTime, new TM_ZeroQVel(world, object), OT_eq, NoArr, 3e1, 1, +1, -1);
  if(k_order>1) addObjective(time, time, new TM_LinAngVel(world, object), OT_eq, NoArr, 1e2, 2, 0);
  else addObjective(time, time, new TM_NoJumpFromParent(world, object), OT_eq, NoArr, 1e2, 1, 0, 0);

  add_touch(startTime, startTime, endeff, object);
  
  //-- place part
  //place inside box support
//  setTask(endTime, endTime, new TM_AboveBox(world, object, placeRef), OT_ineq, NoArr, 1e1);
  add_aboveBox(endTime, endTime, object, placeRef);
  
  //disconnect object from grasp ref
//  setKinematicSwitch(endTime, true, "delete", endeff, object);

  //connect object to table
//  Transformation rel = 0;
//  rel.pos.set(0,0, h);
//  setKinematicSwitch(endTime, true, "transXYPhiZero", placeRef, object, rel );

  //-- slide constraints!
  //keep height of object above table
  double h = .5*(shapeSize(world, object) + shapeSize(world, placeRef));
  addObjective(startTime, endTime,
          new TM_LinTrans(make_shared<TM_Default>(TMT_posDiff, world, object, NoVector, placeRef), ~ARR(0,0,1), ARR(0)),
          OT_sos, ARR(h), 1e1);
  //keep object vertial
  addObjective(startTime, endTime,
          new TM_Default(TMT_vecDiff, world, object, Vector_z, placeRef, Vector_z), OT_sos, {}, 1e1);
          
//  if(stepsPerPhase>2){ //velocities down and up
//    setTask(startTime-.15, startTime, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,-.1}, 3e0, 1); //move down
//    setTask(endTime, endTime+.15, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,.1}, 3e0, 1); // move up
//  }
}

void KOMO_ext::setSlideAlong(double time, const char* stick, const char* object, const char* wall, int verbose) {
  if(verbose>0) cout <<"KOMO_setSlideAlong t=" <<time <<" obj=" <<object<<" wall=" <<wall <<endl;
  
  double endTime = time+1.;
  
  //stick normal alignes with slider direction
  addObjective(time, time+1., new TM_Default(TMT_vecAlign, world, stick, -Vector_y, object, Vector_x), OT_sos, {1.}, 1e0);
  //stick horizontal is orthogonal to world vertical
  addObjective(time, time+1., new TM_Default(TMT_vecAlign, world, stick, Vector_x, NULL, Vector_z), OT_sos, {0.}, 1e1);
  
  double dist = .5*shapeSize(world, object, 0)+.01;
  addObjective(time, time+1., new TM_InsideBox(world, object, {dist, .0, .0}, stick), OT_ineq);
  
  add_touch(time, time+1., stick, wall);
  
  //    //disconnect object from table
  //    setKinematicSwitch(time, true, "delete", NULL, object);
  //    //connect graspRef with object
  //    setKinematicSwitch(startTime, true, "ballZero", endeff, object);
  
  Transformation rel = 0;
  rel.rot.setDeg(-90, {1, 0, 0});
  rel.pos.set(0, -.5*(shapeSize(world, wall, 1) - shapeSize(world, object)), +.5*(shapeSize(world, wall, 2) + shapeSize(world, object, 1)));
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transX, wall, object, world));
  addSwitch(time, true, new KinematicSwitch(SW_insertEffJoint, JT_transZ, NULL, object, world, SWInit_zero, 0, rel));
  //    setKinematicSwitch(time, true, "insert_trans3", NULL, object);
  //    setTask(time, time, new TM_InsideBox(world, endeff, NoVector, object), OT_ineq, NoArr, 1e1);
  
  if(stepsPerPhase>2) { //velocities down and up
    addObjective(endTime+.0, endTime+.05, new TM_Default(TMT_pos, world, stick), OT_sos, {0.,0., 0}, 3e0, 1); //hold still
    addObjective(endTime+.1, endTime+.3, new TM_Default(TMT_pos, world, stick), OT_sos, {0.,0., .05}, 3e0, 1); // move up
  }
}

void KOMO_ext::setDrop(double time, const char* object, const char* from, const char* to, int verbose) {

  if(from) { //require the object outside the margin of its bounding box
    setOverTheEdge(time, object, from, .05);
  }
  
  //disconnect object from anything
//  setKinematicSwitch(time, true, "delete", NULL, object);

  //connect to world with lift
//  setKinematicSwitch(time, true, "JT_trans3", "world", object);
  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_transZ, to, object, world));
  addSwitch(time, true, new KinematicSwitch(SW_insertEffJoint, JT_transXY, NULL, object, world));
  
  addFlag(time, new Flag(FL_xPosAccCosts, world[object]->ID, 0, true)); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
//  setFlag(time, new Flag(FT_gravityAcc, world[object]->ID, 0, true),+1); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
//  setFlag(time, new Flag(FT_noQControlCosts, world[object]->ID, 0, true),+1);

//  if(stepsPerPhase>2){ //velocities down and up
//    setTask(time, time+.2, new TM_Default(TMT_pos, world, object), OT_sos, {0.,0.,-.1}, 3e0, 1); // move down
//  }
}

void KOMO_ext::setDropEdgeFixed(double time, const char* object, const char* to, const Transformation &relFrom, const Transformation &relTo, int verbose) {

  //disconnect object from anything
//  setKinematicSwitch(time, true, "delete", NULL, object);

  //connect to world with lift
//  setKinematicSwitch(time, true, "JT_trans3", "world", object);

  addSwitch(time, true, new KinematicSwitch(SW_actJoint, JT_hingeX, to, object, world, SWInit_zero, 0, relFrom, relTo));
//  setKinematicSwitch(time, true, new KinematicSwitch(insertActuated, JT_transZ, NULL, object, world, 0));
//  setKinematicSwitch(time, true, new KinematicSwitch(SW_insertEffJoint, JT_trans3, NULL, object, world, 0));

//  if(stepsPerPhase>2){ //velocities down and up
//    setTask(time, time+.2, new TM_Default(TMT_pos, world, object), OT_sos, {0.,0.,-.1}, 3e0, 1); // move down
//  }
}

void KOMO_ext::setAttach(double time, const char* endeff, const char* object1, const char* object2, Transformation& rel, int verbose) {
  if(verbose>0) cout <<"KOMO_setAttach t=" <<time <<" endeff=" <<endeff <<" obj1=" <<object1 <<" obj2=" <<object2 <<endl;
  
  //hand center at object center (could be replaced by touch)
//  setTask(time, time, new TM_Default(TMT_pos, world, object2, NoVector, object1, NoVector), OT_sos, rel.pos.getArr(), 3e1);
//  setTask(time, time, new TM_Default(TMT_quatDiff, world, object2, NoVector, object1, NoVector), OT_sos, conv_quat2arr(rel.rot), 3e1);

//  setTask(time, time, new TM_Default(TMT_vecAlign, world, newHolder, Vector_y, object, Vector_x), OT_sos, {-1.}, 3e0);

  //disconnect object from grasp ref
//  setKinematicSwitch(time, true, "delete", endeff, object2);

//  Transformation rel = 0;
//  rel.addRelativeTranslation( 0., 0., .5*(shapeSize(world.getFrameByName(object)) + shapeSize(world.getFrameByName(placeRef))));
  addSwitch(time, true, "rigidZero", object1, object2, rel);
  
}

void KOMO::setSlow(double startTime, double endTime, double prec, bool hardConstrained) {
  if(stepsPerPhase>2) { //otherwise: no velocities
#if 1
    uintA selectedBodies;
    arr scale;
    for(rai::Frame *f:world.frames) if(f->joint && f->joint->dim>0 && f->joint->dim<7 && f->joint->type!=rai::JT_time && f->joint->active && f->joint->H>0.){
      selectedBodies.append(TUP(f->ID, f->parent->ID));
    }
    selectedBodies.reshape(selectedBodies.N/2,2);
    ptr<Feature> map = make_shared<TM_qItself>(selectedBodies);
#else
    Feature *map = new TM_qItself;
#endif
    if(!hardConstrained) addObjective(startTime, endTime, map, OT_sos, NoArr, prec, 1);
    else addObjective(startTime, endTime, map, OT_eq, NoArr, prec, 1);
  }
  //#    _MinSumOfSqr_qItself_vel(MinSumOfSqr qItself){ order=1 time=[0.98 1] scale=3e0 } #slow down
}

void KOMO::setSlowAround(double time, double delta, double prec, bool hardConstrained) {
  setSlow(time-delta, time+delta, prec, hardConstrained);
}

void KOMO_ext::setFine_grasp(double time, const char* endeff, const char* object, double above, double gripSize, const char* gripper, const char* gripper2) {
  double t1=-.25; //time when gripper is positined above
  double t2=-.1;  //time when gripper is lowered
  double t3=-.05; //time when gripper is closed
  
  //position above
  addObjective(time+t1, 1., new TM_Default(TMT_vec, world, endeff, Vector_z), OT_sos, {0.,0.,1.}, 1e0);
  addObjective(time+t1, t1, new TM_Default(TMT_posDiff, world, endeff, NoVector, object, NoVector), OT_sos, {0.,0.,above+.1}, 3e1);
  addObjective(time+t1, 1., new TM_Default(TMT_vecAlign, world, endeff, Vector_x, object, Vector_y), OT_sos, NoArr, 3e0);
  addObjective(time+t1, 1., new TM_Default(TMT_vecAlign, world, endeff, Vector_x, object, Vector_z), OT_sos, NoArr, 3e0);
  //open gripper
  if(gripper)  addObjective(time+t1, .85, new TM_qItself(QIP_byJointNames, {gripper}, world), OT_sos, {gripSize + .05});
  if(gripper2) addObjective(time+t1, .85, new TM_qItself(QIP_byJointNames, {gripper2}, world), OT_sos, {::asin((gripSize + .05)/(2.*.10))});
  //lower
  addObjective(time+t2, 1., new TM_Default(TMT_posDiff, world, endeff, NoVector, object, NoVector), OT_sos, {0.,0.,above}, 3e1);
  //close gripper
  if(gripper)  addObjective(time+t3, 1., new TM_qItself(QIP_byJointNames, {gripper}, world), OT_sos, {gripSize});
  if(gripper2) addObjective(time+t3, 1., new TM_qItself(QIP_byJointNames, {gripper2}, world), OT_sos, {::asin((gripSize)/(2.*.10))});
  setSlowAround(time, .05, 3e1);
}

/// translate a list of facts (typically facts in a FOL state) to LGP tasks
void KOMO_ext::setAbstractTask(double phase, const Graph& facts, int verbose) {
//  CHECK_LE(phase, maxPhase,"");
//  listWrite(facts, cout,"\n");  cout <<endl;
  for(Node *n:facts) {
    if(!n->parents.N) continue;
    StringL symbols;
    for(Node *p:n->parents) symbols.append(&p->keys.last());
    double time=1.; //n->get<double>(); //komo tag needs to be double valued!
    if(n->keys.N==1 && n->keys.scalar() == "komo") {
      if(*symbols(0)=="grasp")                      setGrasp(phase+time, *symbols(1), *symbols(2), verbose);
      else if(*symbols(0)=="push")                  setPush(phase+time, phase+time+1., *symbols(1), *symbols(2), *symbols(3), verbose); //TODO: the +1. assumes pushes always have duration 1
      else if(*symbols(0)=="place" && symbols.N==3) setPlace(phase+time, NULL, *symbols(1), *symbols(2), verbose);
      else if(*symbols(0)=="place" && symbols.N==4) setPlace(phase+time, *symbols(1), *symbols(2), *symbols(3), verbose);
      else if(*symbols(0)=="graspSlide")            setGraspSlide(phase+time, *symbols(1), *symbols(2), *symbols(3), verbose);
      else if(*symbols(0)=="handover")              setHandover(phase+time, *symbols(1), *symbols(2), *symbols(3), verbose);
      
      //elementary
      else if(*symbols(0)=="flagClear")  {} //           setFlag(phase+time, new Flag(FL_clear, world[*symbols(1)]->ID, 0, true));
      else if(*symbols(0)=="touch")                 add_touch(phase+time, phase+time, *symbols(1), *symbols(2));
      else if(*symbols(0)=="lift")                  setLiftDownUp(phase+time, *symbols(1));
      else if(*symbols(0)=="impulse")               add_impulse(phase+time, *symbols(1), *symbols(2));
      else if(*symbols(0)=="inside")                add_insideBox(phase+time, phase+time, *symbols(1), *symbols(2));
      else if(*symbols(0)=="above")                 add_aboveBox(phase+time, phase+time+.9, *symbols(1), *symbols(2));
      else if(*symbols(0)=="stable")                addSwitch_stable(phase+time, phase+time+1., *symbols(1), *symbols(2));
      else if(*symbols(0)=="stableOn")              addSwitch_stableOn(phase+time, phase+time+1., *symbols(1), *symbols(2));
      else if(*symbols(0)=="dynamic")               addSwitch_dynamic(phase+time, phase+time+1., "base", *symbols(1));
      else if(*symbols(0)=="dynamicTrans")          addSwitch_dynamicTrans(phase+time, phase+time+1., "base", *symbols(1));
      else if(*symbols(0)=="dynamicOn")             addSwitch_dynamicOn(phase+time, phase+time+1., *symbols(1), *symbols(2));
      
      else if(*symbols(0)=="notAbove") {
        double margin = .05;
        double negMargin = margin + .5*shapeSize(world, *symbols(1), 0); //how much outside the bounding box?
        addObjective(phase+time, phase+time+.9,
                new TM_Max(new TM_AboveBox(world, *symbols(1),*symbols(2), -negMargin), true), //this is the max selection -- only one of the four numbers need to be outside the BB
                OT_ineq, {}, 3e0); //NOTE: usually this is an inequality constraint <0; here we say this should be zero for a negative margin (->outside support)
      } else if(*symbols(0)=="fricSlide") {
        Transformation rel = 0;
        rel.pos.set(0,0, .5*(shapeSize(world, *symbols(1)) + shapeSize(world, *symbols(2))));
        addSwitch(phase+time, false, new KinematicSwitch(SW_actJoint, JT_transXYPhi, *symbols(2), *symbols(1), world, SWInit_zero, 0, rel));
        addFlag(phase+time, new Flag(FL_clear, world[*symbols(1)]->ID, 0, true), +1);
        addFlag(phase+time, new Flag(FL_impulseExchange, world[*symbols(1)]->ID), 0);
        addFlag(phase+time, new Flag(FL_xPosVelCosts, world[*symbols(1)]->ID, 0, true), +1);
      } else if(*symbols(0)=="dynVert")               {
        addSwitch(phase+time, true, new KinematicSwitch(SW_actJoint, JT_transZ, *symbols(2), *symbols(1), world));
        addSwitch(phase+time, true, new KinematicSwitch(SW_insertEffJoint, JT_transXY, NULL, *symbols(1), world));
        addFlag(phase+time, new Flag(FL_clear, world[*symbols(1)]->ID, 0, true), +1);
//        setFlag(phase+time, new Flag(FL_zeroAcc, world[*symbols(1)]->ID, 0, true), +1);
        addFlag(phase+time, new Flag(FL_xPosAccCosts, world[*symbols(1)]->ID, 0, true), +1); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
        
      } else HALT("UNKNOWN komo symbol: '" <<*symbols(0) <<"'");
    } else if(n->keys.N && n->keys.last().startsWith("komo")) {
      if(n->keys.last()=="komoSlideAlong") setSlideAlong(phase+time, *symbols(0), *symbols(1), *symbols(2), verbose);
      else if(n->keys.last()=="komoDrop") {
        if(symbols.N==2) setDrop(phase+time, *symbols(0), NULL, *symbols(1), verbose);
        else setDrop(phase+time, *symbols(0), *symbols(1), *symbols(2), verbose);
      } else if(n->keys.last()=="komoThrow") {
//        setInertialMotion(phase+time, phase+time+1., *symbols(0), "base", -.1, 0.);
        addSwitch(phase+time, true, new KinematicSwitch(SW_actJoint, JT_trans3, "base", *symbols(0), world));
        addFlag(phase+time, new Flag(FL_gravityAcc, world[*symbols(0)]->ID, 0, true), +1); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
      } else if(n->keys.last()=="komoHit") {
        setImpact(phase+time, *symbols(0), *symbols(1));
        if(symbols.N==2) {
          addSwitch(phase+time, true, new KinematicSwitch(SW_actJoint, JT_trans3, "base", *symbols(1), world));
          addFlag(phase+time, new Flag(FL_gravityAcc, world[*symbols(1)]->ID, 0, true), +1); //why +1: the kinematic switch triggers 'FixSwitchedObjects' to enforce acc 0 for time slide +0
        } else if(symbols.N==3) {
          //const char* bat = *symbols(0);
          const char* object = *symbols(1);
          const char* placeRef = *symbols(2);
          Transformation rel = 0;
          rel.pos.set(0,0, .5*(shapeSize(world, object) + shapeSize(world, placeRef)));
          
          addSwitch(phase+time, true, new KinematicSwitch(SW_actJoint, JT_transXYPhi, placeRef, object, world, SWInit_zero, 0, rel));
          addFlag(phase+time, new Flag(FL_clear, world[object]->ID, 0, true));
          addFlag(phase+time, new Flag(FL_zeroAcc, world[object]->ID, 0, true));
          
//          setKinematicSwitch(phase+time, false, new KinematicSwitch(SW_actJoint, JT_transXYPhi, placeRef, bat, world, 0, rel));
//          setFlag(phase+time, new Flag(FL_clear, world[bat]->ID, 0, true), +1);
//          setFlag(phase+time, new Flag(FL_xPosVelCosts, world[bat]->ID, 0, true), +1);
        } else NIY;
      } else if(n->keys.last()=="komoAttach") {
        Node *attachableSymbol = facts["attachable"];
        CHECK(attachableSymbol!=NULL,"");
        Node *attachableFact = facts.getEdge({attachableSymbol, n->parents(1), n->parents(2)});
        Transformation rel = attachableFact->get<Transformation>();
        setAttach(phase+time, *symbols(0), *symbols(1), *symbols(2), rel, verbose);
      } else HALT("UNKNOWN komo TAG: '" <<n->keys.last() <<"'");
    }
  }
}

void KOMO::setSkeleton(const Skeleton &S, bool ignoreSwitches) {
  //-- add objectives for mode switches
  intA switches = getSwitchesFromSkeleton(S);
  if(!ignoreSwitches){
    for(uint i=0;i<switches.d0;i++) {
      int j = switches(i,0);
      int k = switches(i,1);
      const char* newFrom=world.frames.first()->name;
      if(S(k).frames.N==2) newFrom = S(k).frames(0);
      if(j<0)
        addSwitch_mode(SY_initial, S(k).symbol, S(k).phase0, S(k).phase1+1., NULL, newFrom, S(k).frames.last());
      else
        addSwitch_mode(S(j).symbol, S(k).symbol, S(k).phase0, S(k).phase1+1., S(j).frames(0), newFrom, S(k).frames.last());
    }
  }
  //-- add objectives for rest
  for(const SkeletonEntry& s:S) {
    switch(s.symbol){
      case SY_none:       HALT("should not be here");  break;
      case SY_initial: case SY_identical: case SY_noCollision:    break;
      case SY_touch:      add_touch(s.phase0, s.phase1, s.frames(0), s.frames(1));  break;
      case SY_above:      add_aboveBox(s.phase0, s.phase1, s.frames(0), s.frames(1));  break;
      case SY_inside:     add_aboveBox(s.phase0, s.phase1, s.frames(0), s.frames(1));  break;
      case SY_impulse:    add_impulse(s.phase0, s.frames(0), s.frames(1));  break;

      case SY_makeFree:   world.makeObjectsFree(s.frames);  break;
      case SY_stableRelPose: addObjective({s.phase0, s.phase1+1.}, OT_eq, FS_poseRel, s.frames, {1e2}, {}, 1);  break;
      case SY_stablePose:  addObjective({s.phase0, s.phase1+1.}, OT_eq, FS_pose, s.frames, {1e2}, {}, 1);  break;


      case SY_liftDownUp: setLiftDownUp(s.phase0, s.frames(0), .4);  break;
      case SY_break:      addObjective(s.phase0, s.phase1, make_shared<TM_NoJumpFromParent>(world, s.frames(0)), OT_eq, NoArr, 1e2, 1, 0, 0);  break;

      case SY_contact:    addContact_slide(s.phase0, s.phase1, s.frames(0), s.frames(1));  break;
      case SY_bounce:     addContact_elasticBounce(s.phase0, s.frames(0), s.frames(1), .9);  break;
        //case SY_contactComplementary:     addContact_Complementary(s.phase0, s.phase1, s.frames(0), s.frames(1));  break;


      case SY_alignByInt: {
        addObjective({s.phase0, s.phase1}, OT_sos, FS_scalarProductXX, s.frames);  break;
        cout <<"THE INTEGER IS: " <<s.frames(2) <<endl;
      } break;

      case SY_push:       setPush(s.phase0, s.phase1+1., s.frames(0), s.frames(1), s.frames(2), verbose);  break;//TODO: the +1. assumes pushes always have duration 1
      case SY_graspSlide: setGraspSlide(s.phase0, s.frames(0), s.frames(1), s.frames(2), verbose);  break;
        //    else case SY_handover)              setHandover(s.phase0, s.frames(0), s.frames(1), s.frames(2), verbose);
        //    else LOG(-2) <<"UNKNOWN PREDICATE!: " <<s;

      //switches are handled above now
      case SY_stable:      //if(!ignoreSwitches) addSwitch_stable(s.phase0, s.phase1+1., s.frames(0), s.frames(1));  break;
      case SY_stableOn:    //if(!ignoreSwitches) addSwitch_stableOn(s.phase0, s.phase1+1., s.frames(0), s.frames(1));  break;
      case SY_dynamic:     //if(!ignoreSwitches) addSwitch_dynamic(s.phase0, s.phase1+1., "base", s.frames(0));  break;
      case SY_dynamicOn:   //if(!ignoreSwitches) addSwitch_dynamicOn(s.phase0, s.phase1+1., s.frames(0), s.frames(1));  break;
      case SY_dynamicTrans:   //if(!ignoreSwitches) addSwitch_dynamicTrans(s.phase0, s.phase1+1., "base", s.frames(0));  break;
        break;
      case SY_magic:      addSwitch_magic(s.phase0, s.phase1, world.frames.first()->name, s.frames(0), 0.);  break;
      case SY_magicTrans: addSwitch_magicTrans(s.phase0, s.phase1, world.frames.first()->name, s.frames(0), 0.);  break;
    }
  }
}

void KOMO_ext::setAlign(double startTime, double endTime, const char* shape, const arr& whichAxis, const char* shapeRel, const arr& whichAxisRel, ObjectiveType type, const arr& target, double prec) {
#if 0
  String map;
  map <<"map=vecAlign ref1="<<shape;
  if(whichAxis) map <<" vec1=[" <<whichAxis <<']';
  if(shapeRel) map <<" ref2=" <<shapeRel <<" vec2=" <<;
  if(whichAxisRel) map <<" vec2=[" <<whichAxisRel <<']';
  setTask(startTime, endTime, map, type, target, prec);
#else
  addObjective(startTime, endTime, new TM_Default(TMT_vecAlign, world, shape, Vector(whichAxis), shapeRel, Vector(whichAxisRel)), type, target, prec);
#endif
  
}

void KOMO::add_touch(double startTime, double endTime, const char* shape1, const char* shape2, ObjectiveType type, const arr& target, double prec) {
  addObjective(startTime, endTime, new TM_PairCollision(world, shape1, shape2, TM_PairCollision::_negScalar, false), type, target, prec);
}

void KOMO::add_aboveBox(double startTime, double endTime, const char* shape1, const char* shape2, double prec) {
  addObjective(startTime, endTime, new TM_AboveBox(world, shape1, shape2), OT_ineq, NoArr, prec);
}

void KOMO::add_insideBox(double startTime, double endTime, const char* shape1, const char* shape2, double prec) {
  addObjective(startTime, endTime, new TM_InsideBox(world, shape1, NoVector, shape2), OT_ineq, NoArr, prec);
}

void KOMO::add_impulse(double time, const char* shape1, const char* shape2, ObjectiveType type, double prec) {
//    setTask(time, time, new TM_ImpulsExchange(world, a, b), OT_sos, {}, 3e1, 2, +1); //+1 deltaStep indicates moved 1 time slot backward (to cover switch)
  if(k_order>=2) {
    addObjective(time, time, new TM_ImpulsExchange(world, shape1, shape2), type, {}, prec, 2, +1, +1); //+1 deltaStep indicates moved 1 time slot backward (to cover switch)
    addFlag(time, new Flag(FL_impulseExchange, world[shape1]->ID), +0);
    addFlag(time, new Flag(FL_impulseExchange, world[shape2]->ID), +0);
  }
}

void KOMO::add_stable(double time, const char* shape1, const char* shape2, ObjectiveType type, double prec){
  addObjective(time, time, new TM_Default(TMT_pose, world, shape1, NoVector, shape2), type, NoArr, prec, 1, 0 );
}

void KOMO_ext::setAlignedStacking(double time, const char* object, ObjectiveType type, double prec) {
  addObjective(time, time, new TM_AlignStacking(world, object), type, NoArr, prec);
}

void KOMO::add_collision(bool hardConstraint, double margin, double prec) {
  if(hardConstraint) { //interpreted as hard constraint (default)
    addObjective(0., -1., new TM_Proxy(TMT_allP, {}, margin), OT_eq, NoArr, prec);
  } else { //cost term
    addObjective(0., -1., new TM_Proxy(TMT_allP, {}, margin), OT_sos, NoArr, prec);
  }
}

void KOMO::add_jointLimits(bool hardConstraint, double margin, double prec) {
  if(hardConstraint) { //interpreted as hard constraint (default)
    addObjective(0., -1., new LimitsConstraint(margin), OT_ineq, NoArr, prec);
  } else { //cost term
    NIY;
//    setTask(0., -1., new TM_Proxy(TMT_allP, {}, margin), OT_sos, NoArr, prec);
  }
}

void KOMO::setLiftDownUp(double time, const char *endeff, double timeToLift) {
  if(stepsPerPhase>2 && timeToLift>0.) { //velocities down and up
    addObjective(time-timeToLift, time-.5*timeToLift, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,-.2}, 1e0, 1); //move down
//    addObjective(time-timeToLift/3,  time+timeToLift/3, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,0.}, 3e0, 1); //move down
    addObjective(time+.5*timeToLift, time+timeToLift, new TM_Default(TMT_pos, world, endeff), OT_sos, {0.,0.,.2}, 1e0, 1); // move up
  }
}

//===========================================================================
//
// config
//

void KOMO::setConfigFromFile() {
  KinematicWorld K(getParameter<String>("KOMO/modelfile"));
//  K.optimizeTree();
  setModel(
    K,
    getParameter<bool>("KOMO/useSwift", true)
  );
  setTiming(
    getParameter<uint>("KOMO/phases"),
    getParameter<uint>("KOMO/stepsPerPhase"),
    getParameter<double>("KOMO/durationPerPhase", 5.),
    getParameter<uint>("KOMO/k_order", 2)
  );
}

void KOMO::setIKOpt() {
  denseOptimization=true;
  stepsPerPhase = 1;
  T = 1;
  tau = 1.;
  k_order = 1;
  setSquaredQVelocities(0.,-1.,1e-1);
  setSquaredQuaternionNorms();
}

void KOMO::setDiscreteOpt(uint k){
  denseOptimization=true;
  stepsPerPhase = 1;
  T = k;
  tau = 1.;
  k_order = 1;
  setSquaredQuaternionNorms();
}

void KOMO::setPoseOpt() {
  denseOptimization=true;
  setTiming(1., 2, 5., 1);
  setSquaredQuaternionNorms();
}

void KOMO::setSequenceOpt(double _phases) {
  setTiming(_phases, 2, 5., 1);
  setSquaredQuaternionNorms();
}

void KOMO::setPathOpt(double _phases, uint stepsPerPhase, double timePerPhase) {
  setTiming(_phases, stepsPerPhase, timePerPhase, 2);
  setSquaredQuaternionNorms();
}

void setTasks(KOMO_ext& MP,
              Frame& endeff,
              Frame& target,
              byte whichAxesToAlign,
              uint iterate,
              int timeSteps,
              double duration) {
              
#if 1
  HALT("deprecated");
#else
  //-- parameters
  double posPrec = getParameter<double>("KOMO/moveTo/precision", 3e1);
  double colPrec = getParameter<double>("KOMO/moveTo/collisionPrecision", -1e0);
  double margin = getParameter<double>("KOMO/moveTo/collisionMargin", .1);
  double zeroVelPrec = getParameter<double>("KOMO/moveTo/finalVelocityZeroPrecision", 3e0);
  double alignPrec = getParameter<double>("KOMO/moveTo/alignPrecision", 3e1);
  
  //-- set up the KOMO
  target.shape->cont=false; //turn off contact penalization with the target
  
//  MP.world.swift().initActivations(MP.world);
  //MP.world.watch(false);
  
  MP.setTiming(1., getParameter<uint>("timeSteps", 50), getParameter<double>("duration", 5.));
  if(timeSteps>=0) MP.setTiming(1., timeSteps, duration);
  if(timeSteps==0) MP.k_order=1;
  
  Task *t;
  
  t = MP.addTask("transitions", new TM_Transition(MP.world), OT_sos);
  if(timeSteps!=0) {
    t->map->order=2; //make this an acceleration task!
  } else {
    t->map->order=1; //make this a velocity task!
  }
  t->setCostSpecs(0, MP.T-1, {0.}, 1e0);
  
  if(timeSteps!=0) {
    t = MP.addTask("final_vel", new TM_qItself(), OT_sos);
    t->map->order=1; //make this a velocity task!
    t->setCostSpecs(MP.T-4, MP.T-1, {0.}, zeroVelPrec);
  }
  
  if(colPrec<0) { //interpreted as hard constraint (default)
    t = MP.addTask("collisionConstraints", new CollisionConstraint(margin), OT_ineq);
    t->setCostSpecs(0, MP.T-1, {0.}, 1.);
  } else { //cost term
    t = MP.addTask("collision", new TM_Proxy(TMT_allP, {}, margin), OT_sos);
    t->setCostSpecs(0, MP.T-1, {0.}, colPrec);
  }
  
  t = MP.addTask("endeff_pos", new TM_Default(TMT_pos, endeff.ID, NoVector, target.ID, NoVector), OT_sos);
  t->setCostSpecs(MP.T-1, MP.T-1, {0.}, posPrec);
  
  for(uint i=0; i<3; i++) if(whichAxesToAlign&(1<<i)) {
      Vector axis;
      axis.setZero();
      axis(i)=1.;
      t = MP.addTask(STRING("endeff_align_"<<i),
                     new TM_Default(TMT_vecAlign, endeff.ID, axis, target.ID, axis),
                     OT_sos);
      t->setCostSpecs(MP.T-1, MP.T-1, {1.}, alignPrec);
    }
#endif
}

void KOMO_ext::setMoveTo(KinematicWorld& world, Frame& endeff, Frame& target, byte whichAxesToAlign) {
//  if(MP) delete MP;
//  MP = new KOMO(world);
  setModel(world);
  this->world.checkConsistency();
  
  setTasks(*this, endeff, target, whichAxesToAlign, 1, -1, -1.);
  reset();
}

void KOMO::setSpline(uint splineT) {
  Spline S;
  S.setUniformNonperiodicBasis(T-1, splineT, 2);
  uint n=dim_x(0);
  splineB = zeros(S.basis.d0*n, S.basis.d1*n);
  for(uint i=0; i<S.basis.d0; i++) for(uint j=0; j<S.basis.d1; j++)
      splineB.setMatrixBlock(S.basis(i,j)*eye(n,n), i*n, j*n);
  z = pseudoInverse(splineB) * x;
}

void KOMO::reset(double initNoise) {
  if(!configurations.N) setupConfigurations();
  x = getPath_decisionVariable();
  dual.clear();
  featureValues.clear();
  featureTypes.clear();
  komo_problem.clear();
  dense_problem.clear();
  if(initNoise>0.)
    rndGauss(x, initNoise, true); //don't initialize at a singular config
  if(splineB.N) {
    z = pseudoInverse(splineB) * x;
  }
}

void KOMO::initWithConstant(const arr& q){
  for(uint t=0;t<T;t++) {
    configurations(k_order+t)->setJointState(q);
  }

  reset(0.);
}

void KOMO::initWithWaypoints(const arrA& waypoints, uint waypointStepsPerPhase, bool sineProfile){
  //assume waypoints correspond to phase times 1,2,3... and get steps
  uintA steps(waypoints.N);
  for(uint i=0;i<steps.N;i++){
    steps(i) = conv_time2step(conv_step2time(i, waypointStepsPerPhase), stepsPerPhase);
  }

  //set the path piece-wise CONSTANT with waypoints (each waypoint may have different dimension!...)
  for(uint i=0;i<steps.N;i++) {
    uint Tstop=T;
    if(i+1<steps.N && steps(i+1)<T) Tstop=steps(i+1);
    for(uint t=steps(i); t<Tstop; t++)
      configurations(k_order+t)->setJointState(waypoints(i));
  }

  //interpolate
#if 1
  for(uint i=0;i<steps.N;i++) {
    uint i1=steps(i);
    uint i0=0; if(i) i0 = steps(i-1);
    //motion profile
    if(i1<T){
      for(uint j=i0+1;j<=i1;j++){
        uintA nonSwitched = getNonSwitchedBodies({configurations(k_order+j), configurations(k_order+i1)});
        arr q0 = configurations(k_order+j)->getJointState(nonSwitched);
        arr q1 = configurations(k_order+i1)->getJointState(nonSwitched);
        arr q;
        double phase = double(j-i0)/double(i1-i0);
        if(sineProfile){
          q = q0 + (.5*(1.-cos(RAI_PI*phase))) * (q1-q0);
        }else{
          q = q0 + phase * (q1-q0);
        }
        configurations(k_order+j)->setJointState(q, nonSwitched);
      }
    }/*else{
      for(uint j=i0+1;j<T;j++){
        configurations(k_order+j)->setJointState(q0, nonSwitched);
      }
    }*/
  }
#endif

  reset(0.);
}

void KOMO::run() {
  KinematicWorld::setJointStateCount=0;
  double timeZero = timerStart();
  CHECK(T,"");
  if(logFile) (*logFile) <<"KOMO_run_log: [" <<endl;
  if(opt) delete opt;
  if(denseOptimization){
    CHECK(!splineB.N, "NIY");
    OptConstrained _opt(x, dual, dense_problem, rai::MAX(verbose-2, 0));
//    OptPrimalDual _opt(x, dual, dense_problem, rai::MAX(verbose-2, 0));
    _opt.logFile = logFile;
    _opt.run();
    timeNewton += _opt.newton.timeNewton;
  } else if(sparseOptimization){
    CHECK(!splineB.N, "NIY");
#if 1
//    ModGraphProblem selG(graph_problem);
//    Conv_Graph_ConstrainedProblem C(selG);
    Conv_Graph_ConstrainedProblem C(graph_problem, logFile);
    OptConstrained _opt(x, dual, C, rai::MAX(verbose-2, 0), NOOPT, logFile);
//    OptPrimalDual _opt(x, dual, C, rai::MAX(verbose-2, 0));
    _opt.run();
    {
//      testing primal dual:
//      rai::wait();
//      OptPrimalDual _opt(x, dual, C, rai::MAX(verbose-2, 0));
//      _opt.run();
    }

    timeNewton += _opt.newton.timeNewton;
#else
    BacktrackingGraphOptimization BGO(graph_problem);
    BGO.evaluate(x);
    BGO.run();
#endif
  } else if(!splineB.N) { //DEFAULT CASE
    Convert C(komo_problem);
    opt = new OptConstrained(x, dual, C, rai::MAX(verbose-2, 0));
    opt->logFile = logFile;
    opt->run();
  } else {
    arr a,b,c,d,e;
    Conv_KOMO_ConstrainedProblem P0(komo_problem);
    Conv_linearlyReparameterize_ConstrainedProblem P(P0, splineB);
    opt = new OptConstrained(z, dual, P, rai::MAX(verbose-2, 0));
    opt->logFile = logFile;
    opt->run();
  }
  runTime = timerRead(true, timeZero);
  if(logFile) (*logFile) <<"\n] #end of KOMO_run_log" <<endl;
  if(verbose>0) {
    cout <<"** optimization time=" <<runTime
        <<" (kin:" <<timeKinematics <<" coll:" <<timeCollisions <<" feat:" <<timeFeatures <<" newton: " <<timeNewton <<")"
         <<" setJointStateCount=" <<KinematicWorld::setJointStateCount <<endl;
  }
  if(verbose>0) cout <<getReport(verbose>1) <<endl;
}

void KOMO::run_sub(const uintA& X, const uintA& Y) {
  KinematicWorld::setJointStateCount=0;
  double timeZero = timerStart();
  if(opt) delete opt;

  {
    GraphProblem_Structure Gstruct(graph_problem);
    //evaluate once with full parameters to adopt initialization
    {
      uintA X;
      X.setStraightPerm(Gstruct.V.N);
      SubGraphProblem G_X(Gstruct, X, {});
      G_X.phi(NoArr, NoArrA, NoArrA, x);
    }

    SubGraphProblem G_XY(Gstruct, X, Y);
    G_XY.optim( rai::MAX(verbose-2, 0) );
    sos = G_XY.sos; eq = G_XY.eq; ineq = G_XY.ineq;
  }

  runTime = timerRead(true, timeZero);
  if(verbose>0) {
    cout <<"** optimization time=" <<runTime
        <<" (kin:" <<timeKinematics <<" coll:" <<timeCollisions <<" feat:" <<timeFeatures <<" newton: " <<timeNewton <<")"
       <<" setJointStateCount=" <<KinematicWorld::setJointStateCount <<endl;
  }
  if(verbose>0) cout <<getReport(verbose>1) <<endl;
}

void KOMO::optimize(bool initialize){
  if(initialize) reset();
  CHECK_EQ(configurations.N, T+k_order, "");

  if(verbose>0) reportProblem();

  run();

//  if(verbose>0){
//    Graph specs = getProblemGraph(true);
//    cout <<specs <<endl;
//    cout <<getReport(verbose>1) <<endl; // Enables plot
//  }
}

void KOMO_ext::getPhysicsReference(uint subSteps, int display) {
  x.resize(T, world.getJointStateDimension());
  PhysXInterface& px = world.physx();
  px.pushFullState(world.frames);
  for(uint t=0; t<T; t++) {
    for(uint s=0; s<subSteps; s++) {
      px.step(tau/subSteps);
      if(display) px.watch((display<0), STRING("t="<<t<<";"<<s));
    }
    x[t] = world.q;
//      K.calc_fwdPropagateFrames();
//    K.watch();
  }
//  K.watch(true);
  world.setJointState(x[0]);
  if(configurations.N) {
    for(uint s=0; s<k_order; s++) {
      configurations(s)->setJointState(x[0]);
    }
  }
}

void KOMO_ext::playInPhysics(uint subSteps, bool display) {
  arr vels;
  PhysXInterface& px = world.physx();
  for(uint t=0; t<T; t++) {
    px.pushFullState(configurations(k_order+t)->frames, NoArr, configurations(k_order+t-1), configurations(k_order+t-2), tau, true);
    for(uint s=0; s<subSteps; s++) {
      if(display) px.watch(false, STRING("t="<<t<<";"<<s));
      world.physx().step(tau/subSteps);
    }
    px.pullDynamicStates(configurations(k_order+t)->frames, vels);
  }
  //  for(uint i=0;i<vels.d0;i++) if(i<world.frames.N) cout <<world.frames(i)->name <<" v=" <<vels[i] <<endl;
}

void KOMO::reportProblem(std::ostream& os) {
  os <<"KOMO Problem:" <<endl;
  os <<"  x-dim:" <<x.N <<"  dual-dim:" <<dual.N <<endl;
  os <<"  T:" <<T <<" k:" <<k_order <<" phases:" <<double(T)/stepsPerPhase <<" stepsPerPhase:" <<stepsPerPhase <<" tau:" <<tau <<endl;
  os <<"  #configurations:" <<configurations.N <<" q-dims: ";
  uintA dims(configurations.N);
  for(uint i=0; i<configurations.N; i++) dims(i)=configurations(i)->q.N;
  writeConsecutiveConstant(os, dims);
  os <<endl;
  
  if(configurations.N){
    arr times = getPath_times();
    if(times.N>10) times.resizeCopy(10);
    os <<"    times:" <<times <<endl;
  }
  
  os <<"  usingSwift:" <<useSwift <<endl;
  for(Objective* t:objectives) os <<"    " <<*t <<endl;
  for(KinematicSwitch* sw:switches) {
    os <<"    ";
    if(sw->timeOfApplication+k_order >= configurations.N) {
      LOG(-1) <<"switch time " <<sw->timeOfApplication <<" is beyond time horizon " <<T;
      sw->write(os, NULL);
    } else {
      sw->write(os, configurations(sw->timeOfApplication+k_order));
    }
    os <<endl;
  }
  for(Flag* fl:flags) {
    os <<"    ";
    if(fl->stepOfApplication+k_order >= configurations.N) {
      LOG(-1) <<"flag time " <<fl->stepOfApplication <<" is beyond time horizon " <<T;
      fl->write(os, NULL);
    } else {
      fl->write(os, configurations(fl->stepOfApplication+k_order));
    }
    os <<endl;
  }
}

void KOMO::checkGradients(bool dense) {
  CHECK(T,"");
  if(!splineB.N) {
#if 0
    checkJacobianCP(Convert(komo_problem), x, 1e-4);
#else
    double tolerance=1e-4;

    Conv_KOMO_ConstrainedProblem CP_komo(komo_problem);
    ConstrainedProblem *CP=&CP_komo;
    if(dense) CP = &dense_problem;

    VectorFunction F = [CP](arr& phi, arr& J, const arr& x) {
      return CP->phi(phi, J, NoArr, NoTermTypeA, x, NoArr);
    };
//    checkJacobian(F, x, tolerance);
    arr J;
    arr JJ=finiteDifferenceJacobian(F, x, J);
    bool succ=true;
    double mmd=0.;
    for(uint i=0; i<J.d0; i++) {
      uint j;
      double md=maxDiff(J[i], JJ[i], &j);
      if(md>mmd) mmd=md;
      if(md>tolerance && md>fabs(J(i,j))*tolerance) {
        if(!dense){
          LOG(-1) <<"FAILURE in line " <<i <<" t=" <<CP_komo.featureTimes(i) <<' ' <<komo_problem.featureNames(i) <<" -- max diff=" <<md <<" |"<<J(i,j)<<'-'<<JJ(i,j)<<"| (stored in files z.J_*)";
        }else{
          LOG(-1) <<"FAILURE in line " <<i <<" t=" <</*CP_komo.featureTimes(i) <<' ' <<komo_problem.featureNames(i) <<*/" -- max diff=" <<md <<" |"<<J(i,j)<<'-'<<JJ(i,j)<<"| (stored in files z.J_*)";
        }
        J[i] >>FILE("z.J_analytical");
        JJ[i] >>FILE("z.J_empirical");
        //cout <<"\nmeasured grad=" <<JJ <<"\ncomputed grad=" <<J <<endl;
        //HALT("");
//        return;
        rai::wait();
        succ=false;
      }
    }
    if(succ) cout <<"jacobianCheck -- SUCCESS (max diff error=" <<mmd <<")" <<endl;
#endif
  } else {
    Conv_KOMO_ConstrainedProblem P0(komo_problem);
    Conv_linearlyReparameterize_ConstrainedProblem P1(P0, splineB);
    checkJacobianCP(P1, z, 1e-4);
  }
}

void KOMO::plotTrajectory() {
  ofstream fil("z.trajectories");
  StringA jointNames = world.getJointNames();
  //first line: legend
  for(auto s:jointNames) fil <<s <<' ';
  fil <<endl;
  
  x.reshape(T, world.getJointStateDimension());
  x.write(fil, NULL, NULL, "  ");
  fil.close();
  
  ofstream fil2("z.trajectories.plt");
  fil2 <<"set key autotitle columnheader" <<endl;
  fil2 <<"set title 'trajectories'" <<endl;
  fil2 <<"set term qt 2" <<endl;
  fil2 <<"plot 'z.trajectories' \\" <<endl;
  for(uint i=1; i<=jointNames.N; i++) fil2 <<(i>1?"  ,''":"     ") <<" u (($0+1)/" <<stepsPerPhase <<"):"<<i<<" w l lw 3 lc " <<i <<" lt " <<1-((i/10)%2) <<" \\" <<endl;
//  if(dual.N) for(uint i=0;i<objectives.N;i++) fil <<"  ,'' u (($0+1)/" <<stepsPerPhase <<"):"<<1+objectives.N+i<<" w l \\" <<endl;
  fil2 <<endl;
  fil2.close();
  
  gnuplot("load 'z.trajectories.plt'");
}

void KOMO::plotPhaseTrajectory() {
  ofstream fil("z.phase");
  //first line: legend
  fil <<"phase" <<endl;
  
  arr X = getPath_times();
  
  X.reshape(T, 1);
  X.write(fil, NULL, NULL, "  ");
  fil.close();
  
  ofstream fil2("z.phase.plt");
  fil2 <<"set key autotitle columnheader" <<endl;
  fil2 <<"set title 'phase'" <<endl;
  fil2 <<"set term qt 2" <<endl;
  fil2 <<"plot 'z.phase' u (($0+1)/" <<stepsPerPhase <<"):1 w l lw 3 lc 1 lt 1" <<endl;
  fil2 <<endl;
  fil2.close();
  
  gnuplot("load 'z.phase.plt'");
}

struct DrawPaths : GLDrawer {
  arr& X;
  DrawPaths(arr& X): X(X) {}
  void glDraw(OpenGL& gl) {
    glColor(0.,0.,0.);
    for(uint i=0; i<X.d0; i++) {
      glBegin(GL_LINES);
      for(uint t=0; t<X.d1; t++) {
        rai::Transformation pose;
        pose.set(&X(i,t,0));
//          glTransform(pose);
        glVertex3d(pose.pos.x, pose.pos.y, pose.pos.z);
      }
      glEnd();
    }
  }
};

bool KOMO::displayTrajectory(double delay, bool watch, bool overlayPaths, const char* saveVideoPrefix) {
  const char* tag = "KOMO planned trajectory";
  rai::String timetag;
  if(!gl) {
    gl = make_shared<OpenGL>("KOMO display");
    gl->camera.setDefault();
  }
  
  uintA allFrames;
  allFrames.setStraightPerm(configurations.first()->frames.N);
  arr X = getPath_frames(allFrames);
  DrawPaths drawX(X);
  
  for(int t=-(int)k_order; t<(int)T; t++) {
    rai::KinematicWorld& K = *configurations(t+k_order);
    timetag.clear() <<tag <<" (config:" <<t <<'/' <<T <<"  s:" <<conv_step2time(t,stepsPerPhase) <<" tau:" <<K.frames.first()->tau <<')';
//    K.reportProxies();
    K.orsDrawProxies=false;
    gl->clear();
    gl->add(glStandardScene, 0);
    gl->add(K);
    if(overlayPaths) gl->add(drawX);
    if(delay<0.) {
      if(delay<-10.) FILE("z.graph") <<K;
      gl->watch(timetag.p);
    } else {
      gl->update(timetag.p);
      if(delay) rai::wait(delay * K.frames.first()->tau);
    }
    if(saveVideoPrefix) write_ppm(gl->captureImage, STRING(saveVideoPrefix<<std::setw(4)<<std::setfill('0')<<t<<".ppm"));
  }
  if(watch) {
    int key = gl->watch(timetag.p);
    return !(key==27 || key=='q');
  }
  gl->clear();
  return false;
}

bool KOMO::displayPath(bool watch, bool full) {
  uintA allFrames;
  allFrames.setStraightPerm(configurations.first()->frames.N);
  arr X = getPath_frames(allFrames);
  CHECK_EQ(X.nd, 3, "");
  CHECK_EQ(X.d2, 7, "");
  
  DrawPaths drawX(X);
  
  if(!gl) {
    gl = make_shared<OpenGL>("KOMO display");
    gl->camera.setDefault();
  }
  gl->clear();
  gl->add(glStandardScene, 0);
  if(!full){
    gl->addDrawer(configurations.last());
    gl->add(drawX);
  }else{
    for(uint t=0;t<T;t++) gl->addDrawer(configurations(k_order+t));
  }
  if(watch) {
    int key = gl->watch();
    gl.reset();
    return !(key==27 || key=='q');
  }
  gl->update(NULL, true);
  gl.reset();
//  gl->clear();
  return true;
}

Camera& KOMO::displayCamera() {
  if(!gl) {
    gl = make_shared<OpenGL>("KOMO display");
    gl->camera.setDefault();
  }
  return gl->camera;
}

//===========================================================================

void KOMO::setupConfigurations() {

  //IMPORTANT: The configurations need to include the k prefix configurations!
  //Therefore configurations(0) is for time=-k and configurations(k+t) is for time=t
  CHECK(!configurations.N,"why setup again?");
//    listDelete(configurations);

  computeMeshNormals(world.frames, true);
  
  configurations.append(new KinematicWorld())->copy(world, true);
  configurations.last()->setTimes(tau); //(-tau*k_order);
  configurations.last()->calc_q();
  configurations.last()->checkConsistency();
  for(KinematicSwitch *sw:switches) {
    if(sw->timeOfApplication+(int)k_order<=0) {
      sw->apply(*configurations.last());
    }
  }
  for(uint s=1; s<k_order+T; s++) {
    configurations.append(new KinematicWorld())->copy(*configurations(s-1), true);
    rai::KinematicWorld& K = *configurations(s);
    K.setTimes(tau); //(tau*(int(s)-int(k_order)));
    K.checkConsistency();
    CHECK_EQ(configurations(s), configurations.last(), "");
    //apply potential graph switches
    for(KinematicSwitch *sw:switches) {
      if(sw->timeOfApplication+k_order==s) {
        sw->apply(K);
      }
    }
    //apply potential PERSISTENT flags
    for(Flag *fl:flags) {
      if(fl->persist && fl->stepOfApplication+k_order==s) {
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
  
  //now apply NON-PERSISTENT flags
  for(uint s=1; s<k_order+T; s++) {
    for(Flag *fl:flags) {
      if(!fl->persist && fl->stepOfApplication+k_order==s) {
        fl->apply(*configurations(s));
      }
    }
  }
}

void KOMO::set_x(const arr& x, const uintA& selectedConfigurationsOnly) {
  if(!configurations.N) setupConfigurations();
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");

  uintA configs;
  if(!!selectedConfigurationsOnly){ //we only set some configurations: those listed in selectedConfigurationsOnly
    configs = selectedConfigurationsOnly;
  }else{
    configs.setStraightPerm(T); //by default, we loop straight through all configurations
  }

  //-- set the configurations' states
  uint x_count=0;
  for(uint t:configs) {
    uint s = t+k_order;
    uint x_dim = dim_x(t);
    if(x_dim) {
      rai::timerRead(true);
      if(x.nd==1)  configurations(s)->setJointState(x({x_count, x_count+x_dim-1}), NoArr);
      else         configurations(s)->setJointState(x[t]);
      timeKinematics += rai::timerRead(true);
      if(useSwift) {
        configurations(s)->stepSwift();
//        configurations(s)->stepFcl();
        //configurations(s)->proxiesToContacts(1.1);
      }
      timeCollisions += rai::timerRead(true);
      x_count += x_dim;
    }
//    configurations(s)->checkConsistency();
  }
  CHECK_EQ(x_count, x.N, "");
}

void KOMO::reportProxies(std::ostream& os, double belowMargin) {
  int s=0;
  for(auto &K:configurations) {
    os <<" **** KOMO PROXY REPORT t=" <<s-(int)k_order <<endl;
    K->reportProxies(os, belowMargin);
    s++;
  }
}

Graph KOMO::getContacts() {
  Graph G;
  int s=0;
  for(auto &K:configurations) {
    for(rai::Contact *con:K->contacts){
      Graph& g = G.newSubgraph();
      g.newNode<int>({"at"}, {}, s-(int)k_order);
      g.newNode<rai::String>({"from"}, {}, con->a.name);
      g.newNode<rai::String>({"to"}, {}, con->b.name);
      g.newNode<arr>({"force"}, {}, con->force);
      g.newNode<arr>({"poa"}, {}, con->position);
    }
    s++;
  }
  return G;
}

struct EffJointInfo {
  rai::Joint *j;
  rai::Transformation Q=0;
  int t, t_start=0, t_end=0;
  double accum=0.;
  EffJointInfo(rai::Joint *j, uint t): j(j), t(t) {}
  void write(ostream& os) const {
    os <<"EffInfo " <<j->frame->parent->name <<"->" <<j->frame->name <<" \t" <<j->type <<" \tt=" <<t_start <<':' <<t_end <<" \tQ=" <<Q;
  }
};
stdOutPipe(EffJointInfo)
bool operator==(const EffJointInfo&, const EffJointInfo&) { return false; }

rai::Array<rai::Transformation> KOMO::reportEffectiveJoints(std::ostream& os) {
  os <<"**** KOMO EFFECTIVE JOINTS" <<endl;
  Graph G;
  std::map<rai::Joint*,Node*> map;
  for(uint s=k_order+1; s<T+k_order; s++) {
    JointL matches = getMatchingJoints({configurations(s-1), configurations(s)}, true);
    for(uint i=0; i<matches.d0; i++) {
      JointL match = matches[i];
      auto *n = new Node_typed<EffJointInfo>(G, {match(1)->frame->name}, {}, EffJointInfo(match(1), s-k_order));
      map[match(1)] = n;
      if(map.find(match(0))==map.end()) map[match(0)] = new Node_typed<EffJointInfo>(G, {match(0)->frame->name}, {}, EffJointInfo(match(0), s-k_order-1));
      Node *other=map[match(0)];
      n->addParent(other);
    }
  }
  
//  for(uint t=0;t<T+k_order;t++){
//    rai::KinematicWorld *K = configurations(t);
//    for(rai::Frame *f:K->frames){
//      if(f->joint && f->joint->constrainToZeroVel)
//        os <<" t=" <<t-k_order <<'\t' <<f->name <<" \t" <<f->joint->type <<" \tq=" <<f->joint->getQ() <<" \tQ=" <<f->Q <<endl;
//    }
//  }

//  G.displayDot();

  for(Node *n:G) {
    if(!n->parents.N) { //a root node -> accumulate all info
      EffJointInfo& info = n->get<EffJointInfo>();
      info.t_start = info.t_end = info.t;
      info.Q = info.j->frame->Q;
      info.accum += 1.;
      Node *c=n;
      for(;;) {
        if(!c->parentOf.N) break;
        c = c->parentOf.scalar();
        EffJointInfo& cinfo = c->get<EffJointInfo>();
        if(info.t_end<cinfo.t) info.t_end=cinfo.t;
        info.Q.rot.add(cinfo.j->frame->Q.rot);
        info.Q.pos += cinfo.j->frame->Q.pos;
        info.accum += 1.;
//        cout <<" t=" <<cinfo.t <<'\t' <<c->keys <<" \t" <<cinfo.j->type <<" \tq=" <<cinfo.j->getQ() <<" \tQ=" <<cinfo.j->frame->Q <<endl;
      }
      info.Q.pos /= info.accum;
      info.Q.rot.normalize();
      cout <<info <<endl;
    }
  }
  
  //-- align this with the switches and return the transforms
  uint s=0;
  rai::Array<rai::Transformation> Qs(switches.N);
  for(Node *n:G) {
    if(!n->parents.N) {
      EffJointInfo& info = n->get<EffJointInfo>();
#ifndef RAI_NOCHECK
      rai::KinematicSwitch *sw = switches(s);
      CHECK_EQ(info.t_start, sw->timeOfApplication, "");
      CHECK_EQ(info.j->type, sw->jointType, "");
//      CHECK_EQ(info.j->frame->parent->ID, sw->fromId, "");
//      CHECK_EQ(info.j->frame->ID, sw->toId, "");
#endif

      Qs(s) = info.Q;
      
      s++;
    }
  }
  
  cout <<Qs <<endl;
  
  return Qs;
}

Graph KOMO::getReport(bool gnuplt, int reportFeatures, std::ostream& featuresOs) {
  bool wasRun = featureValues.N!=0;
  
  //-- collect all task costs and constraints
  StringA name; name.resize(objectives.N);
  arr err=zeros(T, objectives.N);
  arr dualSolution; if(dual.N) dualSolution=zeros(T, objectives.N);
  arr taskC=zeros(objectives.N);
  arr taskG=zeros(objectives.N);
  uint M=0;
  if(!featureDense){
    for(uint t=0; t<T; t++) {
      for(uint i=0; i<objectives.N; i++) {
        Objective *task = objectives(i);
        if(task->isActive(t)) {
          uint d=0;
          if(wasRun) {
            d=task->map->__dim_phi(configurations({t,t+k_order}));
            for(uint j=0; j<d; j++) CHECK_EQ(featureTypes(M+j), task->type,"");
            if(d) {
              if(task->type==OT_sos) {
                for(uint j=0; j<d; j++) err(t,i) += sqr(featureValues(M+j)); //sumOfSqr(phi.sub(M,M+d-1));
                if(dual.N) dualSolution(t, i) = dual(M);
                taskC(i) += err(t,i);
              }
              if(task->type==OT_ineq) {
                for(uint j=0; j<d; j++) err(t,i) += MAX(0., featureValues(M+j));
                if(dual.N) dualSolution(t, i) = dual(M);
                taskG(i) += err(t,i);
              }
              if(task->type==OT_eq) {
                for(uint j=0; j<d; j++) err(t,i) += fabs(featureValues(M+j));
                if(dual.N) dualSolution(t, i) = dual(M);
                taskG(i) += err(t,i);
              }
              M += d;
            }
          }
          if(reportFeatures==1) {
            featuresOs <<std::setw(4) <<t <<' ' <<std::setw(2) <<i <<' ' <<std::setw(2) <<d
                      <<' ' <<std::setw(40) <<task->name
                     <<" k=" <<task->map->order <<" ot=" <<task->type <<" prec=" <<std::setw(4) <<task->map->scale;
            if(task->map->target.N<5) featuresOs <<" y*=[" <<task->map->target <<']'; else featuresOs<<"y*=[..]";
            featuresOs <<" y^2=" <<err(t,i) <<endl;
          }
        }
      }
    }
  }else{ //featureDense=true
    for(uint i=0; i<objectives.N; i++) {
      Objective *task = objectives.elem(i);
      for(uint t=0;t<task->vars.d0;t++) {
        WorldL Ktuple = configurations.sub(convert<uint,int>(task->vars[t]+(int)k_order));
        uint d=0;
        uint time=task->vars(t,-1);
        if(wasRun) {
          d=task->map->__dim_phi(Ktuple);
          for(uint j=0; j<d; j++) CHECK_EQ(featureTypes(M+j), task->type,"");
          if(d) {
            if(task->type==OT_sos) {
              for(uint j=0; j<d; j++) err(time,i) += sqr(featureValues(M+j));
              taskC(i) += err(time,i);
            }
            if(task->type==OT_ineq) {
              for(uint j=0; j<d; j++) err(time,i) += MAX(0., featureValues(M+j));
              taskG(i) += err(time,i);
            }
            if(task->type==OT_eq) {
              for(uint j=0; j<d; j++) err(time,i) += fabs(featureValues(M+j));
              taskG(i) += err(time,i);
            }
            M += d;
          }
        }
        if(reportFeatures==1) {
          featuresOs <<std::setw(4) <<time <<' ' <<std::setw(2) <<i <<' ' <<std::setw(2) <<d <<task->vars[t]
                    <<' ' <<std::setw(40) <<task->name
                   <<" k=" <<task->map->order <<" ot=" <<task->type <<" prec=" <<std::setw(4) <<task->map->scale;
          if(task->map->target.N<5) featuresOs <<" y*=[" <<task->map->target <<']'; else featuresOs<<"y*=[..]";
          featuresOs <<" y^2=" <<err(time,i) <<endl;
        }
      }
    }
  }
  CHECK_EQ(M , featureValues.N, "");
  
  //-- generate a report graph
  Graph report;
  double totalC=0., totalG=0.;
  for(uint i=0; i<objectives.N; i++) {
    Objective *c = objectives(i);
    Graph& g = report.newSubgraph({c->name}, {});
    g.newNode<double>({"order"}, {}, c->map->order);
    g.newNode<String>({"type"}, {}, STRING(c->type.name()));
    g.newNode<double>({"sqrCosts"}, {}, taskC(i));
    g.newNode<double>({"constraints"}, {}, taskG(i));
    totalC += taskC(i);
    totalG += taskG(i);
  }
  report.newNode<double>({"total","sqrCosts"}, {}, totalC);
  report.newNode<double>({"total","constraints"}, {}, totalG);
  
  if(gnuplt) {
    //-- write a nice gnuplot file
    ofstream fil("z.costReport");
    //first line: legend
    for(auto c:objectives) fil <<c->name <<' ';
    for(auto c:objectives) if(c->type==OT_ineq && dualSolution.N) fil <<c->name <<"_dual ";
    fil <<endl;
    
    //rest: just the matrix
    if(true){ // && !dualSolution.N) {
      err.write(fil,NULL,NULL,"  ");
    } else {
      dualSolution.reshape(T, dualSolution.N/(T));
      catCol(err, dualSolution).write(fil,NULL,NULL,"  ");
    }
    fil.close();
    
    ofstream fil2("z.costReport.plt");
    fil2 <<"set key autotitle columnheader" <<endl;
    fil2 <<"set title 'costReport ( plotting sqrt(costs) )'" <<endl;
    fil2 <<"plot 'z.costReport' \\" <<endl;
    for(uint i=1; i<=objectives.N; i++) fil2 <<(i>1?"  ,''":"     ") <<" u (($0+1)/" <<stepsPerPhase <<"):"<<i<<" w l lw 3 lc " <<i <<" lt " <<1-((i/10)%2) <<" \\" <<endl;
    if(dualSolution.N) for(uint i=0; i<objectives.N; i++) fil2 <<"  ,'' u (($0+1)/" <<stepsPerPhase <<"):"<<1+objectives.N+i<<" w l \\" <<endl;
    fil2 <<endl;
    fil2.close();
    
    if(gnuplt) {
//      cout <<"KOMO Report\n" <<report <<endl;
      gnuplot("load 'z.costReport.plt'");
    }
  }
  
  return report;
}

/// output the defined problem as a generic graph, that can also be displayed, saved and loaded
Graph KOMO::getProblemGraph(bool includeValues){
  Graph K;
  //header
#if 0
  Graph& g = K.newSubgraph({"KOMO_specs"}) -> graph();
  g.newNode<uint>({"x_dim"}, {}, x.N);
  g.newNode<uint>({"T"}, {}, T);
  g.newNode<uint>({"k_order"}, {}, k_order);
  g.newNode<double>({"tau"}, {}, tau);
  //  uintA dims(configurations.N);
  //  for(uint i=0; i<configurations.N; i++) dims(i)=configurations(i)->q.N;
  //  g.newNode<uintA>({"q_dims"}, {}, dims);
  //  arr times(configurations.N);
  //  for(uint i=0; i<configurations.N; i++) times(i)=configurations(i)->frames.first()->time;
  //  g.newNode<double>({"times"}, {}, times);
  g.newNode<bool>({"useSwift"}, {}, useSwift);
#endif

  //nodes for each configuration
  //  for(uint s=0;s<configurations.N;s++){
  //    Graph& g = K.newSubgraph({STRING((int)s-(int)k_order)}) -> graph();
  //    g.newNode<uint>({"dim"}, {}, configurations(s)->q.N);
  //    g.newNode<double>({"tau"}, {}, configurations(s)->frames.first()->time);
  //  }

  //  NodeL configs = K.list();
  //  configs.remove(0);

  //objectives
  //  uint t_count=0;
  for(Objective* task : objectives){
    //        Graph& g = K.newSubgraph({}, configs.sub(i+k_order-t->map->order,i+k_order)) -> graph();
    //        Graph& g = K.newSubgraph({}, configs.sub(convert<uint>(task->vars[t]+(int)k_order))) -> graph();
    Graph& g = K.newSubgraph();
    g.isNodeOfGraph->keys.append(task->name);
    g.newNode<rai::String>({"type"}, {}, STRING(task->type));
//    g.newNode<uint>({"dim"}, {}, task->map->__dim_phi(configurations({0,k_order})));
    if(task->vars.N) g.newNode<intA>({"confs"}, {}, task->vars);
    g.copy(task->map->getSpec(world), true);
    if(includeValues){
      arr y,V;
      if(!featureDense){
        for(uint t=0;t<task->vars.N && t<T;t++) if(task->isActive(t)) {
          task->map->__phi(y, NoArr, configurations({t,t+k_order}));
          V.append(y);
        }
      }else{
        for(uint t=0;t<task->vars.d0;t++) {
          WorldL Ktuple = configurations.sub(convert<uint,int>(task->vars[t]+(int)k_order));
          task->map->__phi(y, NoArr, Ktuple);
          V.append(y);
        }
      }
      if(task->type==OT_sos){
        g.newNode<double>({"sos_value"}, {}, sumOfSqr(V));
      }else if(task->type==OT_eq){
        g.newNode<double>({"eq_sumOfAbs"}, {}, sumOfAbs(V));
      }else if(task->type==OT_ineq){
        double c=0.;
        for(double& v:V) if(v>0) c+=v;
        g.newNode<double>({"inEq_sumOfPos"}, {}, c);
      }
    }
    //    t_count++;
  }

//  if(switches.N) RAI_MSG("not implemented for switches yet");
//  if(flags.N) RAI_MSG("not implemented for flags yet");

  return K;
}

double KOMO::getConstraintViolations(){
  Graph R = getReport(false);
  return R.get<double>("constraints");
}

double KOMO::getCosts(){
  Graph R = getReport(false);
  return R.get<double>("sqrCosts");
}

void KOMO::Conv_MotionProblem_KOMO_Problem::getStructure(uintA& variableDimensions, uintA& featureTimes, ObjectiveTypeA& featureTypes) {
  CHECK_EQ(komo.configurations.N, komo.k_order+komo.T, "configurations are not setup yet: use komo.reset()");
  if(!!variableDimensions) {
    variableDimensions.resize(komo.T);
    for(uint t=0; t<komo.T; t++) variableDimensions(t) = komo.configurations(t+komo.k_order)->getJointStateDimension();
  }
  
  if(!!featureTimes) featureTimes.clear();
  if(!!featureTypes) featureTypes.clear();
  featureNames.clear();
  uint M=0;
  phiIndex.resize(komo.T, komo.objectives.N); phiIndex.setZero();
  phiDim.resize(komo.T, komo.objectives.N);   phiDim.setZero();
  for(uint t=0; t<komo.T; t++) {
    for(uint i=0; i<komo.objectives.N; i++) {
      Objective *task = komo.objectives.elem(i);
      if(task->isActive(t)) {
        uint m = task->map->__dim_phi(komo.configurations({t,t+komo.k_order})); //dimensionality of this task
        
        if(!!featureTimes) featureTimes.append(t, m); //consts<uint>(t, m));
        if(!!featureTypes) featureTypes.append(task->type, m); //consts<ObjectiveType>(task->type, m));
        for(uint j=0; j<m; j++)  featureNames.append(STRING(task->name <<'_'<<j));
        
        //store indexing phi <-> tasks
        phiIndex(t, i) = M;
        phiDim(t, i) = m;
        M += m;
      }
    }
  }
  dimPhi = M;
  CHECK_EQ(M, sum(phiDim), "");
}

bool WARN_FIRST_TIME=true;

void KOMO::Conv_MotionProblem_KOMO_Problem::phi(arr& phi, arrA& J, arrA& H, uintA& featureTimes, ObjectiveTypeA& tt, const arr& x, arr& lambda) {
  //==================
  if(!!lambda) prevLambda = lambda;
  const uintA prevPhiIndex=phiIndex, prevPhiDim=phiDim;
  
#if 0
  if(!!lambda && lambda.N>dimPhi) {
    //store old lambdas directly in the constraints....
    uint C=0;
    for(uint t=0; t<komo.T; t++) {
      KinematicWorld& K = *komo.configurations(t+komo.k_order);
      for(Frame *f:K.frames) for(Contact *c:f->contacts) if(&c->a==f) {
        c->lagrangeParameter = lambda(dimPhi + C);
        C++;
      }
    }
    //    cout <<"ENTER: #" <<C <<" constraints" <<endl;
    CHECK_EQ(dimPhi+C, lambda.N, "");
    //cut of the stored lambdas
    lambda.resizeCopy(dimPhi);
  }
#endif
  //==================
  
  //-- set the trajectory
  komo.set_x(x);
  
  CHECK(dimPhi,"getStructure must be called first");
  //  getStructure(NoUintA, featureTimes, tt);
  //  if(WARN_FIRST_TIME){ LOG(-1)<<"calling inefficient getStructure"; WARN_FIRST_TIME=false; }
  phi.resize(dimPhi);
  if(!!tt) tt.resize(dimPhi);
  if(!!J) J.resize(dimPhi);
  if(!!lambda && lambda.N) { lambda.resize(dimPhi); lambda.setZero(); }
  
  arr y, Jy;
  uint M=0;
  for(uint t=0; t<komo.T; t++) {
    //build the Ktuple with order given by map
    WorldL Ktuple = komo.configurations({t, t+komo.k_order});
    uintA Ktuple_dim = getKtupleDim(Ktuple);
    
    for(uint i=0; i<komo.objectives.N; i++) {
      Objective *task = komo.objectives.elem(i);
      if(task->isActive(t)) {
        //query the task map and check dimensionalities of returns
        task->map->__phi(y, (!!J?Jy:NoArr), Ktuple);
//        uint m = task->map->__dim_phi(Ktuple);
//        CHECK_EQ(m,y.N,"");
        if(!!J) CHECK_EQ(y.N, Jy.d0, "");
        if(!!J) CHECK_EQ(Jy.nd, 2, "");
        if(!!J) CHECK_EQ(Jy.d1, Ktuple_dim.last(), "");
        if(!y.N) continue;
        if(absMax(y)>1e10) RAI_MSG("WARNING y=" <<y);
        
#if 0
        //linear transform (target shift)
        arr target;
        if(task->target.N==1) target = consts<double>(task->target.scalar(), y.N);
        else if(task->target.nd==1) target = task->target;
        else if(task->target.nd==2) target = task->target[t];
        if(target.N) {
          if(task->map->flipTargetSignOnNegScalarProduct && scalarProduct(y, target)<-.0) target *= -1.;
          y -= target;
        }
        y *= task->prec(t);
#endif
        
        //write into phi and J
        phi.setVectorBlock(y, M);
        if(!!J) {
//          Jy *= task->prec(t);
          if(t<komo.k_order) Jy.delColumns(0, Ktuple_dim(komo.k_order-t-1)); //delete the columns that correspond to the prefix!!
//          if(t<komo.k_order) Jy.delColumns(0,(komo.k_order-t)*komo.configurations(0)->q.N); //delete the columns that correspond to the prefix!!
          for(uint i=0; i<y.N; i++) J(M+i) = Jy[i]; //copy it to J(M+i); which is the Jacobian of the M+i'th feature w.r.t. its variables
        }
        if(!!tt) for(uint i=0; i<y.N; i++) tt(M+i) = task->type;
        
        //transfer Lambda values
        if(!!lambda && lambda.N && y.N==prevPhiDim(t,i)) {
          lambda.setVectorBlock(prevLambda({prevPhiIndex(t,i), prevPhiIndex(t,i)+y.N-1}), M);
        }
        
//        //store indexing phi <-> tasks
//        phiIndex(t, i) = M;
//        phiDim(t, i) = y.N;

        //counter for features phi
        M += y.N;
      }
    }
  }
  
  CHECK_EQ(M, dimPhi, "");
//  if(!!lambda) CHECK_EQ(prevLambda, lambda, ""); //this ASSERT only holds is none of the tasks is variable dim!
  komo.featureValues = phi;
  if(!!tt) komo.featureTypes = tt;
  komo.featureDense=false;

  if(komo.animateOptimization>0) {
    if(komo.animateOptimization>1){
      if(komo.animateOptimization>2)
        cout <<komo.getReport(true) <<endl;
      komo.displayPath(true);
    }else{
      komo.displayPath(false);
    }
//    komo.plotPhaseTrajectory();
//    rai::wait();
  }

  //==================
#if 0
  uint C=0;
  bool updateLambda = ((&lambda) && lambda.N==dimPhi);
  for(uint t=0; t<komo.T; t++) {
    WorldL Ktuple = komo.configurations({t, t+komo.k_order});
    KinematicWorld& K = *komo.configurations(t+komo.k_order);
    for(Frame *f:K.frames) for(Contact *c:f->contacts) if(&c->a==f) {
          Feature *map = c->getTM_ContactNegDistance();
          map->phi(y, (!!J?Jy:NoArr), Ktuple, komo.tau, t);
          c->y = y.scalar();
          phi.append(c->y);
//      featureTypes.append(OT_ineq);
          if(!!featureTimes) featureTimes.append(t);
          if(!!J) {
            if(t<komo.k_order) Jy.delColumns(0,(komo.k_order-t)*komo.configurations(0)->q.N); //delete the columns that correspond to the prefix!!
            J.append(Jy);   //copy it to J(M+i); which is the Jacobian of the M+i'th feature w.r.t. its variables
          }
          if(!!tt) tt.append(OT_ineq);
          
          if(updateLambda) {
            lambda.append(c->lagrangeParameter);
//        cout <<"APPENDED: " <<C <<" t=" <<t <<' ' <<*c <<endl;
          }
          C++;
        }
  }
  if(updateLambda) CHECK_EQ(lambda.N, phi.N, "");
//  cout <<"EXIT:  #" <<C <<" constraints" <<endl;
#endif
  //==================
  
}

void KOMO::Conv_MotionProblem_DenseProblem::phi(arr& phi, arr& J, arr& H, ObjectiveTypeA& tt, const arr& x, arr& lambda) {
  //-- set the trajectory
  komo.set_x(x);

  if(!dimPhi) getDimPhi();
//  CHECK(dimPhi,"getStructure must be called first");
//  getStructure(NoUintA, featureTimes, tt);
//  if(WARN_FIRST_TIME){ LOG(-1)<<"calling inefficient getStructure"; WARN_FIRST_TIME=false; }
  phi.resize(dimPhi);
  if(!!tt) tt.resize(dimPhi);
  if(!!J) J.resize(dimPhi, x.N).setZero();
  if(!!lambda && lambda.N) { lambda.resize(dimPhi); lambda.setZero(); }

  uintA x_index = getKtupleDim(komo.configurations({komo.k_order,-1}));
  x_index.prepend(0);
//  for(uint t=0; t<komo.T; t++) {
//    //build the Ktuple with order given by map

  arr y, Jy;
  uint M=0;
  for(uint i=0; i<komo.objectives.N; i++) {
    Objective *task = komo.objectives.elem(i);
    for(uint t=0;t<task->vars.d0;t++) {
      WorldL Ktuple = komo.configurations.sub(convert<uint,int>(task->vars[t]+(int)komo.k_order));
      uintA kdim = getKtupleDim(Ktuple);
      kdim.prepend(0);

      //query the task map and check dimensionalities of returns
      task->map->__phi(y, (!!J?Jy:NoArr), Ktuple);
      if(!!J) CHECK_EQ(y.N, Jy.d0, "");
      if(!!J) CHECK_EQ(Jy.nd, 2, "");
      if(!!J) CHECK_EQ(Jy.d1, kdim.last(), "");
      if(!y.N) continue;
      if(absMax(y)>1e10) RAI_MSG("WARNING y=" <<y);

#if 0
      //linear transform (target shift)
      arr target;
      if(task->target.N==1) target = consts<double>(task->target.scalar(), y.N);
      else if(task->target.nd==1) target = task->target;
      else if(task->target.nd==2) target = task->target[t];
      if(target.N) {
        if(task->map->flipTargetSignOnNegScalarProduct && scalarProduct(y, target)<-.0) target *= -1.;
        y -= target;
      }
      y *= task->prec(t);
#endif

      //write into phi and J
      phi.setVectorBlock(y, M);

      if(!!J) {
//        Jy *= task->prec(t);
        for(uint j=0;j<task->vars.d1;j++){
          if(task->vars(t,j)>=0){
            J.setMatrixBlock(Jy.sub(0,-1,kdim(j),kdim(j+1)-1), M, x_index(task->vars(t,j)));
          }
        }
      }

      if(!!tt) for(uint i=0; i<y.N; i++) tt(M+i) = task->type;

      //counter for features phi
      M += y.N;
    }
  }

  CHECK_EQ(M, dimPhi, "");
//  if(!!lambda) CHECK_EQ(prevLambda, lambda, ""); //this ASSERT only holds is none of the tasks is variable dim!
  komo.featureValues = phi;
  if(!!tt) komo.featureTypes = tt;
  komo.featureDense=true;

  if(komo.animateOptimization>0) {
    if(komo.animateOptimization>1){
      if(komo.animateOptimization>2)
        cout <<komo.getReport(true) <<endl;
      komo.displayPath(true);
    }else{
      komo.displayPath(false);
    }
//    komo.plotPhaseTrajectory();
//    rai::wait();
  }

}

void KOMO::Conv_MotionProblem_DenseProblem::getDimPhi() {
  CHECK_EQ(komo.configurations.N, komo.k_order+komo.T, "configurations are not setup yet: use komo.reset()");
  uint M=0;
  for(uint i=0; i<komo.objectives.N; i++) {
    Objective *task = komo.objectives.elem(i);
    for(uint t=0;t<task->vars.d0;t++) {
      M += task->map->__dim_phi(komo.configurations({t,t+komo.k_order})); //dimensionality of this task
    }
  }
  dimPhi = M;
}

void KOMO::Conv_MotionProblem_GraphProblem::getStructure(uintA& variableDimensions, intAA& featureVariables, ObjectiveTypeA& featureTypes) {
  CHECK_EQ(komo.configurations.N, komo.k_order+komo.T, "configurations are not setup yet: use komo.reset()");
  if(!!variableDimensions) {
    variableDimensions.resize(komo.T);
    for(uint t=0; t<komo.T; t++) variableDimensions(t) = komo.configurations(t+komo.k_order)->getJointStateDimension();
  }

  if(!!featureVariables) featureVariables.clear();
  if(!!featureTypes) featureTypes.clear();
  uint M=0;
  for(Objective *ob:komo.objectives) {
    CHECK_EQ(ob->vars.nd, 2, "in sparse mode, vars need to be tuples of variables");
    for(uint t=0;t<ob->vars.d0;t++) {
      WorldL Ktuple = komo.configurations.sub(convert<uint,int>(ob->vars[t]+(int)komo.k_order));
      uint m = ob->map->__dim_phi(Ktuple); //dimensionality of this task
      if(!!featureVariables) featureVariables.append(ob->vars[t], m);
      if(!!featureTypes) featureTypes.append(ob->type, m);
      M += m;
    }
  }

  if(!!featureTypes) komo.featureTypes = featureTypes;

  dimPhi = M;
}

void KOMO::Conv_MotionProblem_GraphProblem::getSemantics(StringA& varNames, StringA& phiNames){
  varNames.resize(komo.T);
  for(uint t=0; t<komo.T; t++) varNames(t) <<"config_" <<t;

  phiNames.clear();
  uint M=0;
  for(Objective *ob:komo.objectives) {
    CHECK_EQ(ob->vars.nd, 2, "in sparse mode, vars need to be tuples of variables");
    for(uint t=0;t<ob->vars.d0;t++) {
      WorldL Ktuple = komo.configurations.sub(convert<uint,int>(ob->vars[t]+(int)komo.k_order));
      uint m = ob->map->__dim_phi(Ktuple); //dimensionality of this task
      phiNames.append(ob->name, m);
      M += m;
    }
  }
}


void KOMO::Conv_MotionProblem_GraphProblem::phi(arr& phi, arrA& J, arrA& H, const arr& x){
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
    for(uint t=0;t<ob->vars.d0;t++) {
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

  if(komo.animateOptimization>0) {
    if(komo.animateOptimization>1){
      if(komo.animateOptimization>2)
        cout <<komo.getReport(true) <<endl;
      komo.displayPath(true);
    }else{
      komo.displayPath(false);
    }
    //    komo.plotPhaseTrajectory();
    //    rai::wait();
  }

}

void KOMO::Conv_MotionProblem_GraphProblem::setPartialX(const uintA& whichX, const arr& x){
  komo.set_x(x, whichX);
}

void KOMO::Conv_MotionProblem_GraphProblem::getPartialPhi(arr& phi, arrA& J, arrA& H, const uintA& whichPhi){
  //NON EFFICIENT

  { //copy and past from full phi!
    CHECK(dimPhi,"getStructure must be called first");
    if(!!phi) phi.resize(dimPhi);
    if(!!J) J.resize(dimPhi);

//    uintA x_index = getKtupleDim(komo.configurations({komo.k_order,-1}));
//    x_index.prepend(0);

    arr y, Jy;
    uint M=0;
    for(Objective *ob:komo.objectives) {
      CHECK_EQ(ob->vars.nd, 2, "in sparse mode, vars need to be tuples of variables");
      for(uint t=0;t<ob->vars.d0;t++) {
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
        if(!!phi) phi.setVectorBlock(y, M);

        if(!!J) {
          for(uint j=ob->vars.d1;j--;){
            if(ob->vars(t,j)<0){
              Jy.delColumns(kdim(j),kdim(j+1)-kdim(j)); //delete the columns that correspond to the prefix!!
            }
          }
          for(uint i=0; i<y.N; i++) J(M+i) = Jy[i];
        }

        //counter for features phi
        M += y.N;
      }
    }

    CHECK_EQ(M, dimPhi, "");
    //  if(!!lambda) CHECK_EQ(prevLambda, lambda, ""); //this ASSERT only holds is none of the tasks is variable dim!
    if(!!phi) komo.featureValues = phi;
    komo.featureDense = true;

    if(komo.animateOptimization>0) {
      if(komo.animateOptimization>1){
        if(komo.animateOptimization>2)
          cout <<komo.getReport(true) <<endl;
        komo.displayPath(true);
      }else{
        komo.displayPath(false);
      }
      //    komo.plotPhaseTrajectory();
      //    rai::wait();
    }
  }

  //now subselect features
  if(!!phi) phi = phi.sub(whichPhi);
  if(!!J) J = J.sub(whichPhi);
}

rai::KinematicWorld& KOMO::getConfiguration(double phase) {
  uint s = k_order + (uint)(phase*double(stepsPerPhase));
  return *configurations(s);
}

arr KOMO::getJointState(double phase) {
  uint s = k_order + (uint)(phase*double(stepsPerPhase));
  return configurations(s)->getJointState();
}

arr KOMO::getFrameState(double phase) {
  uint s = k_order + (uint)(phase*double(stepsPerPhase));
  return configurations(s)->getFrameState();
}

arr KOMO::getPath_decisionVariable() {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  arr x;
  for(uint t=0; t<T; t++) x.append(configurations(t+k_order)->getJointState());
  return x;
}

arr KOMO::getPath(const StringA &joints) {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  uint n = joints.N;
  if(!n) n = world.getJointStateDimension();
  arr X(T, n);
  for(uint t=0; t<T; t++) {
    if(joints.N)
      X[t] = configurations(t+k_order)->getJointState(joints);
    else
      X[t] = configurations(t+k_order)->getJointState();
  }
  return X;
}

arr KOMO::getPath(const uintA &joints) {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  arr X = configurations(k_order)->getJointState(joints);
  X.resizeCopy(T, X.N);
  for(uint t=1; t<T; t++) {
    X[t] = configurations(k_order+t)->getJointState(joints);
  }
  return X;
}

arr KOMO::getPath_frames(const StringA &frame) {
  uintA _frames;
  if(frame.N) for(const rai::String& f:frame) _frames.append( world.getFrameByName(f)->ID );
  else _frames.setStraightPerm(world.frames.N);
  return getPath_frames(_frames);
}

arr KOMO::getPath_frames(const uintA &frames) {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  arr X(frames.N, T, 7);
  for(uint t=0; t<T; t++) {
    for(uint i=0; i<frames.N; i++) {
      X(i, t, {}) = configurations(t+k_order)->frames(frames(i))->X.getArr7d();
    }
  }
  return X;
}

arrA KOMO::getPath_q() {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  arrA q(T);
  for(uint t=0; t<T; t++) {
    q(t) = configurations(t+k_order)->getJointState();
  }
  return q;
}

arr KOMO::getPath_tau() {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  arr X(T);
  for(uint t=0; t<T; t++) {
    X(t) = configurations(t+k_order)->frames.first()->tau;
  }
  return X;
}

arr KOMO::getPath_times() {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  arr X(T);
  double time=0.;
  for(uint t=0; t<T; t++) {
    time += configurations(t+k_order)->frames.first()->tau;
    X(t) = time;
  }
  return X;
}

arr KOMO::getPath_energies() {
  CHECK_EQ(configurations.N, k_order+T, "configurations are not setup yet");
  TM_Energy E;
  E.order=1;
  arr X(T), y;
  for(uint t=0; t<T; t++) {
    E.phi(y, NoArr, {configurations(t+k_order-1), configurations(t+k_order)});
    X(t) = y.scalar();
  }
  return X;
}


template<> const char* rai::Enum<SkeletonSymbol>::names []= {
  "touch",
  "above",
  "inside",
  "impulse",
  "initial",
  "stable",
  "stableOn",
  "dynamic",
  "dynamicOn",
  "dynamicTrans",
  "liftDownUp",
  "break",

  "contact",
  "bounce",

  "magic",
  "magicTrans",

  "push",
  "graspSlide",

  "noCollision",
  "identical",

  "alignByInt",

  "makeFree",
  "stableRelPose",
  "stablePose",
  NULL
};

intA getSwitchesFromSkeleton(const Skeleton& S){
  rai::Array<SkeletonSymbol> modes = { SY_stable, SY_stableOn, SY_dynamic, SY_dynamicOn, SY_dynamicTrans };

  intA ret;
  for(int i=0;i<(int)S.N;i++){
    if(modes.contains(S.elem(i).symbol)){
      int j=i-1;
      for(;j--;){
        if(modes.contains(S.elem(j).symbol) && S.elem(j).frames.last()==S.elem(i).frames.last()){
          break;
        }
      }
      ret.append({j,i});
    }
  }
  ret.reshape(ret.N/2,2);

  return ret;
}

void writeSkeleton(ostream& os, const Skeleton& S, const intA& switches){
  os <<"SKELETON:" <<endl;
  for(auto &s:S) os <<"  " <<s <<endl;
  if(switches.N){
    os <<"SWITCHES:" <<endl;
    for(uint i=0;i<switches.d0;i++){
      int j = switches(i,0);
      if(j<0)
        os <<"  START  -->  " <<S(switches(i,1)) <<endl;
      else
        os <<"  " <<S(j) <<"  -->  " <<S(switches(i,1)) <<endl;
    }
  }
}

