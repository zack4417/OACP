/*  ------------------------------------------------------------------
    Copyright (c) 2017 Marc Toussaint
    email: marc.toussaint@informatik.uni-stuttgart.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file
 * @ingroup group_ors
 */
/**
 * @addtogroup group_ors
 * @{
 */

#undef abs
#include <algorithm>
#include <sstream>
#include <climits>
#include "kin.h"
#include "frame.h"
#include "contact.h"
#include "uncertainty.h"
#include "proxy.h"
#include "kin_swift.h"
#include "kin_physx.h"
#include "kin_ode.h"
#include "kin_feather.h"
#include "featureSymbols.h"
#include <Geo/fclInterface.h>
#include <Geo/qhull.h>
#include <Geo/mesh_readAssimp.h>
#include <GeoOptim/geoOptim.h>
#include <Gui/opengl.h>
#include <Algo/algos.h>
#include <iomanip>

#ifndef RAI_ORS_ONLY_BASICS
#  include <Core/graph.h>
//#  include <Plot/plot.h>
#endif

#define RAI_extern_ply
#ifdef RAI_extern_ply
#  include <Geo/ply/ply.h>
#endif

#ifdef RAI_GL
#  include <GL/gl.h>
#  include <GL/glu.h>
#endif

#define RAI_NO_DYNAMICS_IN_FRAMES

#define SL_DEBUG_LEVEL 1
#define SL_DEBUG(l, x) if(l<=SL_DEBUG_LEVEL) x;

#define Qstate

void lib_ors() { cout <<"force loading lib/ors" <<endl; }

#define LEN .2

#ifndef RAI_ORS_ONLY_BASICS

uint rai::KinematicWorld::setJointStateCount = 0;

//===========================================================================
//
// contants
//

rai::Frame& NoFrame = *((rai::Frame*)NULL);
rai::Shape& NoShape = *((rai::Shape*)NULL);
rai::Joint& NoJoint = *((rai::Joint*)NULL);
rai::KinematicWorld __NoWorld;
rai::KinematicWorld& NoWorld = *((rai::KinematicWorld*)&__NoWorld);

uintA stringListToShapeIndices(const rai::Array<const char*>& names, const rai::KinematicWorld& K) {
  uintA I(names.N);
  for(uint i=0; i<names.N; i++) {
    rai::Frame *f = K.getFrameByName(names(i));
    if(!f) HALT("shape name '"<<names(i)<<"' doesn't exist");
    I(i) = f->ID;
  }
  return I;
}

uintA shapesToShapeIndices(const FrameL& frames) {
  uintA I;
  resizeAs(I, frames);
  for(uint i=0; i<frames.N; i++) I.elem(i) = frames.elem(i)->ID;
  return I;
}

void makeConvexHulls(FrameL& frames, bool onlyContactShapes) {
  for(rai::Frame *f: frames) if(f->shape && (!onlyContactShapes || f->shape->cont))
    f->shape->mesh().makeConvexHull();
}

void computeOptimalSSBoxes(FrameL &frames) {
  NIY;
#if 0
  //  for(rai::Shape *s: shapes) s->mesh.computeOptimalSSBox(s->mesh.V);
  rai::Shape *s;
  for(rai::Frame *f: frames) if((s=f->shape)) {
    if(!(s->type()==rai::ST_mesh && s->mesh.V.N)) continue;
    rai::Transformation t;
    arr x;
    computeOptimalSSBox(s->mesh, x, t, s->mesh.V);
    s->type() = rai::ST_ssBox;
    s->size(0)=2.*x(0); s->size(1)=2.*x(1); s->size(2)=2.*x(2); s->size(3)=x(3);
    s->mesh.setSSBox(s->size(0), s->size(1), s->size(2), s->size(3));
    s->frame.Q.appendTransformation(t);
  }
#endif
}

void computeMeshNormals(FrameL& frames, bool force) {
  for(rai::Frame *f: frames) if(f->shape) {
    rai::Shape *s = f->shape;
    if(force || s->mesh().V.d0!=s->mesh().Vn.d0 || s->mesh().T.d0!=s->mesh().Tn.d0) s->mesh().computeNormals();
    if(force || s->sscCore().V.d0!=s->sscCore().Vn.d0 || s->sscCore().T.d0!=s->sscCore().Tn.d0) s->sscCore().computeNormals();
  }
}

bool always_unlocked(void*) { return false; }

//===========================================================================
//
// KinematicWorld
//

namespace rai {
  struct sKinematicWorld {
    OpenGL *gl;
    std::shared_ptr<SwiftInterface> swift;
    ptr<FclInterface> fcl;
    PhysXInterface *physx;
    OdeInterface *ode;
    FeatherstoneInterface *fs = NULL;
    sKinematicWorld():gl(NULL), physx(NULL), ode(NULL) {}
    ~sKinematicWorld() {
      if(gl) delete gl;
      if(physx) delete physx;
      if(ode) delete ode;
    }
  };

}

rai::KinematicWorld::KinematicWorld() : s(NULL) {
  frames.memMove=proxies.memMove=true;
  s=new sKinematicWorld;
}

rai::KinematicWorld::KinematicWorld(const rai::KinematicWorld& other, bool referenceSwiftOnCopy) : KinematicWorld() {
  copy(other, referenceSwiftOnCopy);
}

rai::KinematicWorld::KinematicWorld(const char* filename) : KinematicWorld() {
  init(filename);
}

rai::KinematicWorld::~KinematicWorld() {
  //delete OpenGL and the extensions first!
  delete s;
  s=NULL;
  clear();
}

void rai::KinematicWorld::init(const char* filename) {
  rai::FileToken file(filename, true);
  Graph G(file);
  G.checkConsistency();
  init(G, false);
  file.cd_start();
}

rai::Frame* rai::KinematicWorld::addFile(const char* filename) {
  uint n=frames.N;
  rai::FileToken file(filename, true);
  Graph G(file);
  init(G, true);
  file.cd_start();
  if(frames.N==n) return 0;
  return frames(n); //returns 1st frame of added file
}

rai::Frame* rai::KinematicWorld::addFile(const char* filename, const char* parentOfRoot, const rai::Transformation& relOfRoot){
  rai::Frame *f = addFile(filename);
  if(parentOfRoot){
    CHECK(f, "nothing added?");
    f->linkFrom(getFrameByName(parentOfRoot));
    new rai::Joint(*f, rai::JT_rigid);
    f->Q = relOfRoot;
  }
  calc_activeSets();
  calc_fwdPropagateFrames();
  return f;
}

void rai::KinematicWorld::addAssimp(const char* filename) {
  AssimpLoader A(filename);
  for(rai::Mesh &m:A.meshes){
    rai::Frame *f = new rai::Frame(*this);
    rai::Shape *s = new rai::Shape(*f);
    s->type() = ST_mesh;
    s->mesh() = m;
  }
}

rai::Frame* rai::KinematicWorld::addFrame(const char* name, const char* parent, const char* args){
  rai::Frame *f = new rai::Frame(*this);
  f->name = name;

  if(parent && parent[0]){
    rai::Frame *p = getFrameByName(parent);
    if(p) f->linkFrom(p);
  }

  if(args && args[0]){
    rai::String(args) >>f->ats;
    f->read(f->ats);
  }

  if(f->parent) f->calc_X_from_parent();

  return f;
}

#if 0
rai::Frame* rai::KinematicWorld::addObject(rai::ShapeType shape, const arr& size, const arr& col){
  rai::Frame *f = new rai::Frame(*this);
  rai::Shape *s = new rai::Shape(*f);
  s->type() = shape;
  if(col.N) s->mesh().C = col;
  if(radius>0.) s->size() = ARR(radius);
  if(shape!=ST_mesh && shape!=ST_ssCvx){
    if(size.N>=1) s->size() = size;
    s->createMeshes();
  }else{
    if(shape==ST_mesh){
      s->mesh().V = size;
      s->mesh().V.reshape(-1,3);
    }
    if(shape==ST_ssCvx){
      s->sscCore().V = size;
      s->sscCore().V.reshape(-1,3);
      CHECK(radius>0., "radius must be greater zero");
      s->size() = ARR(radius);
    }
  }
  return f;
}
#endif

rai::Frame* rai::KinematicWorld::addObject(const char* name, const char* parent, rai::ShapeType shape, const arr& size, const arr& col, const arr& pos, const arr& rot){
  rai::Frame *f = addFrame(name, parent);
  if(f->parent) f->setJoint(rai::JT_rigid);
  f->setShape(shape, size);
  f->setContact(-1);
  if(col.N) f->setColor(col);
  if(f->parent){
    if(pos.N) f->setRelativePosition(pos);
    if(rot.N) f->setRelativeQuaternion(rot);
  }else{
    if(pos.N) f->setPosition(pos);
    if(rot.N) f->setQuaternion(rot);
  }
  return f;
}

/// the list F can be from another (not this) Configuration
void rai::KinematicWorld::addFramesCopy(const FrameL& F){
  uint maxId=0;
  for(Frame *f:F) if(f->ID>maxId) maxId=f->ID;
  intA FId2thisId(maxId+1);
  FId2thisId = -1;
  for(Frame *f:F) {
    Frame *a = new Frame(*this, f);
    FId2thisId(f->ID)=a->ID;
  }
  for(Frame *f:F) if(f->parent && f->parent->ID<=maxId && FId2thisId(f->parent->ID)!=-1){
    frames(FId2thisId(f->ID))->linkFrom(frames(FId2thisId(f->parent->ID)));
  }
}

void rai::KinematicWorld::clear() {
  reset_q();
  proxies.clear(); //while(proxies.N){ delete proxies.last(); /*checkConsistency();*/ }
  while(frames.N) { delete frames.last(); /*checkConsistency();*/ }
  reset_q();
}

void rai::KinematicWorld::reset_q() {
  q.clear();
  qdot.clear();
  fwdActiveSet.clear();
  fwdActiveJoints.clear();
}

FrameL rai::KinematicWorld::calc_topSort() const {
  FrameL fringe;
  FrameL order;
  boolA done = consts<byte>(false, frames.N);
  
  for(Frame *a:frames) if(!a->parent) fringe.append(a);
  if(frames.N) CHECK(fringe.N, "none of the frames is a root -- must be loopy!");
  
  while(fringe.N) {
    Frame *a = fringe.popFirst();
    order.append(a);
    done(a->ID) = true;
    
    for(Frame *ch : a->parentOf) fringe.append(ch);
  }
  
  for(uint i=0; i<done.N; i++) if(!done(i)) LOG(-1) <<"not done: " <<frames(i)->name <<endl;
  CHECK_EQ(order.N, frames.N, "can't top sort");
  
  return order;
}

bool rai::KinematicWorld::check_topSort() const {
  if(fwdActiveSet.N != frames.N) return false;
  
  //compute levels
  intA level = consts<int>(0, frames.N);
  for(Frame *f: fwdActiveSet) if(f->parent) level(f->ID) = level(f->parent->ID)+1;
  //check levels are strictly increasing across links
  for(Frame *f: fwdActiveSet) if(f->parent && level(f->parent->ID) >= level(f->ID)) return false;
  
  return true;
}

void rai::KinematicWorld::calc_activeSets() {
  reset_q();
  if(!check_topSort()) {
    fwdActiveSet = calc_topSort(); //graphGetTopsortOrder<Frame>(frames);
  }
  fwdActiveJoints.clear();
  for(Frame *f:fwdActiveSet) if(f->joint && f->joint->active)
    fwdActiveJoints.append(f->joint);
}

void rai::KinematicWorld::calc_q() {
  calc_activeSets();
  analyzeJointStateDimensions();
  calc_q_from_Q();
}

void rai::KinematicWorld::copy(const rai::KinematicWorld& K, bool referenceSwiftOnCopy) {
  CHECK(this != &K, "never copy K onto itself");

  clear();
  orsDrawProxies = K.orsDrawProxies;
  //copy frames; first each Frame/Link/Joint directly, where all links go to the origin K (!!!); then relink to itself
  for(Frame *f:K.frames) new Frame(*this, f);
  for(Frame *f:K.frames) if(f->parent) frames(f->ID)->linkFrom(frames(f->parent->ID));
  //copy proxies; first they point to origin frames; afterwards, let them point to own frames
  copyProxies(K);
  //  proxies = K.proxies;
  //  for(Proxy& p:proxies) { p.a = frames(p.a->ID); p.b = frames(p.b->ID);  p.coll.reset(); }
  //copy contacts
  for(Contact *c:K.contacts) new Contact(*frames(c->a.ID), *frames(c->b.ID), c);
  //copy swift reference
  if(referenceSwiftOnCopy) s->swift = K.s->swift;
  calc_activeSets();
  q = K.q;
  qdot = K.qdot;
}

bool rai::KinematicWorld::operator!() const { return this==&NoWorld; }

/** @brief KINEMATICS: given the (absolute) frames of root nodes and the relative frames
    on the edges, this calculates the absolute frames of all other nodes (propagating forward
    through trees and testing consistency of loops). */
void rai::KinematicWorld::calc_fwdPropagateFrames() {
  if(fwdActiveSet.N!=frames.N) calc_activeSets();
  for(Frame *f:fwdActiveSet) {
#if 1
    if(f->parent) f->calc_X_from_parent();
#else
    if(f->parent) {
      Transformation &from = f->parent->X;
      Transformation &to = f->X;
      to = from;
      to.appendTransformation(f->Q);
      CHECK_EQ(to.pos.x, to.pos.x, "NAN transformation:" <<from <<'*' <<f->Q);
      if(f->joint) {
        Joint *j = f->joint;
        if(j->type==JT_hingeX || j->type==JT_transX || j->type==JT_XBall)  j->axis = from.rot.getX();
        if(j->type==JT_hingeY || j->type==JT_transY)  j->axis = from.rot.getY();
        if(j->type==JT_hingeZ || j->type==JT_transZ)  j->axis = from.rot.getZ();
        if(j->type==JT_transXYPhi)  j->axis = from.rot.getZ();
        if(j->type==JT_phiTransXY)  j->axis = from.rot.getZ();
      }
    }
#endif
  }
}

arr rai::KinematicWorld::calc_fwdPropagateVelocities() {
  if(fwdActiveSet.N!=frames.N) calc_activeSets();
  arr vel(frames.N, 2, 3);  //for every frame we have a linVel and angVel, each 3D
  vel.setZero();
  rai::Transformation f;
  Vector linVel, angVel, q_vel, q_angvel;
  for(Frame *f : fwdActiveSet) { //this has no bailout for loopy graphs!
    if(f->parent) {
      Frame *from = f->parent;
      Joint *j = f->joint;
      if(j) {
        linVel = vel(from->ID, 0, {});
        angVel = vel(from->ID, 1, {});
        
        if(j->type==JT_hingeX) {
          q_vel.setZero();
          q_angvel.set(qdot(j->qIndex) ,0., 0.);
        } else if(j->type==JT_transX) {
          q_vel.set(qdot(j->qIndex), 0., 0.);
          q_angvel.setZero();
        } else if(j->type==JT_rigid) {
          q_vel.setZero();
          q_angvel.setZero();
        } else if(j->type==JT_transXYPhi) {
          q_vel.set(qdot(j->qIndex), qdot(j->qIndex+1), 0.);
          q_angvel.set(0.,0.,qdot(j->qIndex+2));
        } else NIY;
        
        Matrix R = j->X().rot.getMatrix();
        Vector qV(R*q_vel); //relative vel in global coords
        Vector qW(R*q_angvel); //relative ang vel in global coords
        linVel += angVel^(f->X.pos - from->X.pos);
        /*if(!isLinkTree) */linVel += qW^(f->X.pos - j->X().pos);
        linVel += qV;
        angVel += qW;
        
        for(uint i=0; i<3; i++) vel(f->ID, 0, i) = linVel(i);
        for(uint i=0; i<3; i++) vel(f->ID, 1, i) = angVel(i);
      } else {
        linVel = vel(from->ID, 0, {});
        angVel = vel(from->ID, 1, {});
        
        linVel += angVel^(f->X.pos - from->X.pos);
        
        for(uint i=0; i<3; i++) vel(f->ID, 0, i) = linVel(i);
        for(uint i=0; i<3; i++) vel(f->ID, 1, i) = angVel(i);
      }
    }
  }
  return vel;
}

/** @brief given the absolute frames of all nodes and the two rigid (relative)
    frames A & B of each edge, this calculates the dynamic (relative) joint
    frame X for each edge (which includes joint transformation and errors) */
void rai::KinematicWorld::calc_Q_from_BodyFrames() {
  for(Frame *f:frames) if(f->parent) {
    f->Q.setDifference(f->parent->X, f->X);
  }
}

arr rai::KinematicWorld::naturalQmetric(double power) const {
  HALT("don't use this anymore. use getHmetric instead");
#if 0
  if(!q.N) getJointStateDimension();
  arr Wdiag(q.N);
  Wdiag=1.;
  return Wdiag;
#else
  //compute generic q-metric depending on tree depth
  arr BM(frames.N);
  BM=1.;
  for(uint i=BM.N; i--;) {
    for(uint j=0; j<frames(i)->parentOf.N; j++) {
      BM(i) = rai::MAX(BM(frames(i)->parentOf(j)->ID)+1., BM(i));
      //      BM(i) += BM(bodies(i)->parentOf(j)->to->index);
    }
  }
  if(!q.N) getJointStateDimension();
  arr Wdiag(q.N);
  for(Joint *j: fwdActiveJoints) {
    for(uint i=0; i<j->qDim(); i++) {
      Wdiag(j->qIndex+i) = ::pow(BM(j->frame->ID), power);
    }
  }
  return Wdiag;
#endif
}

/** @brief revert the topological orientation of a joint (edge),
   e.g., when choosing another body as root of a tree */
void rai::KinematicWorld::flipFrames(rai::Frame *a, rai::Frame *b) {
  CHECK_EQ(b->parent, a, "");
  CHECK(!a->parent, "");
  CHECK(!a->joint, "");
  if(b->joint){
    b->joint->flip();
  }
  a->Q = -b->Q;
  b->Q.setZero();
  b->unLink();
  a->linkFrom(b);
}

/** @brief re-orient all joints (edges) such that n becomes
  the root of the configuration */
void rai::KinematicWorld::reconfigureRootOfSubtree(Frame *root) {
  FrameL pathToOldRoot = root->getPathToRoot();
  
  for(Frame *f : pathToOldRoot) {
    if(f->parent) flipFrames(f->parent, f);
  }
  
  checkConsistency();
}

uint rai::KinematicWorld::analyzeJointStateDimensions() const {
  uint qdim=0;
  for(Joint *j: fwdActiveJoints) {
    if(!j->mimic) {
      j->dim = j->getDimFromType();
      j->qIndex = qdim;
      if(!j->uncertainty)
        qdim += j->qDim();
      else
        qdim += 2*j->qDim();
    } else {
      j->dim = 0;
      j->qIndex = j->mimic->qIndex;
    }
  }
  for(Contact *c: contacts) {
    CHECK_EQ(c->qDim(), 6, "");
    //    c->dim = c->getDimFromType();
    c->qIndex = qdim;
    qdim += c->qDim();
  }
  return qdim;
}

/** @brief returns the joint (actuator) dimensionality */
uint rai::KinematicWorld::getJointStateDimension() const {
  if(!q.nd)((KinematicWorld*)this)->calc_q();
  return q.N;
}

void rai::KinematicWorld::getJointState(arr &_q, arr& _qdot) const {
  if(!q.nd)((KinematicWorld*)this)->calc_q();
  _q=q;
  if(!!_qdot) {
    _qdot=qdot;
    if(!_qdot.N) _qdot.resizeAs(q).setZero();
  }
}

arr rai::KinematicWorld::getJointState() const {
  if(!q.nd)((KinematicWorld*)this)->calc_q();
  return q;
}

arr rai::KinematicWorld::getJointState(const StringA& joints) const {
  if(!q.nd)((KinematicWorld*)this)->calc_q();
  arr x(joints.N);
  for(uint i=0; i<joints.N; i++) {
    String s = joints.elem(i);
    uint d=0;
    if(s(-2)==':') { d=s(-1)-'0'; s.resize(s.N-2,true); }
    Joint *j = getFrameByName(s)->joint;
    CHECK(!j->dim || d<j->dim,"");
    x(i) = q(j->qIndex+d);
  }
  return x;
}

arr rai::KinematicWorld::getJointState(const uintA& joints) const {
  if(!q.nd)((KinematicWorld*)this)->calc_q();
  uint nd=0;
  for(uint i=0; i<joints.N; i++) {
    rai::Joint *j = frames(joints(i))->joint;
    if(!j || !j->active) continue;
    nd += j->dim;
  }

  arr x(nd);
  nd=0;
  for(uint i=0; i<joints.N; i++) {
    rai::Joint *j = frames(joints(i))->joint;
    if(!j || !j->active) continue;
    for(uint ii=0;ii<j->dim;ii++) x(nd+ii) = q(j->qIndex+ii);
    nd += j->dim;
  }
  CHECK_EQ(nd, x.N, "");
  return x;
}

arr rai::KinematicWorld::getFrameState() const{
  arr X(frames.N, 7);
  for(uint i=0; i<X.d0; i++) {
    X[i] = frames(i)->X.getArr7d();
  }
  return X;
}

/** @brief returns the vector of joint limts */
arr rai::KinematicWorld::getLimits() const {
  uint N=getJointStateDimension();
  arr limits(N,2);
  limits.setZero();
  for(Joint *j: fwdActiveJoints) {
    uint i=j->qIndex;
    uint d=j->qDim();
    for(uint k=0; k<d; k++) { //in case joint has multiple dimensions
      if(j->limits.N) {
        limits(i+k,0)=j->limits(0); //lo
        limits(i+k,1)=j->limits(1); //up
      } else {
        limits(i+k,0)=0.; //lo
        limits(i+k,1)=0.; //up
      }
    }
  }
  //  cout <<"limits:" <<limits <<endl;
  return limits;
}

void rai::KinematicWorld::calc_q_from_Q() {
  uint N=q.N;
  if(!N) N=analyzeJointStateDimensions();
  q.resize(N).setZero();
  qdot.resize(N).setZero();
  
  uint n=0;
  for(Joint *j: fwdActiveJoints) {
    if(j->mimic) continue; //don't count dependent joints
    CHECK_EQ(j->qIndex, n, "joint indexing is inconsistent");
    arr joint_q = j->calc_q_from_Q(j->frame->Q);
    CHECK_EQ(joint_q.N, j->dim, "");
    if(!j->dim) continue; //nothing to do
    q.setVectorBlock(joint_q, j->qIndex);
    n += j->dim;
    if(j->uncertainty) {
      q.setVectorBlock(j->uncertainty->sigma, j->qIndex+j->dim);
      n += j->dim;
    }
  }
  for(Contact *c: contacts) {
    CHECK_EQ(c->qIndex, n, "joint indexing is inconsistent");
    arr contact_q = c->calc_q_from_F();
    CHECK_EQ(contact_q.N, c->qDim(), "");
    q.setVectorBlock(contact_q, c->qIndex);
    n += c->qDim();
  }
  CHECK_EQ(n,N,"");
}

void rai::KinematicWorld::calc_Q_from_q() {
  uint n=0;
  for(Joint *j: fwdActiveJoints) {
    if(!j->mimic) CHECK_EQ(j->qIndex, n, "joint indexing is inconsistent");
    j->calc_Q_from_q(q, j->qIndex);
    if(!j->mimic) {
      n += j->dim;
      if(j->uncertainty) {
        j->uncertainty->sigma = q.sub(j->qIndex+j->dim, j->qIndex+2*j->dim-1);
        n += j->dim;
      }
    }
  }
  for(Contact *c: contacts) {
    CHECK_EQ(c->qIndex, n, "joint indexing is inconsistent");
    c->calc_F_from_q(q, c->qIndex);
    n += c->qDim();
  }
  CHECK_EQ(n, q.N, "");
}

void rai::KinematicWorld::selectJointsByGroup(const StringA &groupNames, bool OnlyTheseOrNotThese, bool deleteInsteadOfLock) {
  Joint *j;
  for(Frame *f:frames) if((j=f->joint)) {
    bool select;
    if(OnlyTheseOrNotThese) { //only these
      select=false;
      for(const String& s:groupNames) if(f->ats[s]) { select=true; break; }
    } else {
      select=true;
      for(const String& s:groupNames) if(f->ats[s]) { select=false; break; }
    }
    if(select) f->joint->active=true;
    else  f->joint->active=false;
    //    if(!select) {
    //      if(deleteInsteadOfLock) delete f->joint;
    //      else f->joint->makeRigid();
    //    }
  }
  reset_q();
  checkConsistency();
  calc_q();
  checkConsistency();
}


/// @name active set selection
void rai::KinematicWorld::selectJointsByName(const StringA& names, bool notThose) {
  for(Frame *f: frames) if(f->joint) f->joint->active = notThose;
  for(const String& s:names) {
    Frame *f = getFrameByName(s);
    CHECK(f, "");
    f = f->getUpwardLink();
    CHECK(f->joint, "");
    f->joint->active = !notThose;
  }
  reset_q();
  checkConsistency();
  calc_q();
  checkConsistency();
}

/** @brief sets the joint state vectors separated in positions and
  velocities */
void rai::KinematicWorld::setJointState(const arr& _q, const arr& _qdot) {
  setJointStateCount++; //global counter
  
#ifndef RAI_NOCHECK
  uint N=getJointStateDimension();
  CHECK_EQ(_q.N, N, "wrong joint state dimensionalities");
  if(!!_qdot) CHECK_EQ(_qdot.N, N, "wrong joint velocity dimensionalities");
#endif

  q=_q;
  if(!!_qdot) qdot=_qdot; else qdot.clear();
  
  calc_Q_from_q();
  
  calc_fwdPropagateFrames();
}

void rai::KinematicWorld::setJointState(const arr& _q, const StringA& joints) {
  setJointStateCount++; //global counter
  getJointState();
  
  CHECK_EQ(_q.N, joints.N, "");
  for(uint i=0; i<_q.N; i++) {
    rai::String frameName = joints(i);
    if(frameName(-2)!=':'){ //1-dim joint
      rai::Joint *j = getFrameByName(frameName)->joint;
      CHECK(j, "frame '" <<frameName <<"' is not a joint!");
      q(j->qIndex) = _q(i);
    }else{
      frameName.resize(frameName.N-2, true);
      rai::Joint *j = getFrameByName(frameName)->joint;
      CHECK(j, "frame '" <<frameName <<"' is not a joint!");
      for(uint k=0;k<j->dim;k++) q(j->qIndex+k) = _q(i+k);
      i += j->dim-1;
    }
  }
  qdot.clear();
  
  calc_Q_from_q();
  
  calc_fwdPropagateFrames();
}

void rai::KinematicWorld::setJointState(const arr& _q, const uintA& joints) {
  setJointStateCount++; //global counter
  getJointState();

  uint nd=0;
  for(uint i=0; i<joints.N; i++) {
    rai::Joint *j = frames(joints(i))->joint;
    if(!j || !j->active) continue;
    for(uint ii=0;ii<j->dim;ii++) q(j->qIndex+ii) = _q(nd+ii);
    nd += j->dim;
  }
  CHECK_EQ(_q.N, nd, "");
  qdot.clear();

  calc_Q_from_q();

  calc_fwdPropagateFrames();
}

void rai::KinematicWorld::setFrameState(const arr& X, const StringA& frameNames, bool calc_q_from_X, bool warnOnDifferentDim){
  if(!frameNames.N){
    if(warnOnDifferentDim){
      if(X.d0 > frames.N) LOG(-1) <<"X.d0=" <<X.d0 <<" is larger than frames.N=" <<frames.N;
      if(X.d0 < frames.N) LOG(-1) <<"X.d0=" <<X.d0 <<" is smaller than frames.N=" <<frames.N;
    }
    for(uint i=0;i<frames.N && i<X.d0;i++){
      frames(i)->X.set(X[i]);
      frames(i)->X.rot.normalize();
    }
  }else{
    if(X.nd==1){
      CHECK_EQ(1, frameNames.N, "X.d0 does not equal #frames");
      rai::Frame *f = getFrameByName(frameNames(0));
      if(!f) return;
      f->X.set(X);
      f->X.rot.normalize();
    }else{
      CHECK_EQ(X.d0, frameNames.N, "X.d0 does not equal #frames");
      for(uint i=0;i<X.d0;i++){
        rai::Frame *f = getFrameByName(frameNames(i));
        if(!f) return;
        f->X.set(X[i]);
        f->X.rot.normalize();
      }
    }
  }
  if(calc_q_from_X){
    calc_Q_from_BodyFrames();
    calc_q_from_Q();
  }
}

void rai::KinematicWorld::setTimes(double t) {
  for(Frame *a:frames) a->tau = t;
}

//===========================================================================
//
// features
//

void rai::KinematicWorld::evalFeature(arr& y, arr& J, FeatureSymbol fs, const StringA& symbols) const{
  ptr<Feature> f = symbols2feature(fs, symbols, *this);
  f->__phi(y, J, *this);
}

//===========================================================================
//
// core: kinematics and dynamics
//

/** @brief return the jacobian \f$J = \frac{\partial\phi_i(q)}{\partial q}\f$ of the position
  of the i-th body (3 x n tensor)*/
void rai::KinematicWorld::kinematicsPos(arr& y, arr& J, Frame *a, const rai::Vector& rel) const {
  CHECK_EQ(&a->K, this, "given frame is not element of this KinematicWorld");
  
  if(!a) {
    RAI_MSG("WARNING: calling kinematics for NULL body");
    if(!!y) y.resize(3).setZero();
    if(!!J) J.resize(3, getJointStateDimension()).setZero();
    return;
  }

  //get position
  rai::Vector pos_world = a->X.pos;
  if(!!rel && !rel.isZero) pos_world += a->X.rot*rel;
  if(!!y) y = conv_vec2arr(pos_world); //return the output
  if(!J) return; //do not return the Jacobian
  
  jacobianPos(J, a, pos_world);
}

#if 1
void rai::KinematicWorld::jacobianPos(arr& J, Frame *a, const rai::Vector& pos_world) const {
  CHECK_EQ(&a->K, this, "");

  //get Jacobian
  uint N=getJointStateDimension();
  J.resize(3, N).setZero();
  while(a) { //loop backward down the kinematic tree
    if(!a->parent) break; //frame has no inlink -> done
    Joint *j=a->joint;
    if(j && j->active) {
      uint j_idx=j->qIndex;
      if(j_idx>=N) CHECK_EQ(j->type, JT_rigid, "");
      if(j_idx<N) {
        if(j->type==JT_hingeX || j->type==JT_hingeY || j->type==JT_hingeZ) {
          rai::Vector tmp = j->axis ^ (pos_world-j->X()*j->Q().pos);
          tmp *= j->scale;
          J(0, j_idx) += tmp.x;
          J(1, j_idx) += tmp.y;
          J(2, j_idx) += tmp.z;
        } else if(j->type==JT_transX || j->type==JT_transY || j->type==JT_transZ || j->type==JT_XBall) {
          J(0, j_idx) += j->scale * j->axis.x;
          J(1, j_idx) += j->scale * j->axis.y;
          J(2, j_idx) += j->scale * j->axis.z;
        } else if(j->type==JT_transXY) {
          if(j->mimic) NIY;
          arr R = j->X().rot.getArr();
          R *= j->scale;
          J.setMatrixBlock(R.sub(0,-1,0,1), 0, j_idx);
        } else if(j->type==JT_transXYPhi) {
          if(j->mimic) NIY;
          arr R = j->X().rot.getArr();
          R *= j->scale;
          J.setMatrixBlock(R.sub(0,-1,0,1), 0, j_idx);
          rai::Vector tmp = j->axis ^ (pos_world-(j->X().pos + j->X().rot*a->Q.pos));
          tmp *= j->scale;
          J(0, j_idx+2) += tmp.x;
          J(1, j_idx+2) += tmp.y;
          J(2, j_idx+2) += tmp.z;
        } else if(j->type==JT_phiTransXY) {
          if(j->mimic) NIY;
          rai::Vector tmp = j->axis ^ (pos_world-j->X().pos);
          tmp *= j->scale;
          J(0, j_idx) += tmp.x;
          J(1, j_idx) += tmp.y;
          J(2, j_idx) += tmp.z;
          arr R = (j->X().rot*a->Q.rot).getArr();
          R *= j->scale;
          J.setMatrixBlock(R.sub(0,-1,0,1), 0, j_idx+1);
        }
        if(j->type==JT_XBall) {
          if(j->mimic) NIY;
          arr R = conv_vec2arr(j->X().rot.getX());
          R *= j->scale;
          R.reshape(3,1);
          J.setMatrixBlock(R, 0, j_idx);
        }
        if(j->type==JT_trans3 || j->type==JT_free) {
          if(j->mimic) NIY;
          arr R = j->X().rot.getArr();
          R *= j->scale;
          J.setMatrixBlock(R, 0, j_idx);
        }
        if(j->type==JT_quatBall || j->type==JT_free || j->type==JT_XBall) {
          uint offset = 0;
          if(j->type==JT_XBall) offset=1;
          if(j->type==JT_free) offset=3;
          arr Jrot = j->X().rot.getArr() * a->Q.rot.getJacobian(); //transform w-vectors into world coordinate
          Jrot = crossProduct(Jrot, conv_vec2arr(pos_world-(j->X().pos+j->X().rot*a->Q.pos)));  //cross-product of all 4 w-vectors with lever
          Jrot /= sqrt(sumOfSqr(q({j->qIndex+offset, j->qIndex+offset+3})));   //account for the potential non-normalization of q
          //          for(uint i=0;i<4;i++) for(uint k=0;k<3;k++) J(k,j_idx+offset+i) += Jrot(k,i);
          Jrot *= j->scale;
          J.setMatrixBlock(Jrot, 0, j_idx+offset);
        }
      }
    }
    a = a->parent;
  }
}

#else
void rai::KinematicWorld::jacobianPos(arr& J, Frame *a, const rai::Vector& pos_world) const {
  J.resize(3, getJointStateDimension()).setZero();
  while(a) { //loop backward down the kinematic tree
    if(!a->parent) break; //frame has no inlink -> done
    Joint *j=a->joint;
    if(j && j->active) {
      uint j_idx=j->qIndex;
      arr screw = j->getScrewMatrix();
      for(uint d=0; d<j->dim; d++) {
        rai::Vector axis = screw(0,d, {});
        rai::Vector tmp = axis ^ pos_world;
        J(0, j_idx+d) += tmp.x + screw(1,d,0);
        J(1, j_idx+d) += tmp.y + screw(1,d,1);
        J(2, j_idx+d) += tmp.z + screw(1,d,2);
      }
    }
    a = a->parent;
  }
}
#endif

void rai::KinematicWorld::kinematicsTau(double& tau, arr& J) const {
  Frame *a = frames.first();
  CHECK(a && a->joint && a->joint->type==JT_time, "this configuration does not have a tau DOF");

  Joint *j = a->joint;
  tau = a->tau;
  if(!!J){
    uint N=getJointStateDimension();
    J.resize(1, N).setZero();
    J(0, j->qIndex) += 1e-1;
  }
}

void rai::KinematicWorld::jacobianTime(arr& J, rai::Frame *a) const {
  CHECK_EQ(&a->K, this, "");

  //get Jacobian
  uint N=getJointStateDimension();
  J.resize(1, N).setZero();
  
  while(a) { //loop backward down the kinematic tree
    Joint *j=a->joint;
    if(j && j->active) {
      uint j_idx=j->qIndex;
      if(j_idx>=N) CHECK_EQ(j->type, JT_rigid, "");
      if(j_idx<N) {
        if(j->type==JT_time) {
          J(0, j_idx) += 1e-1;
        }
      }
    }
    if(!a->parent) break; //frame has no inlink -> done
    a = a->parent;
  }
}

/** @brief return the jacobian \f$J = \frac{\partial\phi_i(q)}{\partial q}\f$ of the position
  of the i-th body W.R.T. the 6 axes of an arbitrary shape-frame, NOT the robot's joints (3 x 6 tensor)
  WARNING: this does not check if s is actually in the kinematic chain from root to b.
*/
void rai::KinematicWorld::kinematicsPos_wrtFrame(arr& y, arr& J, Frame *b, const rai::Vector& rel, Frame *s) const {
  if(!b && !!J) { J.resize(3, getJointStateDimension()).setZero();  return; }
  
  //get position
  rai::Vector pos_world = b->X.pos;
  if(!!rel) pos_world += b->X.rot*rel;
  if(!!y) y = conv_vec2arr(pos_world); //return the output
  if(!J) return; //do not return the Jacobian
  
  //get Jacobian
  J.resize(3, 6).setZero();
  rai::Vector diff = pos_world - s->X.pos;
  rai::Array<rai::Vector> axes = {s->X.rot.getX(), s->X.rot.getY(), s->X.rot.getZ()};
  
  //3 translational axes
  for(uint i=0; i<3; i++) {
    J(0, i) += axes(i).x;
    J(1, i) += axes(i).y;
    J(2, i) += axes(i).z;
  }
  
  //3 rotational axes
  for(uint i=0; i<3; i++) {
    rai::Vector tmp = axes(i) ^ diff;
    J(0, 3+i) += tmp.x;
    J(1, 3+i) += tmp.y;
    J(2, 3+i) += tmp.z;
  }
}

/** @brief return the Hessian \f$H = \frac{\partial^2\phi_i(q)}{\partial q\partial q}\f$ of the position
  of the i-th body (3 x n x n tensor) */
void rai::KinematicWorld::hessianPos(arr& H, Frame *a, rai::Vector *rel) const {
  HALT("this is buggy: a sign error: see examples/Kin/ors testKinematics");
  Joint *j1, *j2;
  uint j1_idx, j2_idx;
  rai::Vector tmp, pos_a;
  
  uint N=getJointStateDimension();
  
  //initialize Jacobian
  H.resize(3, N, N);
  H.setZero();
  
  //get reference frame
  pos_a = a->X.pos;
  if(rel) pos_a += a->X.rot*(*rel);
  
  if((j1=a->joint)) {
    while(j1) {
      j1_idx=j1->qIndex;
      
      j2=j1;
      while(j2) {
        j2_idx=j2->qIndex;
        
        if(j1->type>=JT_hingeX && j1->type<=JT_hingeZ && j2->type>=JT_hingeX && j2->type<=JT_hingeZ) { //both are hinges
          tmp = j2->axis ^ (j1->axis ^ (pos_a-j1->X().pos));
          H(0, j1_idx, j2_idx) = H(0, j2_idx, j1_idx) = tmp.x;
          H(1, j1_idx, j2_idx) = H(1, j2_idx, j1_idx) = tmp.y;
          H(2, j1_idx, j2_idx) = H(2, j2_idx, j1_idx) = tmp.z;
        } else if(j1->type>=JT_transX && j1->type<=JT_transZ && j2->type>=JT_hingeX && j2->type<=JT_hingeZ) { //i=trans, j=hinge
          tmp = j1->axis ^ j2->axis;
          H(0, j1_idx, j2_idx) = H(0, j2_idx, j1_idx) = tmp.x;
          H(1, j1_idx, j2_idx) = H(1, j2_idx, j1_idx) = tmp.y;
          H(2, j1_idx, j2_idx) = H(2, j2_idx, j1_idx) = tmp.z;
        } else if(j1->type==JT_transXY && j2->type>=JT_hingeX && j2->type<=JT_hingeZ) { //i=trans3, j=hinge
          NIY;
        } else if(j1->type==JT_transXYPhi && j1->type==JT_phiTransXY && j2->type>=JT_hingeX && j2->type<=JT_hingeZ) { //i=trans3, j=hinge
          NIY;
        } else if(j1->type==JT_trans3 && j2->type>=JT_hingeX && j2->type<=JT_hingeZ) { //i=trans3, j=hinge
          Matrix R,A;
          j1->X().rot.getMatrix(R.p());
          A.setSkew(j2->axis);
          R = R*A;
          H(0, j1_idx  , j2_idx) = H(0, j2_idx  , j1_idx) = R.m00;
          H(1, j1_idx  , j2_idx) = H(1, j2_idx  , j1_idx) = R.m10;
          H(2, j1_idx  , j2_idx) = H(2, j2_idx  , j1_idx) = R.m20;
          H(0, j1_idx+1, j2_idx) = H(0, j2_idx, j1_idx+1) = R.m01;
          H(1, j1_idx+1, j2_idx) = H(1, j2_idx, j1_idx+1) = R.m11;
          H(2, j1_idx+1, j2_idx) = H(2, j2_idx, j1_idx+1) = R.m21;
          H(0, j1_idx+2, j2_idx) = H(0, j2_idx, j1_idx+2) = R.m02;
          H(1, j1_idx+2, j2_idx) = H(1, j2_idx, j1_idx+2) = R.m12;
          H(2, j1_idx+2, j2_idx) = H(2, j2_idx, j1_idx+2) = R.m22;
        } else if(j1->type>=JT_hingeX && j1->type<=JT_hingeZ && j2->type>=JT_transX && j2->type<=JT_trans3) { //i=hinge, j=trans
          //nothing! Hessian is zero (ej is closer to root than ei)
        } else NIY;
        
        j2=j2->from()->joint;
        if(!j2) break;
      }
      j1=j1->from()->joint;
      if(!j1) break;
    }
  }
}

/* takes the joint state x and returns the jacobian dz of
   the position of the ith body (w.r.t. all joints) -> 2D array */
/// Jacobian of the i-th body's z-orientation vector
void rai::KinematicWorld::kinematicsVec(arr& y, arr& J, Frame *a, const rai::Vector& vec) const {
  CHECK_EQ(&a->K, this, "");
  //get the vectoreference frame
  rai::Vector vec_world;
  if(!!vec) vec_world = a->X.rot*vec;
  else     vec_world = a->X.rot.getZ();
  if(!!y) y = conv_vec2arr(vec_world); //return the vec
  if(!!J) {
    arr A;
    axesMatrix(A, a);
    J = crossProduct(A, conv_vec2arr(vec_world));
  }
}

/* takes the joint state x and returns the jacobian dz of
   the position of the ith body (w.r.t. all joints) -> 2D array */
/// Jacobian of the i-th body's z-orientation vector
void rai::KinematicWorld::kinematicsQuat(arr& y, arr& J, Frame *a) const { //TODO: allow for relative quat
  CHECK_EQ(&a->K, this, "");
  rai::Quaternion rot_a = a->X.rot;
  if(!!y) y = conv_quat2arr(rot_a); //return the vec
  if(!!J) {
    arr A;
    axesMatrix(A, a);
    J.resize(4, A.d1);
    for(uint i=0; i<J.d1; i++) {
      rai::Quaternion tmp(0., 0.5*A(0,i), 0.5*A(1,i), 0.5*A(2,i));  //this is unnormalized!!
      tmp = tmp * rot_a;
      J(0, i) = tmp.w;
      J(1, i) = tmp.x;
      J(2, i) = tmp.y;
      J(3, i) = tmp.z;
    }
  }
}

////* This Jacobian directly gives the implied rotation vector: multiplied with \dot q it gives the angular velocity of body b */
//void rai::KinematicWorld::posMatrix(arr& J, Frame *a) const {
//  uint N = getJointStateDimension();
//  J.resize(3, N).setZero();

//  while(a) { //loop backward down the kinematic tree
//    Joint *j=a->joint;
//    if(j && j->active) {
//      uint j_idx=j->qIndex;
//      if(j_idx>=N) CHECK_EQ(j->type, JT_rigid, "");
//      if(j_idx<N){
//        J(0, j_idx) += j->X().pos.x;
//        J(1, j_idx) += j->X().pos.y;
//        J(2, j_idx) += j->X().pos.z;
//        }
//      }
//    }
//    a = a->parent;
//  }
//}

//* This Jacobian directly gives the implied rotation vector: multiplied with \dot q it gives the angular velocity of body b */
void rai::KinematicWorld::axesMatrix(arr& J, Frame *a) const {
  uint N = getJointStateDimension();
  J.resize(3, N).setZero();
  
  while(a) { //loop backward down the kinematic tree
    Joint *j=a->joint;
    if(j && j->active) {
      uint j_idx=j->qIndex;
      if(j_idx>=N) CHECK_EQ(j->type, JT_rigid, "");
      if(j_idx<N) {
        if((j->type>=JT_hingeX && j->type<=JT_hingeZ) || j->type==JT_transXYPhi || j->type==JT_phiTransXY) {
          if(j->type==JT_transXYPhi) j_idx += 2; //refer to the phi only
          J(0, j_idx) += j->scale * j->axis.x;
          J(1, j_idx) += j->scale * j->axis.y;
          J(2, j_idx) += j->scale * j->axis.z;
        }
        if(j->type==JT_quatBall || j->type==JT_free || j->type==JT_XBall) {
          uint offset = 0;
          if(j->type==JT_XBall) offset=1;
          if(j->type==JT_free) offset=3;
          arr Jrot = j->X().rot.getArr() * a->Q.rot.getJacobian(); //transform w-vectors into world coordinate
          Jrot /= sqrt(sumOfSqr(q({j->qIndex+offset,j->qIndex+offset+3}))); //account for the potential non-normalization of q
          //          for(uint i=0;i<4;i++) for(uint k=0;k<3;k++) J(k,j_idx+offset+i) += Jrot(k,i);
          Jrot *= j->scale;
          J.setMatrixBlock(Jrot, 0, j_idx+offset);
        }
        //all other joints: J=0 !!
      }
    }
    a = a->parent;
  }
}

/// The position vec1, attached to b1, relative to the frame of b2 (plus vec2)
void rai::KinematicWorld::kinematicsRelPos(arr& y, arr& J, Frame *a, const rai::Vector& vec1, Frame *b, const rai::Vector& vec2) const {
  arr y1,y2,J1,J2;
  kinematicsPos(y1, J1, a, vec1);
  kinematicsPos(y2, J2, b, vec2);
  arr Rinv = ~(b->X.rot.getArr());
  y = Rinv * (y1 - y2);
  if(!!J) {
    arr A;
    axesMatrix(A, b);
    J = Rinv * (J1 - J2 - crossProduct(A, y1 - y2));
  }
}

/// The vector vec1, attached to b1, relative to the frame of b2
void rai::KinematicWorld::kinematicsRelVec(arr& y, arr& J, Frame *a, const rai::Vector& vec1, Frame *b) const {
  arr y1,J1;
  kinematicsVec(y1, J1, a, vec1);
  //  kinematicsVec(y2, J2, b2, vec2);
  arr Rinv = ~(b->X.rot.getArr());
  y = Rinv * y1;
  if(!!J) {
    arr A;
    axesMatrix(A, b);
    J = Rinv * (J1 - crossProduct(A, y1));
  }
}

/// The position vec1, attached to b1, relative to the frame of b2 (plus vec2)
void rai::KinematicWorld::kinematicsRelRot(arr& y, arr& J, Frame *a, Frame *b) const {
  rai::Quaternion rot_b = a->X.rot;
  if(!!y) y = conv_vec2arr(rot_b.getVec());
  if(!!J) {
    double phi=acos(rot_b.w);
    double s=2.*phi/sin(phi);
    double ss=-2./(1.-rai::sqr(rot_b.w)) * (1.-phi/tan(phi));
    arr A;
    axesMatrix(A, a);
    J = 0.5 * (rot_b.w*A*s + crossProduct(A, y));
    J -= 0.5 * ss/s/s*(y*~y*A);
  }
}

void rai::KinematicWorld::kinematicsContactPOA(arr& y, arr& J, rai::Contact *c) const{
  y = c->position;
  if(!!J){
    J = zeros(3, q.N);
    for(uint i=0;i<3;i++) J(i, c->qIndex+i) = 1.;
  }
}

void rai::KinematicWorld::kinematicsContactForce(arr& y, arr& J, rai::Contact *c) const{
  y = c->force;
  if(!!J){
    J = zeros(3, q.N);
    for(uint i=0;i<3;i++) J(i, c->qIndex+3+i) = 1.;
  }
}

/** @brief return the configuration's inertia tensor $M$ (n x n tensor)*/
void rai::KinematicWorld::inertia(arr& M) {
  uint j1_idx, j2_idx;
  rai::Transformation Xa, Xi, Xj;
  Joint *j1, *j2;
  rai::Vector vi, vj, ti, tj;
  double tmp;
  
  uint N=getJointStateDimension();
  
  //initialize Jacobian
  M.resize(N, N);
  M.setZero();
  
  for(Frame *a: frames) {
    //get reference frame
    Xa = a->X;
    
    j1=a->joint;
    while(j1) {
      j1_idx=j1->qIndex;
      
      Xi = j1->from()->X;
      //      Xi.appendTransformation(j1->A);
      ti = Xi.rot.getX();
      
      vi = ti ^(Xa.pos-Xi.pos);
      
      j2=j1;
      while(j2) {
        j2_idx=j2->qIndex;
        
        Xj = j2->from()->X;
        //        Xj.appendTransformation(j2->A);
        tj = Xj.rot.getX();
        
        vj = tj ^(Xa.pos-Xj.pos);
        
        tmp = a->inertia->mass * (vi*vj);
        //tmp += scalarProduct(a->a.inertia, ti, tj);
        
        M(j1_idx, j2_idx) += tmp;
        
        j2=j2->from()->joint;
        if(!j2) break;
      }
      j1=j1->from()->joint;
      if(!j1) break;
    }
  }
  //symmetric: fill in other half
  for(j1_idx=0; j1_idx<N; j1_idx++) for(j2_idx=0; j2_idx<j1_idx; j2_idx++) M(j2_idx, j1_idx) = M(j1_idx, j2_idx);
}

void rai::KinematicWorld::equationOfMotion(arr& M, arr& F, bool gravity) {
  if(gravity) {
    clearForces();
    gravityToForces();
  }
  fs().update();
  //  cout <<tree <<endl;
  if(!qdot.N) qdot.resize(q.N).setZero();
  fs().equationOfMotion(M, F, qdot);
}

/** @brief return the joint accelerations \f$\ddot q\f$ given the
  joint torques \f$\tau\f$ (computed via Featherstone's Articulated Body Algorithm in O(n)) */
void rai::KinematicWorld::fwdDynamics(arr& qdd, const arr& qd, const arr& tau, bool gravity) {
  if(gravity) {
    clearForces();
    gravityToForces();
  }
  fs().update();
  //  cout <<tree <<endl;
  fs().fwdDynamics_MF(qdd, qd, tau);
  //  fs().fwdDynamics_aba_1D(qdd, qd, tau); //works
  //  rai::fwdDynamics_aba_nD(qdd, tree, qd, tau); //does not work
}

/** @brief return the necessary joint torques \f$\tau\f$ to achieve joint accelerations
  \f$\ddot q\f$ (computed via the Recursive Newton-Euler Algorithm in O(n)) */
void rai::KinematicWorld::inverseDynamics(arr& tau, const arr& qd, const arr& qdd, bool gravity) {
  if(gravity) {
    clearForces();
    gravityToForces();
  }
  fs().update();
  fs().invDynamics(tau, qd, qdd);
}

/*void rai::KinematicWorld::impulsePropagation(arr& qd1, const arr& qd0){
  static rai::Array<Featherstone::Link> tree;
  if(!tree.N) GraphToTree(tree, *this);
  else updateGraphToTree(tree, *this);
  mimickImpulsePropagation(tree);
  Featherstone::RF_abd(qdd, tree, qd, tau);
}*/

/** @brief checks if all names of the bodies are disjoint */
bool rai::KinematicWorld::checkUniqueNames() const {
  for(Frame *a:  frames) for(Frame *b: frames) {
    if(a==b) break;
    if(a->name==b->name) return false;
  }
  return true;
}

/// find body with specific name
rai::Frame* rai::KinematicWorld::getFrameByName(const char* name, bool warnIfNotExist, bool reverse) const {
  if(!reverse){
    for(Frame *b: frames) if(b->name==name) return b;
  }else{
    for(uint i=frames.N;i--;) if(frames(i)->name==name) return frames(i);
  }
  if(strcmp("glCamera", name)!=0)
    if(warnIfNotExist) RAI_MSG("cannot find Body named '" <<name <<"' in Graph");
  return 0;
}

FrameL rai::KinematicWorld::getFramesByNames(const StringA& frameNames) const{
  FrameL F;
  for(const rai::String& name:frameNames) F.append(getFrameByName(name), true);
  return F;
}

///// find shape with specific name
//rai::Shape* rai::KinematicWorld::getShapeByName(const char* name, bool warnIfNotExist) const {
//  Frame *f = getFrameByName(name, warnIfNotExist);
//  return f->shape;
//}

///// find shape with specific name
//rai::Joint* rai::KinematicWorld::getJointByName(const char* name, bool warnIfNotExist) const {
//  Frame *f = getFrameByName(name, warnIfNotExist);
//  return f->joint();
//}

/// find joint connecting two bodies
//rai::Link* rai::KinematicWorld::getLinkByBodies(const Frame* from, const Frame* to) const {
//  if(to->link && to->link->from==from) return to->link;
//  return NULL;
//}

/// find joint connecting two bodies
rai::Joint* rai::KinematicWorld::getJointByBodies(const Frame* from, const Frame* to) const {
  if(to->joint && to->parent==from) return to->joint;
  return NULL;
}

/// find joint connecting two bodies with specific names
rai::Joint* rai::KinematicWorld::getJointByBodyNames(const char* from, const char* to) const {
  Frame *f = getFrameByName(from);
  Frame *t = getFrameByName(to);
  if(!f || !t) return NULL;
  return getJointByBodies(f, t);
}

/// find joint connecting two bodies with specific names
rai::Joint* rai::KinematicWorld::getJointByBodyIndices(uint ifrom, uint ito) const {
  if(ifrom>=frames.N || ito>=frames.N) return NULL;
  Frame *f = frames(ifrom);
  Frame *t = frames(ito);
  return getJointByBodies(f, t);
}

uintA rai::KinematicWorld::getQindicesByNames(const StringA& jointNames) const{
  FrameL F = getFramesByNames(jointNames);
  uintA Qidx;
  for(rai::Frame* f: F){
    CHECK(f->joint, "");
    Qidx.append(f->joint->qIndex);
  }
  return Qidx;
}

StringA rai::KinematicWorld::getJointNames() const {
  if(!q.nd)((KinematicWorld*)this)->calc_q();
  StringA names(getJointStateDimension());
  for(Joint *j:fwdActiveJoints) {
    rai::String name=j->frame->name;
    if(!name) name <<'q' <<j->qIndex;
    if(j->dim==1) names(j->qIndex) <<name;
    else for(uint i=0; i<j->dim; i++) names(j->qIndex+i) <<name <<':' <<i;
    
    if(j->uncertainty) {
      if(j->dim) {
        for(uint i=j->dim; i<2*j->dim; i++) names(j->qIndex+i) <<name <<":UC:" <<i;
      } else {
        names(j->qIndex+1) <<name <<":UC";
      }
    }
  }
  return names;
}

StringA rai::KinematicWorld::getFrameNames() const {
  StringA names(frames.N);
  for(uint i=0;i<frames.N;i++) {
    names(i) = frames(i)->name;
  }
  return names;
}

/** @brief creates uniques names by prefixing the node-index-number to each name */
void rai::KinematicWorld::prefixNames(bool clear) {
  if(!clear) for(Frame *a: frames) a->name=STRING('_' <<a->ID <<'_' <<a->name);
  else       for(Frame *a: frames) a->name.clear() <<a->ID;
}

/// return a OpenGL extension
OpenGL& rai::KinematicWorld::gl(const char* window_title) {
  if(!s->gl) {
    s->gl = new OpenGL(window_title);
    s->gl->add(glStandardScene, 0);
    s->gl->addDrawer(this);
    s->gl->camera.setDefault();
  }
  return *s->gl;
}

/// return a Swift extension
SwiftInterface& rai::KinematicWorld::swift() {
  if(!s->swift) s->swift = make_shared<SwiftInterface>(*this, .1);
  return *s->swift;
}

rai::FclInterface& rai::KinematicWorld::fcl(){
  if(!s->fcl){
    Array<ptr<Mesh>> geometries(frames.N);
    for(Frame *f:frames){
      if(f->shape && f->shape->cont){
        if(!f->shape->mesh().V.N) f->shape->createMeshes();
        geometries(f->ID) = f->shape->_mesh;
      }
    }
    s->fcl = make_shared<rai::FclInterface>(geometries, .0);
  }
  return *s->fcl;
}

void rai::KinematicWorld::swiftDelete() {
  s->swift.reset();
}

/// return a PhysX extension
PhysXInterface& rai::KinematicWorld::physx() {
  if(!s->physx) {
    s->physx = new PhysXInterface(*this);
    //    s->physx->setArticulatedBodiesKinematic();
  }
  return *s->physx;
}

/// return a ODE extension
OdeInterface& rai::KinematicWorld::ode() {
  if(!s->ode) s->ode = new OdeInterface(*this);
  return *s->ode;
}

FeatherstoneInterface& rai::KinematicWorld::fs() {
  if(!s->fs) s->fs = new FeatherstoneInterface(*this);
  return *s->fs;
}

int rai::KinematicWorld::watch(bool pause, const char* txt) {
//  gl().pressedkey=0;
  int key;
  if(pause){
    if(!txt) txt="Config::watch";
    key = gl().watch(txt);
  }else{
    key = gl().update(txt, true);
  }
  return key;
}

void rai::KinematicWorld::saveVideoPic(uint& t, const char* pathPrefix){
  write_ppm(gl().captureImage, STRING(pathPrefix <<std::setw(4)<<std::setfill('0')<<t++<<".ppm"));
}

void rai::KinematicWorld::glAdd(void (*call)(void*,OpenGL&), void* classP){
  gl().add(call, classP);
}

int rai::KinematicWorld::glAnimate() {
  return animateConfiguration(*this, NULL);
}

void rai::KinematicWorld::glClose(){
  if(s->gl){ delete s->gl; s->gl=0; }
}

void rai::KinematicWorld::glGetMasks(int w, int h, bool rgbIndices) {
  gl().clear();
  gl().addDrawer(this);
  if(rgbIndices) {
    gl().setClearColors(0,0,0,0);
    orsDrawIndexColors = true;
    orsDrawMarkers = orsDrawJoints = orsDrawProxies = false;
  }
  gl().renderInBack(w, h);
  //  indexRgb = gl().captureImage;
  //  depth = gl().captureDepth;

  gl().clear();
  gl().add(glStandardScene, 0);
  gl().addDrawer(this);
  if(rgbIndices) {
    gl().setClearColors(1,1,1,0);
    orsDrawIndexColors = false;
    orsDrawMarkers = orsDrawJoints = orsDrawProxies = true;
  }
}

void rai::KinematicWorld::stepSwift() {
  swift().step(*this, false);
  //  reportProxies();
  //  watch(true);
  //  gl().closeWindow();
}

void rai::KinematicWorld::stepFcl(){
  arr X(frames.N, 7);
  X.setZero();
  for(Frame *f:frames){
    if(f->shape && f->shape->cont) X[f->ID] = f->X.getArr7d();
  }
  fcl().step(X);
  uintA& COL = fcl().collisions;
  boolA filter(COL.d0);
  uint n=0;
  for(uint i=0;i<COL.d0;i++){
    bool canCollide = frames(COL(i,0))->shape->canCollideWith(frames(COL(i,1)));
    filter(i) = canCollide;
    if(canCollide) n++;
  }
  proxies.clear();
  proxies.resize(n);
  for(uint i=0,j=0;i<COL.d0;i++){
    if(filter(i)){
      Proxy& p = proxies(j);
      p.a = frames(COL(i,0));
      p.b = frames(COL(i,1));
      p.d = -0.;
      p.posA = frames(COL(i,0))->shape->mesh().getCenter();
      p.posB = frames(COL(i,1))->shape->mesh().getCenter();
      j++;
    }
  }
}

void rai::KinematicWorld::stepPhysx(double tau) {
  physx().step(tau);
}

void rai::KinematicWorld::stepOde(double tau) {
#ifdef RAI_ODE
  ode().setMotorVel(qdot, 100.);
  ode().step(tau);
  ode().importStateFromOde();
#endif
}

void rai::KinematicWorld::stepDynamics(const arr& Bu_control, double tau, double dynamicNoise, bool gravity) {

  struct DiffEqn:VectorFunction {
    rai::KinematicWorld &S;
    const arr& Bu;
    bool gravity;
    DiffEqn(rai::KinematicWorld& _S, const arr& _Bu, bool _gravity):S(_S), Bu(_Bu), gravity(_gravity) {
      VectorFunction::operator=([this](arr& y, arr& J, const arr& x) -> void {
        this->fv(y, J, x);
      });
    }
    void fv(arr& y, arr& J, const arr& x) {
      S.setJointState(x[0], x[1]);
      arr M,Minv,F;
      S.equationOfMotion(M, F, gravity);
      inverse_SymPosDef(Minv, M);
      //Minv = inverse(M); //TODO why does symPosDef fail?
      y = Minv * (Bu - F);
    }
  } eqn(*this, Bu_control, gravity);
  
#if 0
  arr M,Minv,F;
  getDynamics(M, F);
  inverse_SymPosDef(Minv,M);
  
  //noisy Euler integration (Runge-Kutte4 would be much more precise...)
  qddot = Minv * (u_control - F);
  if(dynamicNoise) rndGauss(qddot, dynamicNoise, true);
  q    += tau * qdot;
  qdot += tau * qddot;
  arr x1=cat(s->q, s->qdot).reshape(2,s->q.N);
#else
  arr x1;
  rk4_2ndOrder(x1, cat(q, qdot).reshape(2,q.N), eqn, tau);
  if(dynamicNoise) rndGauss(x1[1](), ::sqrt(tau)*dynamicNoise, true);
#endif
  
  setJointState(x1[0], x1[1]);
}

void __merge(rai::Contact *c, rai::Proxy *p) {
  CHECK(&c->a==p->a && &c->b==p->b, "");
  if(!p->coll) p->calc_coll(c->a.K);
  //  c->a_rel = c->a.X / rai::Vector(p->coll->p1);
  //  c->b_rel = c->b.X / rai::Vector(p->coll->p2);
  //  c->a_norm = c->a.X.rot / rai::Vector(-p->coll->normal);
  //  c->b_norm = c->b.X.rot / rai::Vector(p->coll->normal);
  //  c->a_type = c->b_type=1;
  //  c->a_rad = c->a.shape->size(3);
  //  c->b_rad = c->b.shape->size(3);
}

#if 0
double __matchingCost(rai::Contact *c, rai::Proxy *p) {
  double cost=0.;
  if(!p->coll) p->calc_coll(c->a.K);
  cost += sqrDistance(c->a.X * c->a_rel, p->coll->p1);
  cost += sqrDistance(c->b.X * c->b_rel, p->coll->p2);
  //normal costs? Perhaps not, if p does not have a proper normal!
  return cost;
}

void __new(rai::KinematicWorld& K, rai::Proxy *p) {
  rai::Contact *c = new rai::Contact(*p->a, *p->b);
  __merge(c, p);
}

void rai::KinematicWorld::filterProxiesToContacts(double margin) {
  for(Proxy& p:proxies) {
    if(!p.coll) p.calc_coll(*this);
    if(p.coll->distance-(p.coll->rad1+p.coll->rad2)>margin) continue;
    Contact *candidate=NULL;
    double candidateMatchingCost=0.;
    for(Contact *c:p.a->contacts) {
      if((&c->a==p.a && &c->b==p.b) || (&c->a==p.b && &c->b==p.a)) {
        double cost = __matchingCost(c, &p);
        if(!candidate || cost<candidateMatchingCost) {
          candidate = c;
          candidateMatchingCost = cost;
        }
        //in any case: adapt the normals:
        c->a_norm = c->a.X.rot / rai::Vector(-p.coll->normal);
        c->b_norm = c->b.X.rot / rai::Vector(p.coll->normal);
      }
    }
    if(candidate && ::sqrt(candidateMatchingCost)<.05) { //cost is roughly measured in sqr-meters
      __merge(candidate, &p);
    } else {
      __new(*this, &p);
    }
  }
  //phase 2: cleanup old and distant contacts
  rai::Array<Contact*> old;
  for(Frame *f:frames) for(Contact *c:f->contacts) if(&c->a==f) {
    if(/*c->get_pDistance()>margin+.05 ||*/ c->getDistance()>margin) old.append(c);
  }
  for(Contact *c:old) delete c;
}
#endif

void rai::KinematicWorld::proxiesToContacts(double margin) {
  for(Frame *f:frames) while(f->contacts.N) delete f->contacts.last();
  
  for(Proxy& p:proxies) {
    if(!p.coll) p.calc_coll(*this);
    Contact *candidate=NULL;
    for(Contact *c:p.a->contacts) {
      if((&c->a==p.a && &c->b==p.b) || (&c->a==p.b && &c->b==p.a)) {
        candidate = c;
        break;
      }
    }
    if(candidate) __merge(candidate, &p);
    else {
      if(p.coll->distance-(p.coll->rad1+p.coll->rad2)<margin) {
        rai::Contact *c = new rai::Contact(*p.a, *p.b);
        __merge(c, &p);
      }
    }
  }
  //phase 2: cleanup old and distant contacts
  NIY;
  //  rai::Array<Contact*> old;
  //  for(Frame *f:frames) for(Contact *c:f->contacts) if(&c->a==f) {
  //    if(c->getDistance()>2.*margin) {
  //      old.append(c);
  //    }
  //  }
  //  for(Contact *c:old) delete c;
}

double rai::KinematicWorld::totalContactPenetration() {
  double D=0.;
  for(const Proxy& p:proxies) {
    //early check: if swift is way out of collision, don't bother computing it precise
    if(p.d > p.a->shape->radius()+p.a->shape->radius()+.01) continue;
    //exact computation
    if(!p.coll) ((Proxy*)&p)->calc_coll(*this);
    double d = p.coll->getDistance();
    if(d<0.) D -= d;
  }
  //  for(Frame *f:frames) for(Contact *c:f->contacts) if(&c->a==f) {
  //        double d = c->getDistance();
  //      }
  return D;
}

void rai::KinematicWorld::copyProxies(const rai::KinematicWorld& K){
  proxies.resize(K.proxies.N);
  for(uint i=0;i<proxies.N;i++) proxies(i).copy(*this, K.proxies(i));
}

/** @brief prototype for \c operator<< */
void rai::KinematicWorld::write(std::ostream& os) const {
  for(Frame *f: frames) if(!f->name.N) f->name <<'_' <<f->ID;
  for(Frame *f: frames) { //fwdActiveSet) {
    //    os <<"frame " <<f->name;
    //    if(f->parent) os <<'(' <<f->parent->name <<')';
    //    os <<" \t{ ";
    f->write(os);
    //    os <<" }\n";
  }
  os <<std::endl;
  //  for(Frame *f: frames) if(f->shape){
  //    os <<"shape ";
  //    os <<"(" <<f->name <<"){ ";
  //    f->shape->write(os);  os <<" }\n";
  //  }
  //  os <<std::endl;
  //  for(Frame *f: fwdActiveSet) if(f->parent) {
  //    if(f->joint){
  //      os <<"joint ";
  //      os <<"(" <<f->parent->name <<' ' <<f->name <<"){ ";
  //      f->joint->write(os);  os <<" }\n";
  //    }else{
  //      os <<"link ";
  //      os <<"(" <<f->parent->name <<' ' <<f->name <<"){ ";
  //      f->write(os);  os <<" }\n";
  //    }
  //  }
}

void rai::KinematicWorld::write(Graph& G) const {
  for(Frame *f: frames) if(!f->name.N) f->name <<'_' <<f->ID;
  for(Frame *f: frames) f->write(G.newSubgraph({f->name}));
}

void rai::KinematicWorld::writeURDF(std::ostream &os, const char* robotName) const {
  os <<"<?xml version=\"1.0\"?>\n";
  os <<"<robot name=\"" <<robotName <<"\">\n";
  
  //-- write base_link first
  
  FrameL bases;
  for(Frame *a:frames) { if(!a->parent) a->getRigidSubFrames(bases); }
  os <<"<link name=\"base_link\">\n";
  for(Frame *a:frames) {
    if(a->shape && a->shape->type()!=ST_mesh && a->shape->type()!=ST_marker) {
      os <<"  <visual>\n    <geometry>\n";
      arr& size = a->shape->size();
      switch(a->shape->type()) {
        case ST_box:       os <<"      <box size=\"" <<size({0,2}) <<"\" />\n";  break;
        case ST_cylinder:  os <<"      <cylinder length=\"" <<size.elem(-2) <<"\" radius=\"" <<size.elem(-1) <<"\" />\n";  break;
        case ST_sphere:    os <<"      <sphere radius=\"" <<size.last() <<"\" />\n";  break;
        case ST_mesh:      os <<"      <mesh filename=\"" <<a->ats.get<rai::FileToken>("mesh").name <<'"';
          if(a->ats["meshscale"]) os <<" scale=\"" <<a->ats.get<arr>("meshscale") <<'"';
          os <<" />\n";  break;
        default:           os <<"      <UNKNOWN_" <<a->shape->type() <<" />\n";  break;
      }
      os <<"      <material> <color rgba=\"" <<a->shape->mesh().C <<"\" /> </material>\n";
      os <<"    </geometry>\n";
      //      os <<"  <origin xyz=\"" <<a->Q.pos.x <<' ' <<a->Q.pos.y <<' ' <<a->Q.pos.z <<"\" />\n";
      os <<"  <inertial>  <mass value=\"1\"/>  </inertial>\n";
      os <<"  </visual>\n";
    }
  }
  os <<"</link>" <<endl;
  
  for(Frame* a:frames) if(a->joint) {
    os <<"<link name=\"" <<a->name <<"\">\n";

    FrameL shapes;
    a->getRigidSubFrames(shapes);
    for(Frame *b:shapes) {
      if(b->shape && b->shape->type()!=ST_mesh && b->shape->type()!=ST_marker) {
        os <<"  <visual>\n    <geometry>\n";
        arr& size = b->shape->size();
        switch(b->shape->type()) {
          case ST_box:       os <<"      <box size=\"" <<size({0,2}) <<"\" />\n";  break;
          case ST_cylinder:  os <<"      <cylinder length=\"" <<size.elem(-2) <<"\" radius=\"" <<size.elem(-1) <<"\" />\n";  break;
          case ST_sphere:    os <<"      <sphere radius=\"" <<size.last() <<"\" />\n";  break;
          case ST_mesh:      os <<"      <mesh filename=\"" <<b->ats.get<rai::FileToken>("mesh").name <<'"';
            if(b->ats["meshscale"]) os <<" scale=\"" <<b->ats.get<arr>("meshscale") <<'"';
            os <<" />\n";  break;
          default:           os <<"      <UNKNOWN_" <<b->shape->type() <<" />\n";  break;
        }
        os <<"      <material> <color rgba=\"" <<b->shape->mesh().C <<"\" /> </material>\n";
        os <<"    </geometry>\n";
        os <<"  <origin xyz=\"" <<b->Q.pos.getArr() <<"\" rpy=\"" <<b->Q.rot.getEulerRPY() <<"\" />\n";
        os <<"  <inertial>  <mass value=\"1\"/>  </inertial>\n";
        os <<"  </visual>\n";
      }
    }
    os <<"</link>" <<endl;

    os <<"<joint name=\"" <<a->name <<"\" type=\"fixed\" >\n";
    rai::Transformation Q=0;
    Frame *p=a->parent;
    while(p && !p->joint) { Q=p->Q*Q; p=p->parent; }
    if(!p)    os <<"  <parent link=\"base_link\"/>\n";
    else      os <<"  <parent link=\"" <<p->name <<"\"/>\n";
    os <<"  <child  link=\"" <<a->name <<"\"/>\n";
    os <<"  <origin xyz=\"" <<Q.pos.getArr() <<"\" rpy=\"" <<Q.rot.getEulerRPY() <<"\" />\n";
    os <<"</joint>" <<endl;
  }

  os <<"</robot>";
}

void rai::KinematicWorld::writeMeshes(const char *pathPrefix) const {
  for(rai::Frame *f:frames) {
    if(f->shape &&
       (f->shape->type()==rai::ST_mesh || f->shape->type()==rai::ST_ssCvx)) {
      rai::String filename = pathPrefix;
      filename <<f->name <<".arr";
      f->ats.getNew<rai::String>("mesh") = filename;
      if(f->shape->type()==rai::ST_mesh) f->shape->mesh().writeArr(FILE(filename));
      if(f->shape->type()==rai::ST_ssCvx) f->shape->sscCore().writeArr(FILE(filename));
    }
  }
}

#define DEBUG(x) //x

/** @brief prototype for \c operator>> */
void rai::KinematicWorld::read(std::istream& is) {
  Graph G(is);
  G.checkConsistency();
  //  cout <<"***KVG:\n" <<G <<endl;
  //  FILE("z.G") <<G;
  init(G);
}

Graph rai::KinematicWorld::getGraph() const {
#if 1
  Graph G;
  //first just create nodes
  for(Frame *f: frames) G.newNode<bool>({STRING(f->name <<" [" <<f->ID <<']')}, {});
  for(Frame *f: frames) {
    Node *n = G.elem(f->ID);
    if(f->parent) {
      n->addParent(G.elem(f->parent->ID));
      n->keys.append(STRING("Q= " <<f->Q));
    }
    if(f->joint) {
      n->keys.append(STRING("joint " <<f->joint->type));
    }
    if(f->shape) {
      n->keys.append(STRING("shape " <<f->shape->type()));
    }
    if(f->inertia) {
      n->keys.append(STRING("inertia m=" <<f->inertia->mass));
    }
  }
#else
  Graph G;
  //first just create nodes
  for(Frame *f: frames) G.newSubgraph({f->name}, {});
  
  for(Frame *f: frames) {
    Graph &ats = G.elem(f->ID)->graph();

    ats.newNode<rai::Transformation>({"X"}, {}, f->X);

    if(f->shape) {
      ats.newNode<int>({"shape"}, {}, f->shape->type);
    }

    if(f->link) {
      G.elem(f->ID)->addParent(G.elem(f->link->from->ID));
      if(f->link->joint) {
        ats.newNode<int>({"joint"}, {}, f->joint()->type);
      } else {
        ats.newNode<rai::Transformation>({"Q"}, {}, f->link->Q);
      }
    }
  }
#endif
  G.checkConsistency();
  return G;
}

namespace rai {
  struct Link {
    Frame* joint=NULL;
    FrameL frames;
    Frame *from() {
      Frame *a = joint->parent;
      while(a && !a->joint) a=a->parent;
      return a;
    }
  };
}

//rai::Array<rai::Link *> rai::KinematicWorld::getLinks(){
//  rai::Array<Link*> links;

//  FrameL bases;
//  for(Frame *a:frames){ if(!a->parent) a->getRigidSubFrames(bases); }
//  Link *l = links.append(new Link);
//  l->frames = bases;

//  for(Frame *a:frames) if(a->joint){
//    Link *l = links.append(new Link);
//    l->joint = a;
//    a->getRigidSubFrames(l->frames);
//  }
//  return links;
//}

rai::Array<rai::Frame*> rai::KinematicWorld::getLinks() const {
  FrameL links;
  for(Frame *a:frames) if(!a->parent || a->joint) links.append(a);
  return links;
}

void rai::KinematicWorld::displayDot() {
  Graph G = getGraph();
  G.displayDot();
}

void rai::KinematicWorld::report(std::ostream &os) const {
  CHECK_EQ(fwdActiveSet.N, frames.N, "you need to calc_activeSets before");
  uint nShapes=0, nUc=0;
  for(Frame *f:fwdActiveSet) if(f->shape) nShapes++;
  for(Joint *j:fwdActiveJoints) if(j->uncertainty) nUc++;
  
  os <<"Config: q.N=" <<q.N
    <<" #frames=" <<frames.N
   <<" #activeFrames=" <<fwdActiveSet.N
  <<" #activeJoints=" <<fwdActiveJoints.N
  <<" #activeShapes=" <<nShapes
  <<" #activeUncertainties=" <<nUc
  <<" #proxies=" <<proxies.N
  <<" #contacts=" <<contacts.N
  <<" #evals=" <<setJointStateCount
  <<endl;

  FrameL parts = getParts();
  os <<"PARTS: ";
  for(Frame *f:parts) os <<*f <<endl;
}

void rai::KinematicWorld::init(const Graph& G, bool addInsteadOfClear) {
  if(!addInsteadOfClear) clear();

  FrameL node2frame(G.N);
  node2frame.setZero();
  
  NodeL bs = G.getNodes("body");
  for(Node *n:  bs) {
    CHECK_EQ(n->keys(0),"body","");
    CHECK(n->isGraph(), "bodies must have value Graph");
    
    Frame *b=new Frame(*this);
    node2frame(n->index) = b;
    if(n->keys.N>1) b->name=n->keys.last();
    b->ats.copy(n->graph(), false, true);
    if(n->keys.N>2) b->ats.newNode<bool>({n->keys.last()});
    b->read(b->ats);
  }
  
  for(Node *n: G) {
    if(n->keys(0)=="body" || n->keys(0)=="shape" || n->keys(0)=="joint") continue;
    //    CHECK_EQ(n->keys(0),"frame","");
    CHECK(n->isGraph(), "frame must have value Graph");
    CHECK_LE(n->parents.N, 1,"frames must have no or one parent: specs=" <<*n <<' ' <<n->index);

    Frame *b = NULL;
    if(!n->parents.N) b = new Frame(*this);
    if(n->parents.N==1) b = new Frame(node2frame(n->parents(0)->index)); //getFrameByName(n->parents(0)->keys.last()));
    node2frame(n->index) = b;
    if(n->keys.N && n->keys.last()!="frame") b->name=n->keys.last();
    b->ats.copy(n->graph(), false, true);
    //    if(n->keys.N>2) b->ats.newNode<bool>({n->keys.last()});
    b->read(b->ats);
  }
  
  NodeL ss = G.getNodes("shape");
  for(Node *n: ss) {
    CHECK_EQ(n->keys(0),"shape","");
    CHECK_LE(n->parents.N, 1,"shapes must have no or one parent");
    CHECK(n->isGraph(),"shape must have value Graph");
    
    Frame* f = new Frame(*this);
    if(n->keys.N>1) f->name=n->keys.last();
    f->ats.copy(n->graph(), false, true);
    Shape *s = new Shape(*f);
    s->read(f->ats);
    
    if(n->parents.N==1) {
      Frame *b = listFindByName(frames, n->parents(0)->keys.last());
      CHECK(b, "could not find frame '" <<n->parents(0)->keys.last() <<"'");
      f->linkFrom(b);
      if(f->ats["rel"]) n->graph().get(f->Q, "rel");
    }
  }
  
  NodeL js = G.getNodes("joint");
  for(Node *n: js) {
    CHECK_EQ(n->keys(0),"joint","joints must be declared as joint: specs=" <<*n <<' ' <<n->index);
    CHECK_EQ(n->parents.N,2,"joints must have two parents: specs=" <<*n <<' ' <<n->index);
    CHECK(n->isGraph(),"joints must have value Graph: specs=" <<*n <<' ' <<n->index);
    
    Frame *from=listFindByName(frames, n->parents(0)->keys.last());
    Frame *to=listFindByName(frames, n->parents(1)->keys.last());
    CHECK(from,"JOINT: from '" <<n->parents(0)->keys.last() <<"' does not exist ["<<*n <<"]");
    CHECK(to,"JOINT: to '" <<n->parents(1)->keys.last() <<"' does not exist ["<<*n <<"]");
    
    Frame *f=new Frame(*this);
    if(n->keys.N>1) {
      f->name=n->keys.last();
    } else {
      f->name <<'|' <<to->name; //the joint frame is actually the link frame of all child frames
    }
    f->ats.copy(n->graph(), false, true);
    
    f->linkFrom(from);
    to->linkFrom(f);
    
    Joint *j=new Joint(*f);
    j->read(f->ats);
  }
  
  //if the joint is coupled to another:
  {
    Joint *j;
    for(Frame *f: frames) if((j=f->joint) && j->mimic==(Joint*)1) {
      Node *mim = f->ats["mimic"];
      rai::String jointName;
      if(mim->isOfType<rai::String>()) jointName = mim->get<rai::String>();
      else if(mim->isOfType<NodeL>()){
        NodeL nodes = mim->get<NodeL>();
        jointName = nodes.scalar()->keys.last();
      }else{
        HALT("could not retrieve minimick frame for joint '" <<f->name <<"' from ats '" <<f->ats <<"'");
      }
      rai::Frame *mimicFrame = getFrameByName(jointName, true, true);
      CHECK(mimicFrame, "");
      j->mimic = mimicFrame->joint;
      if(!j->mimic) HALT("The joint '" <<*j <<"' is declared coupled to '" <<jointName <<"' -- but that doesn't exist!");
      j->type = j->mimic->type;

      delete mim;
      f->ats.index();
    }
  }
  
  NodeL ucs = G.getNodes("Uncertainty");
  for(Node *n: ucs) {
    CHECK_EQ(n->keys(0), "Uncertainty", "");
    CHECK_EQ(n->parents.N, 1,"Uncertainties must have one parent");
    CHECK(n->isGraph(),"Uncertainties must have value Graph");
    
    Frame* f = getFrameByName(n->parents(0)->keys.last());
    CHECK(f, "");
    Joint *j = f->joint;
    CHECK(j, "Uncertainty parent must be a joint");
    Uncertainty *uc = new Uncertainty(j);
    uc->read(n->graph());
  }
  
  //-- clean up the graph
  calc_q();
  checkConsistency();
  calc_fwdPropagateFrames();
}

void rai::KinematicWorld::writePlyFile(const char* filename) const {
  ofstream os;
  rai::open(os, filename);
  uint nT=0,nV=0;
  uint j;
  rai::Mesh *m;
  for(Frame *f: frames) if(f->shape) { nV += f->shape->mesh().V.d0; nT += f->shape->mesh().T.d0; }
  
  os <<"\
       ply\n\
       format ascii 1.0\n\
       element vertex " <<nV <<"\n\
       property float x\n\
       property float y\n\
       property float z\n\
       property uchar red\n\
       property uchar green\n\
       property uchar blue\n\
       element face " <<nT <<"\n\
       property list uchar int vertex_index\n\
       end_header\n";

       uint k=0;
  rai::Transformation t;
  rai::Vector v;
  Shape * s;
  for(Frame *f: frames) if((s=f->shape)) {
    m = &s->mesh();
    arr col = m->C;
    CHECK_EQ(col.N, 3,"");
    t = s->frame.X;
    if(m->C.d0!=m->V.d0) {
      m->C.resizeAs(m->V);
      for(j=0; j<m->C.d0; j++) m->C[j]=col;
    }
    for(j=0; j<m->V.d0; j++) {
      v.set(m->V(j, 0), m->V(j, 1), m->V(j, 2));
      v = t*v;
      os <<' ' <<v.x <<' ' <<v.y <<' ' <<v.z
        <<' ' <<int(255.f*m->C(j, 0)) <<' ' <<int(255.f*m->C(j, 1)) <<' ' <<int(255.f*m->C(j, 2)) <<endl;
    }
    k+=j;
  }
  uint offset=0;
  for(Frame *f: frames) if((s=f->shape)) {
    m=&s->mesh();
    for(j=0; j<m->T.d0; j++) {
      os <<"3 " <<offset+m->T(j, 0) <<' ' <<offset+m->T(j, 1) <<' ' <<offset+m->T(j, 2) <<endl;
    }
    offset+=m->V.d0;
  }
}

/// dump the list of current proximities on the screen
void rai::KinematicWorld::reportProxies(std::ostream& os, double belowMargin, bool brief) const {
  os <<"Proximity report: #" <<proxies.N <<endl;
  uint i=0;
  for(const Proxy& p: proxies) {
    if(p.d>belowMargin) continue;
    os  <<i <<" ("
       <<p.a->name <<")-("
      <<p.b->name
     <<") d=" <<p.d;
    if(!brief)
      os <<" |A-B|=" <<(p.posB-p.posA).length()
           //        <<" d^2=" <<(p.posB-p.posA).lengthSqr()
        <<" v=" <<(p.posB-p.posA)
       <<" normal=" <<p.normal
      <<" posA=" <<p.posA
      <<" posB=" <<p.posB;
    os <<endl;
    i++;
  }
  os <<"Contact report:" <<endl;
  for(Frame *a:frames) for(Contact *c:a->contacts) if(&c->a==a) {
    c->coll();
    os <<*c <<endl;
  }

}

bool ProxySortComp(const rai::Proxy *a, const rai::Proxy *b) {
  return (a->a < b->a) || (a->a==b->a && a->b<b->b) || (a->a==b->a && a->b==b->b && a->d < b->d);
}

/// clear all forces currently stored at bodies
void rai::KinematicWorld::clearForces() {
  for(Frame *f:  frames) if(f->inertia) {
    f->inertia->force.setZero();
    f->inertia->torque.setZero();
  }
}

/// apply a force on body n
void rai::KinematicWorld::addForce(rai::Vector force, rai::Frame *f) {
  CHECK(f->inertia, "");
  f->inertia->force += force;
  if(!s->physx) {
    NIY;
  } else {
    s->physx->addForce(force, f);
  }
  //n->torque += (pos - n->X.p) ^ force;
}

/// apply a force on body n at position pos (in world coordinates)
void rai::KinematicWorld::addForce(rai::Vector force, rai::Frame *f, rai::Vector pos) {
  CHECK(f->inertia, "");
  f->inertia->force += force;
  if(!s->physx) {
    NIY;
  } else {
    s->physx->addForce(force, f, pos);
  }
  //n->torque += (pos - n->X.p) ^ force;
}

void rai::KinematicWorld::gravityToForces(double g) {
  rai::Vector grav(0, 0, g);
  for(Frame *f: frames) if(f->inertia) f->inertia->force += f->inertia->mass * grav;
}

/** similar to invDynamics using NewtonEuler; but only computing the backward pass */
void rai::KinematicWorld::NewtonEuler_backward() {
  CHECK_EQ(fwdActiveSet.N, frames.N, "you need to calc_activeSets before");
  uint N=fwdActiveSet.N;
  rai::Array<arr> h(N);
  arr Q(N, 6, 6);
  arr force(frames.N,6);
  force.setZero();
  
  for(uint i=0; i<N; i++) {
    h(i).resize(6).setZero();
    Frame *f = fwdActiveSet.elem(i);
    if(f->joint) {
      h(i) = f->joint->get_h();
    }
    if(f->parent) {
      Q[i] = f->Q.getWrenchTransform();
    } else {
      Q[i].setId();
    }
    if(f->inertia) {
      force[f->ID] = f->inertia->getFrameRelativeWrench();
    }
  }
  
  for(uint i=N; i--;) {
    Frame *f = fwdActiveSet.elem(i);
    if(f->parent) force[f->parent->ID] += ~Q[i] * force[f->ID];
  }
  
  for(Frame *f:frames) {
    rai::Transformation R = f->X; //rotate to world, but no translate to origin
    R.pos.setZero();
    force[f->ID] = ~R.getWrenchTransform() * force[f->ID];
    cout <<f->name <<":\t " <<force[f->ID] <<endl;
  }
}

/// compute forces from the current contacts
void rai::KinematicWorld::contactsToForces(double hook, double damp) {
  rai::Vector trans, transvel, force;
  for(const Proxy& p:proxies) if(p.d<0.) {
    //if(!i || proxies(i-1).a!=a || proxies(i-1).b!=b) continue; //no old reference sticking-frame
    //trans = p.rel.p - proxies(i-1).rel.p; //translation relative to sticking-frame
    trans    = p.posB-p.posA;
    //transvel = p.velB-p.velA;
    //d=trans.length();

    force.setZero();
    force += (hook) * trans; //*(1.+ hook*hook*d*d)
    //force += damp * transvel;
    SL_DEBUG(1, cout <<"applying force: [" <<*p.a <<':' <<*p.b <<"] " <<force <<endl);

    addForce(force, p.a, p.posA);
    addForce(-force, p.b, p.posB);
  }
}

void rai::KinematicWorld::kinematicsPenetrations(arr& y, arr& J, bool penetrationsOnly, double activeMargin) const {
  y.resize(proxies.N).setZero();
  if(!!J) J.resize(y.N, getJointStateDimension()).setZero();
  uint i=0;
  for(const Proxy& p:proxies) {
    if(!p.coll)((Proxy*)&p)->calc_coll(*this);
    
    arr Jp1, Jp2;
    if(!!J) {
      jacobianPos(Jp1, p.a, p.coll->p1);
      jacobianPos(Jp2, p.b, p.coll->p2);
    }
    
    arr y_dist, J_dist;
    p.coll->kinDistance(y_dist, (!!J?J_dist:NoArr), Jp1, Jp2);
    
    if(!penetrationsOnly || y_dist.scalar()<activeMargin) {
      y(i) = -y_dist.scalar();
      if(!!J) J[i] = -J_dist;
    }
  }
}

void rai::KinematicWorld::kinematicsProxyDist(arr& y, arr& J, const Proxy& p, double margin, bool useCenterDist, bool addValues) const {
  y.resize(1);
  if(!!J) J.resize(1, getJointStateDimension());
  if(!addValues) { y.setZero();  if(!!J) J.setZero(); }
  
  //  //costs
  //  if(a->type==rai::ST_sphere && b->type==rai::ST_sphere){
  //    rai::Vector diff=a->X.pos-b->X.pos;
  //    double d = diff.length() - a->size(3) - b->size(3);
  //    y(0) = d;
  //    if(!!J){
  //      arr Jpos;
  //      arr normal = conv_vec2arr(diff)/diff.length(); normal.reshape(1, 3);
  //      kinematicsPos(NoArr, Jpos, a->body);  J += (normal*Jpos);
  //      kinematicsPos(NoArr, Jpos, b->body);  J -= (normal*Jpos);
  //    }
  //    return;
  //  }
  y(0) = p.d;
  if(!!J) {
    arr Jpos;
    rai::Vector arel, brel;
    if(p.d>0.) { //we have a gradient on pos only when outside
      arel=p.a->X.rot/(p.posA-p.a->X.pos);
      brel=p.b->X.rot/(p.posB-p.b->X.pos);
      CHECK(p.normal.isNormalized(), "proxy normal is not normalized");
      arr normal; normal.referTo(&p.normal.x, 3); normal.reshape(1, 3);
      kinematicsPos(NoArr, Jpos, p.a, arel);  J += (normal*Jpos);
      kinematicsPos(NoArr, Jpos, p.b, brel);  J -= (normal*Jpos);
    }
  }
}

void rai::KinematicWorld::kinematicsProxyCost(arr& y, arr& J, const Proxy& p, double margin, bool addValues) const {
  CHECK(p.a->shape,"");
  CHECK(p.b->shape,"");

  y.resize(1);
  if(!!J) J.resize(1, getJointStateDimension());
  if(!addValues) { y.setZero();  if(!!J) J.setZero(); }

  //early check: if swift is way out of collision, don't bother computing it precise
  if(p.d>p.a->shape->radius()+p.a->shape->radius()+.01+margin) return;
  
#if 1
  if(!p.coll) ((Proxy*)&p)->calc_coll(*this);

  if(p.coll->getDistance()>margin) return;
  
  arr Jp1, Jp2;
  if(!!J) {
    jacobianPos(Jp1, p.a, p.coll->p1);
    jacobianPos(Jp2, p.b, p.coll->p2);
  }
  
  arr y_dist, J_dist;
  p.coll->kinDistance(y_dist, (!!J?J_dist:NoArr), Jp1, Jp2);
    
  if(y_dist.scalar()>margin) return;
  y += margin-y_dist.scalar();
  if(!!J)  J -= J_dist;
  
#else
  CHECK(a->shape->mesh_radius>0.,"");
  CHECK(b->shape->mesh_radius>0.,"");
  
  y.resize(1);
  if(!!J) J.resize(1, getJointStateDimension());
  if(!addValues) { y.setZero();  if(!!J) J.setZero(); }
  
  //costs
  if(a->shape->type()==rai::ST_sphere && b->shape->type()==rai::ST_sphere) {
    rai::Vector diff=a->X.pos-b->X.pos;
    double d = diff.length() - a->shape->size(3) - b->shape->size(3);
    y(0) = 1. - d/margin;
    if(!!J) {
      arr Jpos;
      arr normal = conv_vec2arr(diff)/diff.length(); normal.reshape(1, 3);
      kinematicsPos(NoArr, Jpos, a);  J -= 1./margin*(normal*Jpos);
      kinematicsPos(NoArr, Jpos, b);  J += 1./margin*(normal*Jpos);
    }
    return;
  }
  double ab_radius = margin + 10.*(a->shape->mesh_radius+b->shape->mesh_radius);
  CHECK(p->d<(1.+1e-6)*margin, "something's really wierd here!");
  CHECK(p->cenD<(1.+1e-6)*ab_radius, "something's really wierd here! You disproved the triangle inequality :-)");
  double d1 = 1.-p->d/margin;
  double d2 = 1.-p->cenD/ab_radius;
  if(d2<0.) d2=0.;
  if(!useCenterDist) d2=1.;
  y(0) += d1*d2;
  
  //Jacobian
  if(!!J) {
    arr Jpos;
    rai::Vector arel, brel;
    if(p->d>0.) { //we have a gradient on pos only when outside
      arel=a->X.rot/(p->posA-a->X.pos);
      brel=b->X.rot/(p->posB-b->X.pos);
      CHECK(p->normal.isNormalized(), "proxy normal is not normalized");
      arr normal; normal.referTo(&p->normal.x, 3); normal.reshape(1, 3);

      kinematicsPos(NoArr, Jpos, a, arel);  J -= d2/margin*(normal*Jpos);
      kinematicsPos(NoArr, Jpos, b, brel);  J += d2/margin*(normal*Jpos);
    }

    if(useCenterDist && d2>0.) {
      arel=a->X.rot/(p->cenA-a->X.pos);
      brel=b->X.rot/(p->cenB-b->X.pos);
      //      CHECK(p->cenN.isNormalized(), "proxy normal is not normalized");
      if(!p->cenN.isNormalized()) {
        RAI_MSG("proxy->cenN is not normalized: objects seem to be at exactly the same place");
      } else {
        arr normal; normal.referTo(&p->cenN.x, 3); normal.reshape(1, 3);

        kinematicsPos(NoArr, Jpos, a, arel);  J -= d1/ab_radius*(normal*Jpos);
        kinematicsPos(NoArr, Jpos, b, brel);  J += d1/ab_radius*(normal*Jpos);
      }
    }
  }
#endif
}

/// measure (=scalar kinematics) for the contact cost summed over all bodies
void rai::KinematicWorld::kinematicsProxyCost(arr &y, arr& J, double margin) const {
  y.resize(1).setZero();
  if(!!J) J.resize(1, getJointStateDimension()).setZero();
  for(const Proxy& p:proxies) { /*if(p.d<margin)*/
    kinematicsProxyCost(y, J, p, margin, true);
  }
}

void rai::KinematicWorld::kinematicsContactCost(arr& y, arr& J, const Contact* c, double margin, bool addValues) const {
  NIY;
  //  Feature *map = c->getTM_ContactNegDistance();
  //  arr y_dist, J_dist;
  //  map->phi(y_dist, (!!J?J_dist:NoArr), *this);
  //  y_dist *= -1.;
  //  if(!!J) J_dist *= -1.;
  
  //  y.resize(1);
  //  if(!!J) J.resize(1, getJointStateDimension());
  //  if(!addValues) { y.setZero();  if(!!J) J.setZero(); }
  
  //  if(y_dist.scalar()>margin) return;
  //  y += margin-y_dist.scalar();
  //  if(!!J)  J -= J_dist;
}

void rai::KinematicWorld::kinematicsContactCost(arr &y, arr& J, double margin) const {
  y.resize(1).setZero();
  if(!!J) J.resize(1, getJointStateDimension()).setZero();
  for(Frame *f:frames) for(Contact *c:f->contacts) if(&c->a==f) {
    kinematicsContactCost(y, J, c, margin, true);
  }
}

void rai::KinematicWorld::kinematicsProxyConstraint(arr& g, arr& J, const Proxy& p, double margin) const {
  if(!!J) J.resize(1, getJointStateDimension()).setZero();
  
  g.resize(1) = margin - p.d;
  
  //Jacobian
  if(!!J) {
    arr Jpos, normal;
    rai::Vector arel,brel;
    if(p.d>0.) { //we have a gradient on pos only when outside
      arel=p.a->X.rot/(p.posA-p.a->X.pos);
      brel=p.b->X.rot/(p.posB-p.b->X.pos);
      CHECK(p.normal.isNormalized(), "proxy normal is not normalized");
      normal.referTo(&p.normal.x, 3);
    } else { //otherwise take gradient w.r.t. centers...
      arel.setZero(); //a->X.rot/(p.cenA-a->X.pos);
      brel.setZero(); //b->X.rot/(p.cenB-b->X.pos);
      CHECK(p.normal.isNormalized(), "proxy normal is not normalized");
      normal.referTo(&p.normal.x, 3);
    }
    normal.reshape(1, 3);
    
    kinematicsPos(NoArr, Jpos, p.a, arel);  J -= (normal*Jpos);
    kinematicsPos(NoArr, Jpos, p.b, brel);  J += (normal*Jpos);
  }
}

void rai::KinematicWorld::kinematicsContactConstraints(arr& y, arr &J) const {
  J.clear();
  rai::Vector normal;
  uint con=0;
  arr Jpos, dnormal, grad(1, q.N);
  
  y.clear();
  for(const rai::Proxy& p: proxies) y.append(p.d);
  
  if(!J) return; //do not return the Jacobian
  
  rai::Vector arel, brel;
  for(const rai::Proxy& p: proxies) {
    arel.setZero();  arel=p.a->X.rot/(p.posA-p.a->X.pos);
    brel.setZero();  brel=p.b->X.rot/(p.posB-p.b->X.pos);
    
    CHECK(p.normal.isNormalized(), "proxy normal is not normalized");
    dnormal = p.normal.getArr(); dnormal.reshape(1, 3);
    grad.setZero();
    kinematicsPos(NoArr, Jpos, p.a, arel); grad += dnormal*Jpos; //moving a long normal b->a increases distance
    kinematicsPos(NoArr, Jpos, p.b, brel); grad -= dnormal*Jpos; //moving b long normal b->a decreases distance
    J.append(grad);
    con++;
  }
  J.reshape(con, q.N);
}

void rai::KinematicWorld::kinematicsLimitsCost(arr &y, arr &J, const arr& limits, double margin) const {
  y.resize(1).setZero();
  if(!!J) J.resize(1, getJointStateDimension()).setZero();
  double d;
  for(uint i=0; i<limits.d0; i++) if(limits(i,1)>limits(i,0)) { //only consider proper limits (non-zero interval)
    double m = margin*(limits(i,1)-limits(i,0));
    d = limits(i, 0) + m - q(i); //lo
    if(d>0.) {  y(0) += d/m;  if(!!J) J(0, i)-=1./m;  }
    d = q(i) - limits(i, 1) + m; //up
    if(d>0.) {  y(0) += d/m;  if(!!J) J(0, i)+=1./m;  }
  }
}

/// Compute the new configuration q such that body is located at ytarget (with deplacement rel).
void rai::KinematicWorld::inverseKinematicsPos(Frame& body, const arr& ytarget,
                                               const rai::Vector& rel_offset, int max_iter) {
  arr q0, q;
  getJointState(q0);
  q = q0;
  arr y; // endeff pos
  arr J; // Jacobian
  arr invJ;
  arr I = eye(q.N);
  
  // general inverse kinematic update
  // first iteration: $q* = q' + J^# (y* - y')$
  // next iterations: $q* = q' + J^# (y* - y') + (I - J# J)(q0 - q')$
  for(int i = 0; i < max_iter; i++) {
    kinematicsPos(y, J, &body, rel_offset);
    invJ = ~J * inverse(J * ~J);  // inverse_SymPosDef should work!?
    q = q + invJ * (ytarget - y);
    
    if(i > 0) {
      q += (I - invJ * J) * (q0 - q);
    }
    setJointState(q);
  }
}

#if 0
/// center of mass of the whole configuration (3 vector)
double rai::KinematicWorld::getCenterOfMass(arr& x_) const {
  double M=0.;
  rai::Vector x;
  x.setZero();
  for(Frame *f: frames) if(f->inertia) {
    M += f->inertia->mass;
    x += f->inertia->mass*f->X.pos;
  }
  x /= M;
  x_ = conv_vec2arr(x);
  return M;
}

/// gradient (Jacobian) of the COM w.r.t. q (3 x n tensor)
void rai::KinematicWorld::getComGradient(arr &grad) const {
  double M=0.;
  arr J(3, getJointStateDimension());
  grad.resizeAs(J); grad.setZero();
  for(Frame *f: frames) if(f->inertia) {
    M += f->inertia->mass;
    kinematicsPos(NoArr, J, f);
    grad += f->inertia->mass * J;
  }
  grad/=M;
}

const rai::Proxy* rai::KinematicWorld::getContact(uint a, uint b) const {
  for(const rai::Proxy& p: proxies) if(p.d<0.) {
    if(p.a->ID==a && p.b->ID==b) return &p;
    if(p.a->ID==b && p.b->ID==a) return &p;
  }
  return NULL;
}

#endif

/** @brief */
double rai::KinematicWorld::getEnergy() {
  double m, v, E;
  rai::Matrix I;
  rai::Vector w;

  arr vel = calc_fwdPropagateVelocities();

  E=0.;
  for(Frame *f: frames) if(f->inertia) {
    Vector linVel = vel(f->ID, 0, {});
    Vector angVel = vel(f->ID, 1, {});

    m=f->inertia->mass;
    rai::Quaternion &rot = f->X.rot;
    I=(rot).getMatrix() * f->inertia->matrix * (-rot).getMatrix();
    v = linVel.length();
    w = angVel;
    E += .5*m*v*v;
    E += 9.81 * m * (f->X*f->inertia->com).z;
    E += .5*(w*(I*w));
  }

  return E;
}

arr rai::KinematicWorld::getHmetric() const {
  arr H = zeros(getJointStateDimension());
  for(Joint *j: fwdActiveJoints) {
    double h=j->H;
    //    CHECK(h>0.,"Hmetric should be larger than 0");
    if(j->type==JT_transXYPhi) {
      H(j->qIndex+0)=h*10.;
      H(j->qIndex+1)=h*10.;
      H(j->qIndex+2)=h;
    } else {
      for(uint k=0; k<j->qDim(); k++) H(j->qIndex+k)=h;
    }
  }
  return H;
}

void rai::KinematicWorld::pruneRigidJoints(int verbose) {
  rai::Joint *j;
  for(Frame *f:frames) if((j=f->joint)) {
    if(j->type == rai::JT_rigid) delete j; //that's all there is to do
  }
}

void rai::KinematicWorld::reconnectLinksToClosestJoints() {
  reset_q();
  for(Frame *f:frames) if(f->parent) {
#if 0
    Frame *link = f->parent;
    rai::Transformation Q=f->Q;
    while(link->parent && !link->joint) { //walk down links until this is a joint
      Q = link->Q * Q;                 //accumulate transforms
      link = link->parent;
    }
#else
    rai::Transformation Q;
    Frame *link = f->getUpwardLink(Q);
    Q.rot.normalize();
#endif
    if(f->joint && !Q.rot.isZero) continue; //only when rot is zero you can subsume the Q transformation into the Q of the joint
    if(link!=f) { //there is a link's root
      if(link!=f->parent) { //we can rewire to the link's root
        f->parent->parentOf.removeValue(f);
        link->parentOf.append(f);
        f->parent = link;
        f->Q = Q;
      }

      //      if(!link->shape && f->shape && f->Q.isZero()){ //f has a shape, link not -> move shape to link
      //        LOG(-1) <<"Shape '" <<f->name <<"' could be reassociated to link '" <<link->name <<"' (child of '" <<(link->parent?link->parent->name:STRING("NONE")) <<"')";
      ////        link->shape = f->shape;
      ////        f->shape = NULL;
      //      }

      //      if(!link->inertia && f->inertia && f->Q.isZero()){ //f has a shape, link not -> move shape to link
      //        LOG(-1) <<"Inertia '" <<f->name <<"' could be reassociated to link '" <<link->name <<"' (child of '" <<(link->parent?link->parent->name:STRING("NONE")) <<"')";
      ////        link->shape = f->shape;
      ////        f->shape = NULL;
      //      }

    }
  }
}

void rai::KinematicWorld::pruneUselessFrames(bool pruneNamed, bool pruneNonContactNonMarker) {
  for(uint i=frames.N; i--;) {
    Frame *f=frames.elem(i);
    if((pruneNamed || !f->name) && !f->parentOf.N && !f->joint && !f->inertia) {
      if(!f->shape)
        delete f; //that's all there is to do
      else if(pruneNonContactNonMarker && !f->shape->cont && f->shape->type()!=ST_marker)
        delete f;
    }
  }
}

void rai::KinematicWorld::optimizeTree(bool _pruneRigidJoints, bool pruneNamed, bool pruneNonContactNonMarker) {
  if(_pruneRigidJoints) pruneRigidJoints(); //problem: rigid joints bear the semantics of where a body ends
  reconnectLinksToClosestJoints();
  pruneUselessFrames(pruneNamed, pruneNonContactNonMarker);
  calc_activeSets();
  checkConsistency();
}

void rai::KinematicWorld::sortFrames() {
  CHECK_EQ(fwdActiveSet.N ,frames.N, "you need to calc_activeSets before");
  frames = fwdActiveSet;
  uint i=0;
  for(Frame *f: frames) f->ID = i++;
}

void rai::KinematicWorld::makeObjectsFree(const StringA &objects, double H_cost){
  for(auto s:objects){
    rai::Frame *a = getFrameByName(s, true);
    CHECK(a, "");
    a = a->getUpwardLink();
    if(!a->parent) a->linkFrom(frames.first());
    if(!a->joint) new rai::Joint(*a);
    a->joint->makeFree(H_cost);
  }
}

void rai::KinematicWorld::addTimeJoint(){
  rai::Joint *jt = new rai::Joint(*frames.first());
  jt->type = rai::JT_time;
  jt->H = 0.;
}

bool rai::KinematicWorld::hasTimeJoint(){
  Frame *f = frames.first();
  return f && f->joint && (f->joint->type==JT_time);
}

bool rai::KinematicWorld::checkConsistency() const {
  //check qdim
  if(q.nd) {
    uint N = analyzeJointStateDimensions();
    CHECK_EQ(1, q.nd, "");
    CHECK_EQ(N, q.N, "");
    if(qdot.N) CHECK_EQ(N, qdot.N, "");
    
    //count yourself and check...
    uint myqdim = 0;
    for(Joint *j: fwdActiveJoints) {
      if(j->mimic) {
        CHECK_EQ(j->qIndex, j->mimic->qIndex, "");
      } else {
        CHECK_EQ(j->qIndex, myqdim, "joint indexing is inconsistent");
        if(!j->uncertainty)
          myqdim += j->qDim();
        else
          myqdim += 2*j->qDim();
      }
    }
    for(Contact *c: contacts) {
      CHECK_EQ(c->qDim(), 6, "");
      CHECK_EQ(c->qIndex, myqdim, "joint indexing is inconsistent");
      myqdim += c->qDim();
    }
    CHECK_EQ(myqdim, N, "qdim is wrong");
  }
  
  for(Frame *a: frames) {
    CHECK(&a->K, "");
    CHECK(&a->K==this,"");
    CHECK_EQ(a, frames(a->ID), "");
    for(Frame *b: a->parentOf) CHECK_EQ(b->parent, a, "");
    if(a->joint) CHECK_EQ(a->joint->frame, a, "");
    if(a->shape) CHECK_EQ(&a->shape->frame, a, "");
    if(a->inertia) CHECK_EQ(&a->inertia->frame, a, "");
    a->ats.checkConsistency();

    a->Q.checkNan();
    a->X.checkNan();
    CHECK_ZERO(a->Q.rot.normalization()-1., 1e-4, "");
    CHECK_ZERO(a->X.rot.normalization()-1., 1e-4, "");
  }
  
  Joint *j;
  for(Frame *f: frames) if((j=f->joint)) {
    if(j->type.x!=JT_time) {
      CHECK(j->from(), "");
      CHECK(j->from()->parentOf.findValue(j->frame)>=0,"");
    }
    CHECK_EQ(j->frame->joint, j,"");
    CHECK_GE(j->type.x, 0, "");
    CHECK_LE(j->type.x, JT_time, "");

    if(j->mimic) {
      CHECK_EQ(j->dim, 0, "");
      CHECK(j->mimic>(void*)1, "mimic was not parsed correctly");
      CHECK(frames.contains(j->mimic->frame), "mimic points to a frame outside this kinematic configuration");
    }else{
      CHECK_EQ(j->dim, j->getDimFromType(), "");
    }
  }

  //check topsort
  intA level = consts<int>(0, frames.N);
  //compute levels
  for(Frame *f: fwdActiveSet)
    if(f->parent) level(f->ID) = level(f->parent->ID)+1;
  //check levels are strictly increasing across links
  for(Frame *f: fwdActiveSet) if(f->parent) {
    CHECK(level(f->parent->ID) < level(f->ID), "joint from '" <<f->parent->name <<"'[" <<f->parent->ID <<"] to '" <<f->name <<"'[" <<f->ID <<"] does not go forward");
  }

  //check active sets
  for(Frame *f: fwdActiveSet) CHECK(f->active, "");
  boolA jointIsInActiveSet = consts<byte>(false, frames.N);
  for(Joint *j: fwdActiveJoints) { CHECK(j->active, ""); jointIsInActiveSet.elem(j->frame->ID)=true; }
  if(q.nd) {
    for(Frame *f: frames) if(f->joint && f->joint->active) CHECK(jointIsInActiveSet(f->ID), "");
  }
  
  //check isZero for all transformations
  for(Frame *a: frames) {
    a->X.pos.checkZero();
    a->X.rot.checkZero();
    a->Q.pos.checkZero();
    a->Q.rot.checkZero();
  }
  
  for(const Proxy& p : proxies) {
    CHECK_EQ(this, &p.a->K, "");
    CHECK_EQ(this, &p.b->K, "");
  }
  
  return true;
}

rai::Joint* rai::KinematicWorld::attach(Frame* a, Frame* b){
  b = b->getUpwardLink();
  if(b->parent) b->unLink();
  b->linkFrom(a, true);
  return new rai::Joint(*b, rai::JT_rigid);
}

rai::Joint* rai::KinematicWorld::attach(const char* _a, const char* _b){
    return attach(getFrameByName(_a), getFrameByName(_b));
}

FrameL rai::KinematicWorld::getParts() const{
  FrameL F;
  for(Frame *f:frames) if(f->isPart()) F.append(f);
  return F;
}

//void rai::KinematicWorld::meldFixedJoints(int verbose) {
//  NIY
//#if 0
//  checkConsistency();
//  for(Joint *j: joints) if(j->type==JT_rigid) {
//    if(verbose>0) LOG(0) <<" -- melding fixed joint (" <<j->from->name <<' ' <<j->to->name <<" )" <<endl;
//    Frame *a = j->from;
//    Frame *b = j->to;
//    Transformation bridge = j->A * j->Q * j->B;
//    //reassociate shapes with a
//    if(b->shape){
//      b->shape->frame=a;
//      CHECK_EQ(a->shape, NULL,"");
//      a->shape = b->shape;
//    }
//    b->shape = NULL;
//    //joints from b-to-c now become joints a-to-c
//    for(Frame *f: b->parentOf) {
//      Joint *j = f->joint();
//      if(j){
//        j->from = a;
//        j->A = bridge * j->A;
//        a->parentOf.append(f);
//      }
//    }
//    b->parentOf.clear();
//    //reassociate mass
//    a->mass += b->mass;
//    a->inertia += b->inertia;
//    b->mass = 0.;
//  }
//  jointSort();
//  calc_q_from_Q();
//  checkConsistency();
//  //-- remove fixed joints and reindex
//  for_list_rev(Joint, jj, joints) if(jj->type==JT_rigid) delete jj;
//  listReindex(joints);
//  //for(Joint * j: joints) { j->ID=j_COUNT;  j->ifrom = j->from->index;  j->ito = j->to->index;  }
//  checkConsistency();
//#endif
//}

void rai::KinematicWorld::glDraw(OpenGL& gl) {
  glDraw_sub(gl);
  
  bool displayUncertainties = false;
  for(Joint *j:fwdActiveJoints) if(j->uncertainty) {
    displayUncertainties=true; break;
  }

  if(displayUncertainties) {
    arr q_org = getJointState();
    for(Joint *j:fwdActiveJoints) if(j->uncertainty) {
      for(uint i=0; i<j->qDim(); i++) {
        arr q=q_org;
        q(j->qIndex+i) -= j->uncertainty->sigma(i);
        setJointState(q);
        glDraw_sub(gl);
        q=q_org;
        q(j->qIndex+i) += j->uncertainty->sigma(i);
        setJointState(q);
        glDraw_sub(gl);
      }
    }
    setJointState(q_org);
  }
}

/// GL routine to draw a rai::KinematicWorld
void rai::KinematicWorld::glDraw_sub(OpenGL& gl) {
#ifdef RAI_GL
  rai::Transformation f;
  double GLmatrix[16];
  
  glPushMatrix();
  
  glColor(.5, .5, .5);
  
  if(orsDrawVisualsOnly)
    orsDrawProxies=orsDrawJoints=orsDrawMarkers=false;

  //proxies
  if(orsDrawProxies) for(const Proxy& p: proxies)((Proxy*)&p)->glDraw(gl);

  //contacts
  if(orsDrawProxies) for(const Frame *fr: frames) for(rai::Contact *c:fr->contacts) if(&c->a==fr) {
    c->glDraw(gl);
  }

  //joints
  Joint *e;
  if(orsDrawJoints) for(Frame *fr: frames) if((e=fr->joint)) {
    //set name (for OpenGL selection)
    glPushName((fr->ID <<2) | 2);

    //    double s=e->A.pos.length()+e->B.pos.length(); //some scale
    double s=.1;

    //    //from body to joint
    //    f=e->from->X;
    //    f.getAffineMatrixGL(GLmatrix);
    //    glLoadMatrixd(GLmatrix);
    //    glColor(1, 1, 0);
    //    //glDrawSphere(.1*s);
    //    glBegin(GL_LINES);
    //    glVertex3f(0, 0, 0);
    //    glVertex3f(e->A.pos.x, e->A.pos.y, e->A.pos.z);
    //    glEnd();

    //joint frame A
    //    f.appendTransformation(e->A);
    f.getAffineMatrixGL(GLmatrix);
    glLoadMatrixd(GLmatrix);
    glDrawAxes(s);
    glColor(1, 0, 0);
    glRotatef(90, 0, 1, 0);  glDrawCylinder(.05*s, .3*s);  glRotatef(-90, 0, 1, 0);

    //joint frame B
    f.appendTransformation(fr->Q);
    f.getAffineMatrixGL(GLmatrix);
    glLoadMatrixd(GLmatrix);
    glDrawAxes(s);

    //    //from joint to body
    //    glColor(1, 0, 1);
    //    glBegin(GL_LINES);
    //    glVertex3f(0, 0, 0);
    //    glVertex3f(e->B.pos.x, e->B.pos.y, e->B.pos.z);
    //    glEnd();
    //    glTranslatef(e->B.pos.x, e->B.pos.y, e->B.pos.z);
    //    //glDrawSphere(.1*s);

    glPopName();
  }

  //shapes
  if(orsDrawBodies) {
    //first non-transparent
    for(Frame *f: frames) if(f->shape && f->shape->alpha()==1. && (f->shape->visual||!orsDrawVisualsOnly)) {
      gl.drawId(f->ID);
      f->shape->glDraw(gl);
    }
    for(Frame *f: frames) if(f->shape && f->shape->alpha()<1. && (f->shape->visual||!orsDrawVisualsOnly)) {
      gl.drawId(f->ID);
      f->shape->glDraw(gl);
    }
  }

  glPopMatrix();
#endif
}

//===========================================================================

void kinVelocity(arr &y, arr &J, uint frameId, const WorldL &Ktuple, double tau) {
  CHECK_GE(Ktuple.N, 1, "");
  rai::KinematicWorld &K0 = *Ktuple(-2);
  rai::KinematicWorld &K1 = *Ktuple(-1);
  rai::Frame *f0 = K0.frames(frameId);
  rai::Frame *f1 = K1.frames(frameId);
  
  arr y0,J0;
  K0.kinematicsPos(y0, J0, f0);
  K1.kinematicsPos(y, J, f1);
  y -= y0;
  J -= J0;
  y /= tau;
  J /= tau;
}

//===========================================================================
//
// helper routines -- in a classical C interface
//

#endif

#undef LEN

double forceClosureFromProxies(rai::KinematicWorld& K, uint bodyIndex, double distanceThreshold, double mu, double torqueWeights) {
  rai::Vector c, cn;
  arr C, Cn;
  for(const rai::Proxy& p: K.proxies) {
    int body_a = p.a?p.a->ID:-1;
    int body_b = p.b?p.b->ID:-1;
    if(p.d<distanceThreshold && (body_a==(int)bodyIndex || body_b==(int)bodyIndex)) {
      if(body_a==(int)bodyIndex) {
        c = p.posA;
        cn=-p.normal;
      } else {
        c = p.posB;
        cn= p.normal;
      }
      C.append(conv_vec2arr(c));
      Cn.append(conv_vec2arr(cn));
    }
  }
  C .reshape(C.N/3, 3);
  Cn.reshape(C.N/3, 3);
  double fc=forceClosure(C, Cn, K.frames(bodyIndex)->X.pos, mu, torqueWeights, NULL);
  return fc;
}

void transferQbetweenTwoWorlds(arr& qto, const arr& qfrom, const rai::KinematicWorld& to, const rai::KinematicWorld& from) {
  arr q = to.getJointState();
  uint T = qfrom.d0;
  uint Nfrom = qfrom.d1;
  
  if(qfrom.d1==0) {T = 1; Nfrom = qfrom.d0;}
  
  qto = repmat(~q,T,1);
  
  intA match(Nfrom);
  match = -1;
  rai::Joint* jfrom;
  for(rai::Frame* f: from.frames) if((jfrom=f->joint)) {
    rai::Joint* jto = to.getJointByBodyNames(jfrom->from()->name, jfrom->frame->name);
    if(!jto || !jfrom->qDim() || !jto->qDim()) continue;
    CHECK_EQ(jfrom->qDim(), jto->qDim(), "joints must have same dimensionality");
    for(uint i=0; i<jfrom->qDim(); i++) {
      match(jfrom->qIndex+i) = jto->qIndex+i;
    }
  }

  for(uint i=0; i<match.N; i++) if(match(i)!=-1) {
    for(uint t=0; t<T; t++) {
      if(qfrom.d1==0) {
        qto(t, match(i)) = qfrom(i);
      } else {
        qto(t, match(i)) = qfrom(t,i);
      }
    }
  }

  if(qfrom.d1==0) qto.reshape(qto.N);
}

#if 0 //nonsensical
void transferQDotbetweenTwoWorlds(arr& qDotTo, const arr& qDotFrom, const rai::KinematicWorld& to, const rai::KinematicWorld& from) {
  //TODO: for saveness reasons, the velocities are zeroed.
  arr qDot;
  qDot = zeros(to.getJointStateDimension());
  uint T, dim;
  if(qDotFrom.d1 > 0) {
    T = qDotFrom.d0;
    qDotTo = repmat(~qDot,T,1);
    dim = qDotFrom.d1;
  } else {
    T = 1;
    qDotTo = qDot;
    dim = qDotFrom.d0;
  }
  
  intA match(dim);
  match = -1;
  for(rai::Joint* jfrom:from.joints) {
    rai::Joint* jto = to.getJointByName(jfrom->name, false); //OLD: to.getJointByBodyNames(jfrom->from->name, jfrom->to->name); why???
    if(!jto || !jfrom->qDim() || !jto->qDim()) continue;
    CHECK_EQ(jfrom->qDim(), jto->qDim(), "joints must have same dimensionality");
    for(uint i=0; i<jfrom->qDim(); i++) {
      match(jfrom->qIndex+i) = jto->qIndex+i;
    }
  }
  if(qDotFrom.d1 > 0) {
    for(uint i=0; i<match.N; i++) if(match(i)!=-1) {
      for(uint t=0; t<T; t++) {
        qDotTo(t, match(i)) = qDotFrom(t,i);
      }
    }
  } else {
    for(uint i=0; i<match.N; i++) if(match(i)!=-1) {
      qDotTo(match(i)) = qDotFrom(i);
    }
  }
  
}

void transferKpBetweenTwoWorlds(arr& KpTo, const arr& KpFrom, const rai::KinematicWorld& to, const rai::KinematicWorld& from) {
  KpTo = zeros(to.getJointStateDimension(),to.getJointStateDimension());
  //use Kp gains from ors file for toWorld, if there are no entries of this joint in fromWorld
  for_list(rai::Joint, j, to.joints) {
    if(j->qDim()>0) {
      arr *info;
      info = j->ats.find<arr>("gains");
      if(info) {
        KpTo(j->qIndex,j->qIndex)=info->elem(0);
      }
    }
  }
  
  intA match(KpFrom.d0);
  match = -1;
  for(rai::Joint* jfrom : from.joints) {
    rai::Joint* jto = to.getJointByName(jfrom->name, false); // OLD: rai::Joint* jto = to.getJointByBodyNames(jfrom->from->name, jfrom->to->name);
    if(!jto || !jfrom->qDim() || !jto->qDim()) continue;
    CHECK_EQ(jfrom->qDim(), jto->qDim(), "joints must have same dimensionality");
    for(uint i=0; i<jfrom->qDim(); i++) {
      match(jfrom->qIndex+i) = jto->qIndex+i;
    }
  }
  
  for(uint i=0; i<match.N; i++) {
    for(uint j=0; j<match.N; j++) {
      KpTo(match(i), match(j)) = KpFrom(i,j);
    }
  }
}

void transferKdBetweenTwoWorlds(arr& KdTo, const arr& KdFrom, const rai::KinematicWorld& to, const rai::KinematicWorld& from) {
  KdTo = zeros(to.getJointStateDimension(),to.getJointStateDimension());
  
  //use Kd gains from ors file for toWorld, if there are no entries of this joint in fromWorld
  for_list(rai::Joint, j, to.joints) {
    if(j->qDim()>0) {
      arr *info;
      info = j->ats.find<arr>("gains");
      if(info) {
        KdTo(j->qIndex,j->qIndex)=info->elem(1);
      }
    }
  }
  
  intA match(KdFrom.d0);
  match = -1;
  for(rai::Joint* jfrom : from.joints) {
    rai::Joint* jto = to.getJointByName(jfrom->name, false); // OLD: rai::Joint* jto = to.getJointByBodyNames(jfrom->from->name, jfrom->to->name);
    if(!jto || !jfrom->qDim() || !jto->qDim()) continue;
    CHECK_EQ(jfrom->qDim(), jto->qDim(), "joints must have same dimensionality");
    for(uint i=0; i<jfrom->qDim(); i++) {
      match(jfrom->qIndex+i) = jto->qIndex+i;
    }
  }
  
  for(uint i=0; i<match.N; i++) {
    for(uint j=0; j<match.N; j++) {
      KdTo(match(i), match(j)) = KdFrom(i,j);
    }
  }
}

void transferU0BetweenTwoWorlds(arr& u0To, const arr& u0From, const rai::KinematicWorld& to, const rai::KinematicWorld& from) {
  u0To = zeros(to.getJointStateDimension());
  
  intA match(u0From.d0);
  match = -1;
  for(rai::Joint* jfrom : from.joints) {
    rai::Joint* jto = to.getJointByName(jfrom->name, false); // OLD: rai::Joint* jto = to.getJointByBodyNames(jfrom->from->name, jfrom->to->name);
    if(!jto || !jfrom->qDim() || !jto->qDim()) continue;
    CHECK_EQ(jfrom->qDim(), jto->qDim(), "joints must have same dimensionality");
    for(uint i=0; i<jfrom->qDim(); i++) {
      match(jfrom->qIndex+i) = jto->qIndex+i;
    }
  }
  
  for(uint i=0; i<match.N; i++) {
    u0To(match(i)) = u0From(i);
  }
}

void transferKI_ft_BetweenTwoWorlds(arr& KI_ft_To, const arr& KI_ft_From, const rai::KinematicWorld& to, const rai::KinematicWorld& from) {
  uint numberOfColumns = KI_ft_From.d1;
  if(KI_ft_From.d1 == 0) {
    numberOfColumns = 1;
    KI_ft_To = zeros(to.getJointStateDimension());
  } else {
    KI_ft_To = zeros(to.getJointStateDimension(), KI_ft_From.d1);
  }
  
  intA match(KI_ft_From.d0);
  match = -1;
  for(rai::Joint* jfrom : from.joints) {
    rai::Joint* jto = to.getJointByName(jfrom->name, false); // OLD: rai::Joint* jto = to.getJointByBodyNames(jfrom->from->name, jfrom->to->name);
    if(!jto || !jfrom->qDim() || !jto->qDim()) continue;
    CHECK_EQ(jfrom->qDim(), jto->qDim(), "joints must have same dimensionality");
    for(uint i=0; i<jfrom->qDim(); i++) {
      match(jfrom->qIndex+i) = jto->qIndex+i;
    }
  }
  
  for(uint i=0; i<match.N; i++) {
    for(uint j=0; j < numberOfColumns; j++) {
      if(numberOfColumns > 1) {
        KI_ft_To(match(i), j) = KI_ft_From(i,j);
      } else {
        KI_ft_To(match(i)) = KI_ft_From(i);
      }
    }
  }
}
#endif

//===========================================================================
//===========================================================================
// opengl
//===========================================================================
//===========================================================================

#ifndef RAI_ORS_ONLY_BASICS

/**
 * @brief Bind ors to OpenGL.
 * Afterwards OpenGL can show the ors graph.
 *
 * @param graph the ors graph.
 * @param gl OpenGL which shows the ors graph.
 */
void bindOrsToOpenGL(rai::KinematicWorld& graph, OpenGL& gl) {
  gl.add(glStandardScene, 0);
  gl.add(rai::glDrawGraph, &graph);
  //  gl.setClearColors(1., 1., 1., 1.);

  rai::Frame* glCamera = graph.getFrameByName("glCamera");
  if(glCamera) {
    gl.camera.X = glCamera->X;
    gl.resize(500,500);
  } else {
    gl.camera.setPosition(10., -15., 8.);
    gl.camera.focus(0, 0, 1.);
    gl.camera.upright();
  }
  gl.update();
}
#endif

#ifndef RAI_ORS_ONLY_BASICS

/// static GL routine to draw a rai::KinematicWorld
void rai::glDrawGraph(void *classP, OpenGL& gl) {
  ((rai::KinematicWorld*)classP)->glDraw(gl);
}

void rai::glDrawProxies(void *P) {
#ifdef RAI_GL
  ProxyL& proxies = *((ProxyL*)P);
  glPushMatrix();
  for(rai::Proxy* p:proxies) p->glDraw(NoOpenGL);
  glPopMatrix();
#endif
}

void displayState(const arr& x, rai::KinematicWorld& G, const char *tag) {
  G.setJointState(x);
  G.watch(true, tag);
}

void displayTrajectory(const arr& _x, int steps, rai::KinematicWorld& G, const KinematicSwitchL& switches, const char *tag, double delay, uint dim_z, bool copyG) {
  NIY;
#if 0
  if(!steps) return;
  rai::Shape *s;
  for(rai::Frame *f : G.frames) if((s=f->shape)) {
    if(s->mesh.V.d0!=s->mesh.Vn.d0 || s->mesh.T.d0!=s->mesh.Tn.d0) {
      s->mesh.computeNormals();
    }
  }
  rai::KinematicWorld *Gcopy;
  if(switches.N) copyG=true;
  if(!copyG) Gcopy=&G;
  else {
    Gcopy = new rai::KinematicWorld;
    Gcopy->copy(G,true);
  }
  arr x,z;
  if(dim_z) {
    x.referToRange(_x,0,-dim_z-1);
    z.referToRange(_x,-dim_z,-1);
  } else {
    x.referTo(_x);
  }
  uint n=Gcopy->getJointStateDimension()-dim_z;
  x.reshape(x.N/n,n);
  uint num, T=x.d0-1;
  if(steps==1 || steps==-1) num=T; else num=steps;
  for(uint k=0; k<=(uint)num; k++) {
    uint t = (T?(k*T/num):0);
    if(switches.N) {
      for(rai::KinematicSwitch *sw: switches)
        if(sw->timeOfApplication==t)
          sw->apply(*Gcopy);
    }
    if(dim_z) Gcopy->setJointState(cat(x[t], z));
    else Gcopy->setJointState(x[t]);
    if(delay<0.) {
      if(delay<-10.) FILE("z.graph") <<*Gcopy;
      Gcopy->gl().watch(STRING(tag <<" (time " <<std::setw(3) <<t <<'/' <<T <<')').p);
    } else {
      Gcopy->gl().update(STRING(tag <<" (time " <<std::setw(3) <<t <<'/' <<T <<')').p);
      if(delay) rai::wait(delay);
    }
  }
  if(steps==1)
    Gcopy->gl().watch(STRING(tag <<" (time " <<std::setw(3) <<T <<'/' <<T <<')').p);
  if(copyG) delete Gcopy;
#endif
}

/* please don't remove yet: code for displaying edges might be useful...

void glDrawOdeWorld(void *classP){
  _glDrawOdeWorld((dWorldID)classP);
}

void _glDrawOdeWorld(dWorldID world)
{
  glStandardLight();
  glColor(3);
  glDrawFloor(4);
  uint i;
  Color c;
  dVector3 vec, vec2;
  dBodyID b;
  dGeomID g, gg;
  dJointID j;
  dReal a, al, ah, r, len;
  glPushName(0);
  int t;

  //bodies
  for(i=0, b=world->firstbody;b;b=(dxBody*)b->next){
    i++;
    glPushName(i);

    //if(b->userdata){ glDrawBody(b->userdata); }
    c.setIndex(i); glColor(c.r, c.g, c.b);
    glShadeModel(GL_FLAT);

    //bodies
    for(g=b->geom;g;g=dGeomGetBodyNext(g)){
      if(dGeomGetClass(g)==dGeomTransformClass){
  ((dxGeomTransform*)g)->computeFinalTx();
        glTransform(((dxGeomTransform*)g)->final_pos, ((dxGeomTransform*)g)->final_R);
  gg=dGeomTransformGetGeom(g);
      }else{
  glTransform(g->pos, g->R);
  gg=g;
      }
      b = dGeomGetBody(gg);
      // set the color of the body, 4. Mar 06 (hh)
      c.r = ((Body*)b->userdata)->cr;
      c.g = ((Body*)b->userdata)->cg;
      c.b = ((Body*)b->userdata)->cb;
      glColor(c.r, c.g, c.b);

      switch(dGeomGetClass(gg))
  {
  case dSphereClass:
    glDrawSphere(dGeomSphereGetRadius(gg));
    break;
  case dBoxClass:
    dGeomBoxGetLengths(gg, vec);
    glDrawBox(vec[0], vec[1], vec[2]);
    break;
  case dCCylinderClass: // 6. Mar 06 (hh)
    dGeomCCylinderGetParams(gg, &r, &len);
    glDrawCappedCylinder(r, len);
    break;
  default: HALT("can't draw that geom yet");
  }
      glPopMatrix();
    }

    // removed shadows,  4. Mar 06 (hh)

    // joints

      dxJointNode *n;
      for(n=b->firstjoint;n;n=n->next){
      j=n->joint;
      t=dJointGetType(j);
      if(t==dJointTypeHinge){
      dJointGetHingeAnchor(j, vec);
      a=dJointGetHingeAngle(j);
      al=dJointGetHingeParam(j, dParamLoStop);
      ah=dJointGetHingeParam(j, dParamHiStop);
      glPushMatrix();
      glTranslatef(vec[0], vec[1], vec[2]);
      dJointGetHingeAxis(j, vec);
      glBegin(GL_LINES);
      glColor3f(1, 0, 0);
      glVertex3f(0, 0, 0);
      glVertex3f(LEN*vec[0], LEN*vec[1], LEN*vec[2]);
      glEnd();
      //glDrawText(STRING(al <<'<' <<a <<'<' <<ah), LEN*vec[0], LEN*vec[1], LEN*vec[2]);
      glPopMatrix();
      }
      if(t==dJointTypeAMotor){
  glPushMatrix();
  glTranslatef(b->pos[0], b->pos[1], b->pos[2]);
  dJointGetAMotorAxis(j, 0, vec);
  glBegin(GL_LINES);
  glColor3f(1, 1, 0);
  glVertex3f(0, 0, 0);
  glVertex3f(LEN*vec[0], LEN*vec[1], LEN*vec[2]);
  glEnd();
  glPopMatrix();
      }
      if(t==dJointTypeBall){
  dJointGetBallAnchor(j, vec);
  dJointGetBallAnchor2(j, vec2);
  glPushMatrix();
  glTranslatef(vec[0], vec[1], vec[2]);
  glBegin(GL_LINES);
  glColor3f(1, 0, 0);
  glVertex3f(-.05, 0, 0);
  glVertex3f(.05, 0, 0);
  glVertex3f(0, -.05, 0);
  glVertex3f(0, .05, 0);
  glVertex3f(0, 0, -.05);
  glVertex3f(0, 0, .05);
  glEnd();
  glPopMatrix();
  glPushMatrix();
  glTranslatef(vec2[0], vec2[1], vec2[2]);
  glBegin(GL_LINES);
  glColor3f(1, 0, 0);
  glVertex3f(-.05, 0, 0);
  glVertex3f(.05, 0, 0);
  glVertex3f(0, -.05, 0);
  glVertex3f(0, .05, 0);
  glVertex3f(0, 0, -.05);
  glVertex3f(0, 0, .05);
  glEnd();
  glPopMatrix();
      }
    }
      glPopName();
  }
  glPopName();
}
*/

int animateConfiguration(rai::KinematicWorld& K, Inotify *ino) {
  arr x, x0;
  K.getJointState(x0);
  arr lim = K.getLimits();
  const int steps = 50;
  K.checkConsistency();
  StringA jointNames = K.getJointNames();
  
  //  uint saveCount=0;

  K.gl().pressedkey=0;
  for(uint i=x0.N; i--;) {
    x=x0;
    double upper_lim = lim(i,1);
    double lower_lim = lim(i,0);
    double delta = upper_lim - lower_lim;
    double center = lower_lim + .5*delta;
    if(delta<=1e-10) { center=x0(i); delta=1.; }
    double offset = acos(2. * (x0(i) - center) / delta);
    if(offset!=offset) offset=0.; //if NAN
    
    for(uint t=0; t<steps; t++) {
      if(ino && ino->poll(false, true)) return -1;
      
      x(i) = center + (delta*(0.5*cos(RAI_2PI*t/steps + offset)));
      // Joint limits
      checkNan(x);
      K.setJointState(x);
      int key = K.gl().update(STRING("DOF = " <<i <<" : " <<jointNames(i) <<" [" <<lim[i] <<"]"), false);
      //      write_ppm(gl.captureImage, STRING("vid/" <<std::setw(3)<<std::setfill('0')<<saveCount++<<".ppm"));

      if(key==13 || key==32 || key==27 || key=='q'){
        K.setJointState(x0);
        return key;
      }
      rai::wait(0.01);
    }
  }
  K.setJointState(x0);
  return K.watch(false);
}

rai::Frame *movingBody=NULL;
rai::Vector selpos;
double seld, selx, sely, selz;

struct EditConfigurationClickCall:OpenGL::GLClickCall {
  rai::KinematicWorld *ors;
  EditConfigurationClickCall(rai::KinematicWorld& _ors) { ors=&_ors; }
  bool clickCallback(OpenGL& gl) {
    OpenGL::GLSelect *top=gl.topSelection;
    if(!top) return false;
    uint i=top->name;
    cout <<"CLICK call: id = 0x" <<std::hex <<gl.topSelection->name <<" : ";
    gl.text.clear();
    if((i&3)==1) {
      rai::Frame *s=ors->frames(i>>2);
      gl.text <<"shape selection: shape=" <<s->name <<" X=" <<s->X <<endl;
      //      listWrite(s->ats, gl.text, "\n");
      cout <<gl.text;
    }
    if((i&3)==2) {
      rai::Joint *j = ors->frames(i>>2)->joint;
      gl.text
          <<"edge selection: " <<j->from()->name <<' ' <<j->frame->name
            //         <<"\nA=" <<j->A <<"\nQ=" <<j->Q <<"\nB=" <<j->B
         <<endl;
      //      listWrite(j->ats, gl.text, "\n");
      cout <<gl.text;
    }
    cout <<endl;
    return true;
  }
};

struct EditConfigurationHoverCall:OpenGL::GLHoverCall {
  rai::KinematicWorld *ors;
  EditConfigurationHoverCall(rai::KinematicWorld& _ors);// { ors=&_ors; }
  bool hoverCallback(OpenGL& gl) {
    //    if(!movingBody) return false;
    if(!movingBody) {
      rai::Joint *j=NULL;
      rai::Frame *s=NULL;
      rai::timerStart(true);
      gl.Select(true);
      OpenGL::GLSelect *top=gl.topSelection;
      if(!top) return false;
      uint i=top->name;
      cout <<rai::timerRead() <<"HOVER call: id = 0x" <<std::hex <<gl.topSelection->name <<endl;
      if((i&3)==1) s=ors->frames(i>>2);
      if((i&3)==2) j=ors->frames(i>>2)->joint;
      gl.text.clear();
      if(s) {
        gl.text <<"shape selection: body=" <<s->name <<" X=" <<s->X;
      }
      if(j) {
        gl.text
            <<"edge selection: " <<j->from()->name <<' ' <<j->frame->name
              //           <<"\nA=" <<j->A <<"\nQ=" <<j->Q <<"\nB=" <<j->B
           <<endl;
        //        listWrite(j->ats, gl.text, "\n");
      }
    } else {
      //gl.Select();
      //double x=0, y=0, z=seld;
      //double x=(double)gl.mouseposx/gl.width(), y=(double)gl.mouseposy/gl.height(), z=seld;
      double x=gl.mouseposx, y=gl.mouseposy, z=seld;
      gl.unproject(x, y, z, true);
      cout <<"x=" <<x <<" y=" <<y <<" z=" <<z <<" d=" <<seld <<endl;
      movingBody->X.pos = selpos + ARR(x-selx, y-sely, z-selz);
    }
    return true;
  }
};

EditConfigurationHoverCall::EditConfigurationHoverCall(rai::KinematicWorld& _ors) {
  ors=&_ors;
}

struct EditConfigurationKeyCall:OpenGL::GLKeyCall {
  rai::KinematicWorld &K;
  bool &exit;
  EditConfigurationKeyCall(rai::KinematicWorld& _K, bool& _exit): K(_K), exit(_exit) {}
  bool keyCallback(OpenGL& gl) {
    if(false && gl.pressedkey==' ') { //grab a body
      if(movingBody) { movingBody=NULL; return true; }
      rai::Joint *j=NULL;
      rai::Frame *s=NULL;
      gl.Select();
      OpenGL::GLSelect *top=gl.topSelection;
      if(!top) { cout <<"No object below mouse!" <<endl;  return false; }
      uint i=top->name;
      //cout <<"HOVER call: id = 0x" <<std::hex <<gl.topSelection->name <<endl;
      if((i&3)==1) s=K.frames(i>>2);
      if((i&3)==2) j=K.frames(i>>2)->joint;
      if(s) {
        cout <<"selected shape " <<s->name <<" of body " <<s->name <<endl;
        selx=top->x;
        sely=top->y;
        selz=top->z;
        seld=top->dmin;
        cout <<"x=" <<selx <<" y=" <<sely <<" z=" <<selz <<" d=" <<seld <<endl;
        selpos = s->X.pos;
        movingBody=s;
      }
      if(j) {
        cout <<"selected joint " <<j->frame->ID <<" connecting " <<j->from()->name <<"--" <<j->frame->name <<endl;
      }
      return true;
    } else switch(gl.pressedkey) {
      case '1':  K.orsDrawBodies^=1;  break;
      case '2':  K.orsDrawShapes^=1;  break;
      case '3':  K.orsDrawJoints^=1;  K.orsDrawMarkers^=1; break;
      case '4':  K.orsDrawProxies^=1;  break;
      case '5':  gl.reportSelects^=1;  break;
      case '6':  gl.reportEvents^=1;  break;
      case '7':  K.writePlyFile("z.ply");  break;
      case 'j':  gl.camera.X.pos += gl.camera.X.rot*rai::Vector(0, 0, .1);  break;
      case 'k':  gl.camera.X.pos -= gl.camera.X.rot*rai::Vector(0, 0, .1);  break;
      case 'i':  gl.camera.X.pos += gl.camera.X.rot*rai::Vector(0, .1, 0);  break;
      case ',':  gl.camera.X.pos -= gl.camera.X.rot*rai::Vector(0, .1, 0);  break;
      case 'l':  gl.camera.X.pos += gl.camera.X.rot*rai::Vector(.1, .0, 0);  break;
      case 'h':  gl.camera.X.pos -= gl.camera.X.rot*rai::Vector(.1, 0, 0);  break;
      case 'a':  gl.camera.focus(
              (gl.camera.X.rot*(gl.camera.foc - gl.camera.X.pos)
               ^ gl.camera.X.rot*rai::Vector(1, 0, 0)) * .001
              + gl.camera.foc);
        break;
      case 's':  gl.camera.X.pos +=
            (
              gl.camera.X.rot*(gl.camera.foc - gl.camera.X.pos)
              ^(gl.camera.X.rot * rai::Vector(1., 0, 0))
              ) * .01;
        break;
      case 'q' :
        cout <<"EXITING" <<endl;
        exit=true;
        break;
    }
    gl.postRedrawEvent(true);
    return true;
  }
};

void editConfiguration(const char* filename, rai::KinematicWorld& K){
  K.checkConsistency();

  //  gl.exitkeys="1234567890qhjklias, "; //TODO: move the key handling to the keyCall!
  bool exit=false;
  //  gl.addHoverCall(new EditConfigurationHoverCall(K));
//  K.gl().addKeyCall(new EditConfigurationKeyCall(K,exit));
//  K.gl().addClickCall(new EditConfigurationClickCall(K));
  Inotify ino(filename);
  for(; !exit;) {
    cout <<"reloading `" <<filename <<"' ... " <<std::endl;
    rai::KinematicWorld W;
    try {
      rai::lineCount=1;
      W.init(filename);
      K.gl().dataLock.lock(RAI_HERE);
      K = W;
      K.gl().dataLock.unlock();
      K.report();
    } catch(std::runtime_error& err) {
      cout <<"line " <<rai::lineCount <<": " <<err.what() <<" -- please check the file and re-save" <<endl;
      //      continue;
    }
    cout <<"watching..." <<endl;
    int key = -1;
    K.gl().pressedkey=0;
    for(;;) {
      key = K.watch(false);
      if(key==13 || key==32 || key==27 || key=='q') break;
      if(ino.poll(false, true)) break;
      rai::wait(.02);
    }
    if(exit) break;
    if(key==13 || key==32){
      cout <<"animating.." <<endl;
      //while(ino.pollForModification());
      key = animateConfiguration(K, &ino);
    }
    if(key==27 || key=='q') break;
    if(key==-1) continue;

    if(!rai::getInteractivity()) {
      exit=true;
    }
  }
}

//#endif

#else ///RAI_GL
#ifndef RAI_ORS_ONLY_BASICS
void bindOrsToOpenGL(rai::KinematicWorld&, OpenGL&) { NICO };
void rai::KinematicWorld::glDraw(OpenGL&) { NICO }
void rai::glDrawGraph(void *classP) { NICO }
void editConfiguration(const char* orsfile, rai::KinematicWorld& C) { NICO }
void animateConfiguration(rai::KinematicWorld& C, OpenGL&, Inotify*) { NICO }
void glTransform(const rai::Transformation&) { NICO }
void displayTrajectory(const arr&, int, rai::KinematicWorld&, const char*, double) { NICO }
void displayState(const arr&, rai::KinematicWorld&, const char*) { NICO }
#endif
#endif
/** @} */

//===========================================================================
//===========================================================================
// featherstone
//===========================================================================
//===========================================================================

//===========================================================================
//===========================================================================
// template instantiations
//===========================================================================
//===========================================================================

#include <Core/util.tpp>

#ifndef  RAI_ORS_ONLY_BASICS
template rai::Array<rai::Shape*>::Array(uint);
//template rai::Shape* listFindByName(const rai::Array<rai::Shape*>&,const char*);

#include <Core/array.tpp>
template rai::Array<rai::Joint*>::Array();
#endif
/** @} */

