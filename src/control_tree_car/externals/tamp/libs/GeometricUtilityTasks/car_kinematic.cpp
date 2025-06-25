#include <car_kinematic.h>
#include <geom_utility.h>

CarKinematic::CarKinematic( const std::string & object, const rai::KinematicWorld& G )
  : object_index_(getFrameIndex(G, object))
{

}

void CarKinematic::phi(arr& y, arr& J, const WorldL& Gs)
{
  CHECK(order==1,"");
  CHECK(Gs.size() >= 1,"");

  rai::Frame *object = Gs(1)->frames(object_index_);

  // initialize y and J
  y.resize(1);//zeros(dim_phi(Gs, t));
  if(&J){
    uintA qidx(Gs.N);
    qidx(0)=0;
    for(uint i=1;i<Gs.N;i++) qidx(i) = qidx(i-1)+Gs(i-1)->q.N;
    J = zeros(y.N, qidx.last()+Gs.last()->q.N);
  }

  // get speed vector
  arr y_vel,Jvel;
  TM_Default vel(TMT_posDiff, object->ID); // consider (0, 0) point per default
  vel.order = 1;
  vel.__phi(y_vel, Jvel, Gs);

  // get orientation vector
  arr y_vec,Jvec;
  TM_Default vec(TMT_vec, object->ID, rai::Vector(0,1,0));
  vec.order = 0;
  vec.phi(y_vec, Jvec, *Gs(1));

  // commit results
  y(0) = scalarProduct(y_vel, y_vec);
  if(&J)
  {
    const auto offset = J.d1 / Gs.N;

    CHECK( Jvec.d1 == offset, "" );

    for(auto i = 0; i < Gs.size(); ++i)
    {
      auto JvelSub = Jvel.sub(0, -1, i * offset, (i+1) * offset -1);
      if( i == 1 )
      {
        J.setMatrixBlock( ~y_vel * Jvec + ~y_vec * JvelSub, 0, i * offset );
      }
      else
      {
        J.setMatrixBlock( ~y_vec * JvelSub, 0, i * offset );
      }
    }
  }
}

void CarKinematic::phi(arr& y, arr& J, const rai::KinematicWorld& G)
{
  CHECK(false,"The phi function taking the list of kinematic worlds should be taken");
}

uint CarKinematic::dim_phi(const rai::KinematicWorld& K)
{
  //CHECK(false,"The phi function taking the list of kinematic worlds should be taken");

  return dim_;
}
