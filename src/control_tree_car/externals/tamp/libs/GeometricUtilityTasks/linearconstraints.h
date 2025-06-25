#pragma once

#include <KOMO/komo.h>
#include <geom_utility.h>
// struct LinearRiskzones
// {
//   arr position;
//   double k1;
//   double k2;
//   double p;
//   double radius;
//   bool visibility=true;
//   double distance;
//   bool is_visible() const {return visibility;}

//   Obstacle(){}
//   Obstacle(const arr & _position, double _p): position(_position), p(_p){}
//   Obstacle(const arr & _position, double _radius, double _distance)
//     : position(_position)
//     , radius((_distance/2.)*1.5+_radius)
//     , distance(_distance)
//     , p(1)
//     {}
//   void adjust_radius(int scale)
//   {
//     radius=radius-distance/2.*1.5+distance/2.*1.5*scale;
//   }
  
// };


struct linearconstraints:Feature{

  linearconstraints( const std::string & object_name,double m1, double c1, double m2, double c2, double x,double y, double current_obs_x, const rai::KinematicWorld& G )
    : object_index_(getFrameIndex(G, object_name))
    , m1_(m1), c1_(c1), m2_(m2), c2_(c2),current_x(x),current_y(y),obs_x(current_obs_x)
  {

  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G)
  {
    rai::Frame * object = G.frames(object_index_);//G.getFrameByName( object_.c_str() ); // avoid doing that all the time (save frma index)
    arr posObject, posJObject;
    G.kinematicsPos(posObject, posJObject, object, offset_);    // get function to minimize and its jacobian in state G
    //
    //posObject = G.q;
    //posJObject = diag(1, 3);
    //
    // y.resize( dim_ );
    // y( 0 ) = posObject( 1 ) > max_y_ ? posObject( 1 ) - max_y_ : 0; // left
    // y( 1 ) = posObject( 1 ) < -max_y_ ? -posObject( 1 ) + max_y_ : 0; // right
    //y( 2 ) = 0; //posObject( 1 ); // centerline
    y.resize(dim_);
    y(0) = (posObject(1)-current_y) - (m1_ * (posObject(0)-current_x) + c1_); // 上界约束：y >= m1 * x + c1
    y(1) = (m2_ * (posObject(0)-current_x) + c2_) - (posObject(1)-current_y); // 下界约束：y <= m2 * x + c2
    y(2) = -(-posObject(0)+obs_x);
    // y(3) = posObject(0)-(obs_x+5);

    if(&J) // jacobian
    {
//       J.resize( dim_, posJObject.dim(1) );

// //      if(posObject( 1 ) > max_y_) // left
// //        J(0, 1) = 1; // left
// //      else
// //        J(0, 1) = 0;
// //      J(0,0) = 0; J(0,2) = 0;

// //      if(posObject( 1 ) < -max_y_) // right
// //        J(1, 1) = 1; // left
// //      else
// //        J(1, 1) = 0;
// //      J(1,0) = 0; J(1,2) = 0;

//       if(posObject( 1 ) > max_y_) // left
//         J.setMatrixBlock( posJObject.row( 1 ), 0 , 0 ); // left
//       else
//         J.setMatrixBlock( zeros(1, J.d1), 0 , 0 );

//       if(posObject( 1 ) < -max_y_) // right
//         J.setMatrixBlock( -posJObject.row( 1 ), 1 , 0 ); // right
//       else
//         J.setMatrixBlock( zeros(1, J.d1), 1 , 0 );

//       //J.setMatrixBlock( posJObject.row( 1 ), 2 , 0 ); // centerline
        // 计算雅可比矩阵
        J.resize(dim_, posJObject.dim(1));
        J.setZero();
        J(0, 0) = -m1_*posJObject(0,0); // 对于第一个约束，x的偏导数为-m1
        J(0, 1) = 1*posJObject(0,1);    // 对于第一个约束，y的偏导数为1
        J(0, 2) = 0;    
        J(1, 0) = m2_*posJObject(1,0);  // 对于第二个约束，x的偏导数为m2
        J(1, 1) = -1*posJObject(1,1);   // 对于第二个约束，y的偏导数为-1
        J(1, 2) = 0;   
        J(2, 0) = 1*posJObject(2,0);   // 对于第三个约束，x的偏导数为-1
        J(2, 1) = 0*posJObject(2,1);    // 对于第三个约束，y的偏导数为0
        J(2, 2) = 0;    
        // J(3, 0) = 1*posJObject(3,0);   // 对于第四个约束，x的偏导数为1
        // J(3, 1) = 0*posJObject(3,1);    // 对于第三个约束，y的偏导数为0
        // J(3, 2) = 0;   
    }
  }

  virtual uint dim_phi(const rai::KinematicWorld& G)
  {
    return dim_;
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("RoadBound");
  }

  void set_parameters(double m1,double c1,double m2,double c2,double x,double y,double obstacle_x)
  {
    m1_=m1;
    c1_=c1;
    m2_=m2;
    c2_=c2;
    current_x=x;
    current_y=y;
    obs_x=obstacle_x;
  }

private:
  static const uint dim_ = 3; //3; (with x line)
  double m1_, c1_; // 第一条直线的斜率和截距
  double m2_, c2_; // 第二条直线的斜率和截距
  double current_x, current_y,obs_x;
  const rai::Vector offset_{4.0, 0, 0}; // part of the vehicle which should stay within the boundaries (here car front)
  uint object_index_{0};
};
