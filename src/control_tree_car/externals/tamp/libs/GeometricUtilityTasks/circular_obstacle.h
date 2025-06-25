#pragma once

#include <KOMO/komo.h>
#include <geom_utility.h>

struct Obstacle
{
  arr position;
  double p;
  double radius;
  bool visibility=true;
  double distance;
  bool is_visible() const {return visibility;}

  Obstacle(){}
  Obstacle(const arr & _position, double _p): position(_position), p(_p){}
  Obstacle(const arr & _position, double _radius, double _distance)
    : position(_position)
    , radius((_distance/1.7)*1.+_radius)
    , distance(_distance)
    , p(1)
    {}
  void adjust_radius(double scale)
  {
    radius=radius-(distance/1.7)*1.+(distance/1.7*1.*scale);
    // std::cout<<"distance="<<distance<<"distance/2*scale="<<distance/2.*scale<<std::endl;
    // std::cout<<"scale="<<scale<<" adjust_radius="<<radius<<std::endl;
  }
  
};

struct CircularObstacle:Feature{

  CircularObstacle( const std::string & agent_object, const arr & position, double radius, const rai::KinematicWorld& G)
    : object_index_(getFrameIndex(G, agent_object))
    , position_( position )
    , radius_( radius )
    , car_circle_radius_(2.15)
    , circle_relative_position_(1.25, 0, 0)
  {
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("CarObstacle");
  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) override
  {
    rai::Frame *object = G.frames(object_index_);
    // init
    y.resize(1);//zeros(dim_phi(Gs, t));
    y(0) = 0;
    if(&J)
    {
      J = zeros(dim_, G.q.d0);
    }

    // compute cost
    punctual_phi(object, circle_relative_position_, y, J, G);
  }

  void punctual_phi(rai::Frame * object, const rai::Vector & rel, arr& y, arr& J, const rai::KinematicWorld& G) const
  {
      // compute cost
      arr pos, Jpos;
      G.kinematicsPos(pos, Jpos, object, rel);

      //
      //pos = G.q;
      //Jpos = diag(1, 3);
      //

      const auto dist = sqrt(pow(pos(0)-position_(0), 2.0) + pow(pos(1)-position_(1), 2.0));
      y(0) += - dist + radius_ + car_circle_radius_;

      //std::cout << "pos " << pos(0) << " " << pos(1) << std::endl;
      //std::cout << "y(0)" << y(0) << std::endl;

      if(&J)
      {
        auto theta = std::atan2(pos(1)-position_(1), pos(0)-position_(0));

        J(0, 0) += - cos(theta) * Jpos(0, 0);
        J(0, 1) += - sin(theta) * Jpos(1, 1);
      }
  }

  virtual uint dim_phi(const rai::KinematicWorld& K) override
  {
    return dim_;
  }

  void set_obstacle_position(const arr & position)
  {
      position_ = position;
  }

private:
  static const uint dim_ = 1;
  const uint object_index_;
  const double radius_;
  arr position_;
  rai::Vector circle_relative_position_;
  const double car_circle_radius_;
};



// struct Car3CirclesCircularObstacle:Feature{

//   Car3CirclesCircularObstacle( const std::string & agent_object, const std::vector<Obstacle> & obstacles, const rai::KinematicWorld& G )
//     : dim_(3* obstacles.size())
//     , object_index_(getFrameIndex(G, agent_object))
//     , car_circle_radius_(0.2) //1.4   0.5
//     , obstacles_( obstacles )
//   {
//       circle_relative_positions_[0] = (1., 0, 0);//1.25   2.15
//       circle_relative_positions_[1] = (-1., 0., 0); //-0.9  -1.9
//       circle_relative_positions_[2] = (0., 0, 0);
//   }

//   virtual rai::String shortTag(const rai::KinematicWorld& G)
//   {
//     return rai::String("Car3CirclesCircularObstacle");
//   }

//   virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) override
//   {
//     rai::Frame *object = G.frames(object_index_);
//     // init
//     //int dim=3*obstacles_.size();
//     y.resize(dim_);//zeros(dim_phi(Gs, t));
//     y(0) = 0;
//     if(&J)
//     {
//       J.resize(dim_, G.q.d0);
//     }

//     // compute cost
//     for(auto i = 0; i < circle_relative_positions_.size(); ++i)
//     {
//       punctual_phi(object, circle_relative_positions_[i], i, y, J, G);
//     }
//   }

//   void punctual_phi(rai::Frame * object, const rai::Vector & rel, std::size_t i, arr& y, arr& J, const rai::KinematicWorld& G) const
//   {
//       // compute cost
//       arr pos, Jpos;
//       G.kinematicsPos(pos, Jpos, object, rel);

//       for(auto j = 0; j < obstacles_.size(); ++j)
//       {
//         uint I = j * 3 + i;
//         const auto& position = obstacles_[j].position;
//         const auto& radius = obstacles_[j].radius;
//         const auto dist = sqrt(pow(pos(0)-position(0), 2.0) + pow(pos(1)-position(1), 2.0))+0.1;
//         y(I) = - dist + radius + car_circle_radius_;

//         //std::cout << "pos " << pos(0) << " " << pos(1) << std::endl;
//         //std::cout << "y(0)" << y(0) << std::endl;

//         if(&J)
//         {
//           const auto theta = std::atan2(pos(1)-position(1), pos(0)-position(0));

//           J(I, 0) = - cos(theta) * Jpos(0, 0);
//           J(I, 1) = - sin(theta) * Jpos(1, 1);
//           J(I, 2) = 0;
//         }
//       }
//   }

//   virtual uint dim_phi(const rai::KinematicWorld& K) override
//   {
//       return dim_;
//       //return 3*obstacles_.size();
//   }

//   void set_obstacles(std::vector<Obstacle> & obstacles)
//   {
//       //CHECK_EQ(obstacles_.size(), obstacles.size(), "should have the same number of obstacles");
//       obstacles_ = obstacles;
//   }

//   void set_obstacles_radius(double radius)
//   {
//       //CHECK_EQ(obstacles_.size(), obstacles.size(), "should have the same number of obstacles");
//       for(auto i=3;i<obstacles_.size();i++)
//       {
//         obstacles_[i].radius/=radius;
//       }
//   }

// private:
//   const uint dim_;
//   const uint object_index_;
//   const double car_circle_radius_;
//   std::array<rai::Vector, 3> circle_relative_positions_;
//   std::vector<Obstacle> obstacles_;
// };

struct Car3CirclesCircularObstacle:Feature{

  Car3CirclesCircularObstacle( const std::string & agent_object, const std::vector<Obstacle> & obstacles, const rai::KinematicWorld& G )
    : dim_(obstacles.size())
    , object_index_(getFrameIndex(G, agent_object))
    , car_circle_radius_(4.5) //1.4   0.5
    , obstacles_( obstacles )
  {
      // circle_relative_positions_[0] = (1., 0, 0);//1.25   2.15
      // circle_relative_positions_[1] = (-1., 0., 0); //-0.9  -1.9
      // circle_relative_positions_[2] = (0., 0, 0);
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("Car3CirclesCircularObstacle");
  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) override
  {
    rai::Frame *object = G.frames(object_index_);
    // init
    //int dim=3*obstacles_.size();
    y.resize(dim_);//zeros(dim_phi(Gs, t));
    y(0) = 0;
    if(&J)
    {
      J.resize(dim_, G.q.d0);
    }

    // compute cost
      punctual_phi(object, y, J, G);
  }

  void punctual_phi(rai::Frame * object, arr& y, arr& J, const rai::KinematicWorld& G) const
  {
      // compute cost
      arr pos, Jpos;
      G.kinematicsPos(pos, Jpos, object);

      for(auto j = 0; j < obstacles_.size(); ++j)
      {
        uint I = j;
        const auto& position = obstacles_[j].position;
        const auto& radius = obstacles_[j].radius;
        const auto dist = sqrt(pow(pos(0)-position(0), 2.0) + pow(pos(1)-position(1), 2.0));
        y(I) = - dist + radius + car_circle_radius_;

        //std::cout << "pos " << pos(0) << " " << pos(1) << std::endl;
        //std::cout << "y(0)" << y(0) << std::endl;

        if(&J)
        {
          const auto theta = std::atan2(pos(1)-position(1), pos(0)-position(0));

          J(I, 0) = - cos(theta) * Jpos(0, 0);
          J(I, 1) = - sin(theta) * Jpos(1, 1);
          J(I, 2) = 0;
        }
      }
  }

  virtual uint dim_phi(const rai::KinematicWorld& K) override
  {
      return dim_;
      //return 3*obstacles_.size();
  }

  void set_obstacles(std::vector<Obstacle> & obstacles)
  {
      //CHECK_EQ(obstacles_.size(), obstacles.size(), "should have the same number of obstacles");
      obstacles_ = obstacles;
  }

  void set_obstacles_radius(double radius)
  {
      //CHECK_EQ(obstacles_.size(), obstacles.size(), "should have the same number of obstacles");
      for(auto i=3;i<obstacles_.size();i++)
      {
        obstacles_[i].radius/=radius;
      }
  }

private:
  const uint dim_;
  const uint object_index_;
  const double car_circle_radius_;
  // std::array<rai::Vector, 3> circle_relative_positions_;
  std::vector<Obstacle> obstacles_;
};


struct Car3CirclesCircularRiskzone:Feature{

  Car3CirclesCircularRiskzone( const std::string & agent_object, const std::vector<Obstacle> & obstacles, const rai::KinematicWorld& G )
    : dim_(obstacles.size())
    , object_index_(getFrameIndex(G, agent_object))
    , car_circle_radius_(0.2) //1.4   0.5
    , obstacles_( obstacles )
  {
      // circle_relative_positions_[0] = (1., 0, 0);//1.25   2.15
      // circle_relative_positions_[1] = (-1., 0., 0); //-0.9  -1.9
      // circle_relative_positions_[2] = (0., 0, 0);
  }

  virtual rai::String shortTag(const rai::KinematicWorld& G)
  {
    return rai::String("Car3CirclesCircularRiskzone");
  }

  virtual void phi(arr& y, arr& J, const rai::KinematicWorld& G) override
  {
    rai::Frame *object = G.frames(object_index_);
    // init
    //int dim=3*obstacles_.size();
    y.resize(dim_);//zeros(dim_phi(Gs, t));
    y(0) = 0;
    if(&J)
    {
      J.resize(dim_, G.q.d0);
    }

    // compute cost
      punctual_phi(object, y, J, G);
  }

  void punctual_phi(rai::Frame * object, arr& y, arr& J, const rai::KinematicWorld& G) const
  {
      // compute cost
      arr pos, Jpos;
      G.kinematicsPos(pos, Jpos, object);

      for(auto j = 0; j < obstacles_.size(); ++j)
      {
        uint I = j;
        const auto& position = obstacles_[j].position;
        const auto& radius = obstacles_[j].radius;
        // std::cout<<"riskzone_radius="<<radius<<std::endl;
        const auto dist = sqrt(pow(pos(0)-position(0), 2.0) + pow(pos(1)-position(1), 2.0));
        y(I) = - dist + radius + car_circle_radius_;

        //std::cout << "pos " << pos(0) << " " << pos(1) << std::endl;
        //std::cout << "y(0)" << y(0) << std::endl;

        if(&J)
        {
          const auto theta = std::atan2(pos(1)-position(1), pos(0)-position(0));

          J(I, 0) = - cos(theta) * Jpos(0, 0);
          J(I, 1) = - sin(theta) * Jpos(1, 1);
          J(I, 2) = 0;
        }
      }
  }

  virtual uint dim_phi(const rai::KinematicWorld& K) override
  {
      return dim_;
      //return 3*obstacles_.size();
  }

  void set_obstacles(std::vector<Obstacle> & obstacles)
  {
      //CHECK_EQ(obstacles_.size(), obstacles.size(), "should have the same number of obstacles");
      obstacles_ = obstacles;
  }

  void set_obstacles_radius(double radius)
  {
      //CHECK_EQ(obstacles_.size(), obstacles.size(), "should have the same number of obstacles");
      for(auto i=3;i<obstacles_.size();i++)
      {
        obstacles_[i].radius/=radius;
      }
  }

private:
  const uint dim_;
  const uint object_index_;
  const double car_circle_radius_;
  // std::array<rai::Vector, 3> circle_relative_positions_;
  std::vector<Obstacle> obstacles_;
};
