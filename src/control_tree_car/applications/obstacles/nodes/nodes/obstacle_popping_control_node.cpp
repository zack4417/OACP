#include <math.h>
#include <boost/bind.hpp>
#include <chrono>
#include <memory>
#include <stdlib.h>
#include <vector>
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>

#include <common/utility.h>
#include <nodes/obstacle_common.h>
#include <common/common.h>
#include <common/utility.h>
#include<bits/stdc++.h>

// TODO
// constraints lateral instead of centerline?

// params
double lane_width = 3.5;
const double reset_x_threshold = -0;
const double distance_ahead = 35;
//const double vanishing_false_positive_distance = 10.0;

static int n_obstacles = 0;
static int n_non_obstacles = 0;

constexpr bool publish_to_gazebo = false;

    std::vector<double> mid_lane_y = {2., -2., 0, 0.,-5., -5.,0,-2.,0+2.5,5};
    std::vector<double> mid_lane_x={-5+20, -9+20, -3+20, -1.5+20,5+20, 3+20,10+20,190+20,220+20,160+20};
    std::vector<double> mid_lane_y_ori(mid_lane_y);
    std::vector<double> mid_lane_x_ori(mid_lane_x);
    std::vector<double> mid_lane_y_bak(mid_lane_y);
    std::vector<double> mid_lane_x_bak(mid_lane_x);
    std::vector<double> velocities_x={0,0,0,0.,0.,0,0,0,0,0};
    std::vector<double> velocities_y={0,0,0,-0.6,0.6,0.,0,0,0,0};
    std::vector<int> indexes={0,1,2,3,4,5};
    std::vector<ros::Publisher> new_pose_publishers;

//static double steadily_increasing(double signed_distance)
//{
//    const auto distance = signed_distance < 0 ? 0 : signed_distance;

//    auto p = (distance_ahead - distance) / distance_ahead;

//    return p;
//}

//static double steadily_decreasing(double signed_distance)
//{
//    const auto distance = signed_distance < 0 ? 0 : signed_distance;

//    auto p = (std::max(0.0, distance - vanishing_false_positive_distance)) / distance_ahead;

//    return p;
//}

class Obstacle
{
public:
    Obstacle(uint id)
      : id_(id)
    {

    }

    virtual double existence_probability(double distance) const = 0;
    virtual Position2D get_position() const = 0;
    virtual void set_position(double x, double y){position_.x=x;position_.y=y;}
    virtual bool is_false_positive() const = 0;
    virtual bool is_visible() const {return visibility;}
    virtual void set_visibility(bool state){visibility=state; return;}
    virtual Position2D get_velocity() const {return velocity_;}
    virtual void set_velocity(double v_x, double v_y)  {velocity_.x=v_x;velocity_.y=v_y;}
    uint id_;
    bool visibility=true;
    Position2D position_{0,0};
    Position2D velocity_ = {0,0};
};

class FalsePositive : public Obstacle
{
public:
    FalsePositive(uint id, const Position2D & position, double p, double certainty_distance)
    : Obstacle(id)
    , position_(position)
    , p_(p)
    , certainty_distance_(certainty_distance)
    {

    }

    double existence_probability(double distance) const
    {
        if(distance > certainty_distance_ + 2)
        {
            return p_;
        }
        else if(distance < certainty_distance_)
        {
            return 0.0;
        }
        else
        {
            double lambda = (distance - certainty_distance_) / 2;
            return 0.0 * (1 - lambda) + p_ * (lambda);
        }
    }

    Position2D get_position() const { return position_; }
    void set_position(double x, double y) {position_.x=x; position_.y=y;}

    bool is_false_positive() const { return true; }
    bool is_visible() const {return visibility;}
    void set_visibility(bool state){visibility=state; return;}
    Position2D get_velocity() const {return velocity_;}

    void set_velocity(double v_x, double v_y)  {velocity_.x=v_x;velocity_.y=v_y;}

private:
    Position2D position_{0,0};
    double p_ = 0.5;
    double certainty_distance_ = 10.0;
    bool visibility=true;
    Position2D velocity_ = {0,0};
};

class TruePositive : public Obstacle
{
public:
    TruePositive(uint id, const Position2D & position, double p, double certainty_distance, tf::TransformListener & tf_listener)
        : Obstacle(id)
        , position_(position)
        , p_(p)
        , certainty_distance_(certainty_distance)
        , tf_listener_(tf_listener)
    {

    }

    double existence_probability(double distance) const
    {
        if(distance > certainty_distance_ + 2)
        {
            return p_;
        }
        else if(distance < certainty_distance_)
        {
            return 1.0;
        }
        else
        {
            double lambda = (distance - certainty_distance_) / 2;
            return 1.0 * (1 - lambda) + p_ * (lambda);
        }

    }

    Position2D get_position() const
    {
        if(publish_to_gazebo)
        {
            tf::StampedTransform transform;

            tf_listener_.lookupTransform("/map", "/lgp_obstacle_" + std::to_string(id_),
                                         ros::Time(0), transform);

            return Position2D{ transform(tf::Vector3(0,0,0)).x(), transform(tf::Vector3(0,0,0)).y() };
        }
        else
            return position_;
    }

    void set_position(double x, double y) {position_.x=x; position_.y=y;}

    bool is_false_positive() const { return false; }

    void set_visibility(bool state){visibility=state; return;}

    bool is_visible() const {return visibility;}

    Position2D get_velocity() const {return velocity_;}

    void set_velocity(double v_x, double v_y)  {velocity_.x=v_x;velocity_.y=v_y;}

private:
    Position2D position_{0,0};
    double p_ = 0.5;
    double certainty_distance_ = 10.0;
    tf::TransformListener & tf_listener_;
    bool visibility=true;
    Position2D velocity_ = {0,0};
};

class ObstacleObserver
{
public:
    ObstacleObserver(tf::TransformListener & tf_listener, int N)
        : tf_listener_(tf_listener)
        , obstacles_(N)
        , ref_xs_(N)
        , scale_noise_(0.2)
    {

    }

    double adjust_std_devs(double base_std_dev,int i)
    {//double adjusted_std_dev;
        auto car=get_car_position();
        auto o1_=obstacles_[i]->get_position();
        double distance=sqrt(pow(o1_.x-car.x,2)+pow(o1_.y-car.y,2));
        // Now, ensuring noise decreases as the distance decreases
        // The +0.1 prevents division by zero and ensures the factor doesn't become excessively large
        double distance_factor = 10.0 / (sqrt(distance + 0.1));  
            //Apply the distance factor in a manner that decreases noise as the factor increases
        double adjusted_std = base_std_dev / std::max(distance_factor, 1.0);  // Ensure noise doesn't increase for very close distances
        if (distance < 15.0)
        {adjusted_std = adjusted_std * 0.0;}

        return adjusted_std;}

    visualization_msgs::MarkerArray observe_obstacles(std::vector<double>noise_x,std::vector<double>noise_y,std::vector<int> indexes) const
    {
        visualization_msgs::MarkerArray markers;

        const auto car_position = get_car_position();
        
        //ROS_INFO_STREAM("Observe...");

        for(auto i = 0; i < obstacles_.size(); ++i)
        {
            double a=1;
            if (count(indexes.begin(), indexes.end(), i))
            {
                a=0;
            }
            const auto& obstacle = obstacles_[i];

            if(obstacle.get())
            {
                const auto obstacle_position = obstacle->get_position();
                const auto signed_dist_to_obstacle = obstacle_position.x - car_position.x;
                const auto lat_dist_to_obstacle = obstacle_position.y - car_position.y;
                const auto existence_probability = obstacle->existence_probability(signed_dist_to_obstacle);
                const double x = obstacle_position.x+noise_x[i];
                const double y = obstacle_position.y+noise_y[i];
                //const double x = obstacle_position.x;
                //const double y = obstacle_position.y;
//                const double sx = 2.0;
//                const double sy = 1.0;
//                const double sz = 1.5;

                const double sx = 3.9; //parked cars
                const double sy = 2.12;
                const double sz = 1.5;

//                const double sx = 1.0; // other kind of obstacle
//                const double sy = 1.0;
//                const double sz = 1.0;

                // real obstacle
                visualization_msgs::Marker obstacle_marker = create_obstacle_marker(x,
                                                                             y,
                                                                             sx,
                                                                             sy,
                                                                             sz,
                                                                             existence_probability,
                                                                             obstacle->id_,0.0,0.7,0.8,a);

                // collision - model position and geometry
                visualization_msgs::Marker collision_marker = create_collision_marker(x,
                                                                               y,
                                                                               sx,
                                                                               sy,
                                                                               sz - 0.5 - 0.5 * i,
                                                                               existence_probability,
                                                                               obstacle->id_,lat_dist_to_obstacle);
                /*visualization_msgs::Marker collision_marker = create_ellipsoid_collision_marker(x,
                                                                               y,
                                                                               sx,
                                                                               sy,
                                                                               sz - 0.5 - 0.5 * i,
                                                                               existence_probability,
                                                                               obstacle->id_);*/

                obstacle_marker.header.stamp = collision_marker.header.stamp = ros::Time::now();

                markers.markers.push_back(obstacle_marker);
                markers.markers.push_back(collision_marker);

                //ROS_INFO("Obstacle existence probability: %f %d", existence_probability, obstacle->id_);
            }
        }

        //ROS_INFO_STREAM("Finished Observe...");

        return markers;
    }

        visualization_msgs::MarkerArray observe_obstacles_indexes(std::vector<int> indexes, std::vector<double>noise_x,std::vector<double>noise_y) const
    {
        visualization_msgs::MarkerArray markers;

        const auto car_position = get_car_position();
        
        //ROS_INFO_STREAM("Observe...");

        for(auto i = 0; i < indexes.size(); ++i)
        {

            double r, g, b, a;
            const auto& obstacle = obstacles_[indexes[i]];
            
            if(obstacle.get())
            {
            auto vel=obstacle->get_velocity();
            if(i<=6)
            {
                r=150./255.;
                a=1.;
                g=150./255.;
                b=150./255.;
                if(vel.x>0)
                {
                    r=200./255.;
                    g=(abs(vel.x)*20.+200)/255.;
                    // b=abs(vel.y)*20./255.;
                }
                else if(vel.x<0)
                {
                    r=200./255;
                    g=(abs(vel.x)*20.+200)/255.;
                    // b=abs(vel.y)*20./255.;
                }
                if(vel.y>0)
                {
                    r=200./255;
                    g=200./255;
                    a=0.9;
                    // g=abs(vel.x)*20./255.;
                    b=(abs(vel.y)*20.+188)/255.;
                }
                else if(vel.y<0)
                {
                    r=200./255;
                    g=200./255;
                    a=0.9;
                    // g=abs(vel.x)*20./255.;
                    b=(abs(vel.y)*20.+188)/255.;
                }

                // a=0.8;
            }
            else{
                r=0.0;
                g=abs(vel.x)*20./255.;
                b=abs(vel.y)*20./255.;
                a=1;
            }
                const auto obstacle_position = obstacle->get_position();
                const auto signed_dist_to_obstacle = obstacle_position.x - car_position.x;
                const auto lat_dist_to_obstacle = obstacle_position.y - car_position.y;
                
                // const auto existence_probability = obstacle->existence_probability(signed_dist_to_obstacle);
                const auto existence_probability = 1;
                //double existence_probability;
                if (!obstacle->is_visible())
                {continue;}

                const double x = obstacle_position.x+noise_x[indexes[i]];
                const double y = obstacle_position.y+noise_y[indexes[i]];
                //const double x = obstacle_position.x;
                //const double y = obstacle_position.y;
//                const double sx = 2.0;
//                const double sy = 1.0;
//                const double sz = 1.5;

                //const double sx = 3.9; //parked cars
                //const double sy = 2.12;
                //const double sz = 1.5;

                const double sx = 1.5; // other kind of obstacle
                const double sy = 1.5;
                const double sz = 1.;

                // real obstacle
                visualization_msgs::Marker obstacle_marker = create_obstacle_marker(x,
                                                                             y,
                                                                             sx,
                                                                             sy,
                                                                             sz,
                                                                             existence_probability,
                                                                             obstacle->id_,r,g,b,a);

                // collision - model position and geometry
                visualization_msgs::Marker collision_marker = create_collision_marker(x,
                                                                               y,
                                                                               sx,
                                                                               sy,
                                                                               sz - 0.5 - 0.5 * i,
                                                                               existence_probability,
                                                                               obstacle->id_,lat_dist_to_obstacle);

                /*visualization_msgs::Marker collision_marker = create_ellipsoid_collision_marker(x,
                                                                               y,
                                                                               sx,
                                                                               sy,
                                                                               sz - 0.5 - 0.5 * i,
                                                                               existence_probability,
                                                                               obstacle->id_);*/
                obstacle_marker.header.stamp = collision_marker.header.stamp = ros::Time::now();

                markers.markers.push_back(obstacle_marker);
                markers.markers.push_back(collision_marker);

                //ROS_INFO("Obstacle existence probability: %f %d", existence_probability, obstacle->id_);
            }
        }

        //ROS_INFO_STREAM("Finished Observe...");

        return markers;
    }

    Position2D get_position(const std::string & frame_name) const
    {
        tf::StampedTransform transform;

        tf_listener_.lookupTransform("/map", frame_name,
                                     ros::Time(0), transform);
        // tf_listener_.lookupTransform("/map", "/tianracer_01/odom", ros::Time(0), transform);

        return Position2D{ transform(tf::Vector3(0,0,0)).x(), transform(tf::Vector3(0,0,0)).y() };
    }

    Position2D get_car_position() const
    {
        return get_position("/lgp_car");
        // return get_position("tianracer_01/odom");
    }

    std::shared_ptr<Obstacle> obstacle(uint i) const { return obstacles_[i]; }

    std::vector<std::shared_ptr<Obstacle>> obstacles() const { return obstacles_; }

    void erase_obstacle(uint i)
    {
        obstacles_[i] = nullptr;
    }

    void set_obstacle(uint i, const std::shared_ptr<Obstacle> & obstacle)
    {
        obstacles_[i] = obstacle;
        ref_xs_[i] = obstacle->get_position().x;
    }

    double ref_x(uint i) const
    {
        return ref_xs_[i];
    }

    std::vector<int> sort_obstacles(int n)
    {
        auto obstacles=obstacles_;
        auto car=get_car_position();
        std::vector<int> index;
        std::vector<int> retired;
        for(int i=0;i<6;i++)
        {int minPosition1 = min_element(obstacles.begin(),obstacles.end(),[&car](const std::shared_ptr<Obstacle> & o1, const std::shared_ptr<Obstacle> & o2)
        {  
        if(!o1)
        {return false;}
        if(!o2)
        {return true;}
        auto o1_=o1->get_position();
        auto o2_=o2->get_position();
        double dis_1=pow(o1_.x-car.x,2)+pow(o1_.y-car.y,2);
        double dis_2=pow(o2_.x-car.x,2)+pow(o2_.y-car.y,2);
        return dis_1<dis_2;}) - obstacles.begin();
        if(obstacles[minPosition1]->get_position().x-car.x>-10)
        {index.push_back(minPosition1);}
        else
        {retired.push_back(minPosition1);}
        obstacles[minPosition1]=NULL;
        }
        index.insert(index.end(),retired.begin(),retired.end());
        double radius=0.75;
        for(auto i=0; i<6;i++)
        {
        if (obstacles_[i]->get_position().x<odometry.x)
        {
            continue;
        }
        double curr_obs_x=obstacles_[i]->get_position().x;
        double curr_obs_y=obstacles_[i]->get_position().y;
        for(auto j=0;j<obstacles_.size();j++)
        {   if(j==i)
            {
                continue;
            }
            //double rel_x=(curr_obs_x-odometry.x)*cos(odometry.yaw)-(curr_obs_y-odometry.y)*sin(odometry.yaw);
            //double rel_y=(curr_obs_y-odometry.y)*cos(odometry.yaw)+(curr_obs_x+odometry.x)*sin(odometry.yaw);

            double rel_x=curr_obs_x-odometry.x;
            double rel_y=curr_obs_y-odometry.y;

            if(((rel_x*rel_y+radius*sqrt(pow(rel_x,2)+pow(rel_y,2)-pow(radius,2)))/(pow(rel_x,2)-pow(radius,2))*(obstacles_[j]->get_position().x-odometry.x)>(obstacles_[j]->get_position().y-odometry.y)) && ((rel_x*rel_y-radius*sqrt(pow(rel_x,2)+pow(rel_y,2)-pow(radius,2)))/(pow(rel_x,2)-pow(radius,2))*(obstacles_[j]->get_position().x-odometry.x)<(obstacles_[j]->get_position().y-odometry.y)) && obstacles_[j]->get_position().x>curr_obs_x)
            {obstacles_[j]->set_visibility(false);}
        }
        }
    return index;
    }

    void reset_visibility()
    {
        for(auto i=0;i<obstacles_.size();i++)
        {
            obstacles_[i]->set_visibility(true);
        }
        return;
    }

    void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg)
    {
    //ROS_INFO("odometry_callback..");

    // retrieve pose
        odometry = odometry_state_from_msg(msg);

        //odo_received_ = true;

    //ROS_ERROR(" x, y, yaw: [%f] [%f] [%f]", odometry_.x, odometry_.y, odometry_.yaw);
    //ROS_ERROR(" v, omega: [%f] [%f]", odometry_.v, odometry_.omega);
    }

    // void pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    // {
    // //ROS_INFO("odometry_callback..");

    // // retrieve pose
    //     odometry = odometry_state_from_posemsg(msg);

    //     //odo_received_ = true;

    // //ROS_ERROR(" x, y, yaw: [%f] [%f] [%f]", odometry_.x, odometry_.y, odometry_.yaw);
    // //ROS_ERROR(" v, omega: [%f] [%f]", odometry_.v, odometry_.omega);
    // }



    /*static bool cmp(const std::shared_ptr<Obsget_car_position()tacle> o1, const std::shared_ptr<Obstacle> o2){
    auto car=ObstacleObserver::get_car_position();
    const auto o1_=o1->get_position();
    const auto o2_=o2->get_position();
    int dis_1=pow(o1_.x-car.x,2)+pow(o1_.y-car.y,2);
    int dis_2=pow(o2_.x-car.x,2)+pow(o2_.y-car.y,2);
    //if (dis_1 == dis_2)
    //    return dis_1 < dis_2;
    return dis_1 > dis_2;
}*/

private:
    tf::TransformListener & tf_listener_;
    std::vector<std::shared_ptr<Obstacle>> obstacles_;
    std::vector<double> ref_xs_;

    // params
    const double scale_noise_;
    OdometryState odometry;

};

    void obs_callback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        double x=msg->pose.position.x;
        double y=msg->pose.position.z;
        mid_lane_x.push_back(x);
        mid_lane_y.push_back(y);
        // Position2D new_position = Position2D{x, y};
        // position_valid = true;
        // std::shared_ptr<Obstacle>obstacle = std::shared_ptr<Obstacle>(new FalsePositive(obstacle_id, new_position, 1, 100) );
        // auto obstacle = draw_new_obstacle(indexes[i], x,y, velocities_x[indexes[i]],0, 1, 100, observer, n, new_pose_publishers, tf_listener);
    }

static std::shared_ptr<Obstacle> draw_new_obstacle(uint obstacle_id,
                                                   double ref_x,
                                                   double p_obstacle,
                                                   double certainty_distance_offset,
                                                   const ObstacleObserver& observer,
                                                   ros::NodeHandle& n,
                                                   std::vector<ros::Publisher> & new_pose_publishers,
                                                   tf::TransformListener& tf_listener)
{
    // draw new OBSTACLE
    std::shared_ptr<Obstacle> obstacle;

    bool position_valid = false;
    Position2D new_position;


    while(!position_valid)
    {
        // X
        const double new_x = ref_x + distance_ahead + rand_m11() * distance_ahead * 0.3;

        // Y
        //const double new_y = rand_m11() * lane_width * 0.5;
        // motorbike
        //const double new_y = rand_m11() > 0 ?  lane_width * 0.5 - 0.6 * rand_01() : -lane_width * 0.5 + 0.6 * rand_01() ;

        // car
//        const double road_width = lane_width + 2.0;
//        const double y = road_width  * 0.5 + (0.4 - 1.1 * rand_01());
//        const double new_y = rand_m11() > 0 ?  y : -y;

        // central obstacle
//        const double y = 0.5 + 0.5 * lane_width * rand_01(); // video
        const double y = 0.5 + 0.5 * lane_width * rand_01(); // video
        const double new_y = rand_m11() > 0 ?  y : -y;

        new_position = Position2D{new_x, new_y};

        position_valid = true;
        for(const auto& o: observer.obstacles())
        {
            if(o.get())
            {
                const auto& other = o->get_position();

                const auto d = dist(new_position, other);

                position_valid = position_valid && d > 8.0;
            }
        }
    }

    // P
    const double p = draw_p(p_obstacle);
    const double certainty_distance = certainty_distance_offset + distance_ahead * rand_01();

    if(draw_bool(p_obstacle))
    {
        //ROS_INFO_STREAM("CREATE TRUE POSITIVE..");
        n_obstacles++;
        obstacle = std::shared_ptr<Obstacle>(new TruePositive(obstacle_id, new_position, p, certainty_distance, tf_listener) );
    }
    else
    {
        //ROS_INFO_STREAM("CREATE FALSE POSITIVE..");
        n_non_obstacles++;
        obstacle = std::shared_ptr<Obstacle>(new FalsePositive(obstacle_id, new_position, p, certainty_distance) );
    }

    if(!obstacle->is_false_positive() && publish_to_gazebo)
    {
        geometry_msgs::Pose2D msg;
        msg.x = new_position.x;
        msg.y = new_position.y;

        new_pose_publishers[obstacle_id].publish(msg);
    }

    n.setParam("/n_total_obstacles", n_obstacles);
    n.setParam("/n_total_non_obstacles", n_non_obstacles);

    return obstacle;
}

static std::shared_ptr<Obstacle> draw_new_obstacle(uint obstacle_id,
                                                   double x,
                                                   double y,
                                                   double v_x,
                                                   double v_y,
                                                   double p_obstacle,
                                                   double certainty_distance_offset,
                                                   const ObstacleObserver& observer,
                                                   ros::NodeHandle& n,
                                                   std::vector<ros::Publisher> & new_pose_publishers,
                                                   tf::TransformListener& tf_listener)
{
    // draw new OBSTACLE
    std::shared_ptr<Obstacle> obstacle;

    bool position_valid = false;
    Position2D new_position;

    //ROS_INFO_STREAM("Start drawing");
    while(!position_valid)
    {
        // X
        const double new_x = x;

        const double new_y = y;

        new_position = Position2D{new_x, new_y};

        position_valid = true;
        
        /*for(const auto& o: observer.obstacles())
        {
            if(o.get())
            {
                const auto& other = o->get_position();

                const auto d = dist(new_position, other);

                position_valid = position_valid && d > 8.0;
            }
        }*/
    }

    // P
    const double p = draw_p(p_obstacle);
    const double certainty_distance = certainty_distance_offset + distance_ahead * rand_01();

    if(draw_bool(p_obstacle))
    {
        //ROS_INFO_STREAM("CREATE TRUE POSITIVE..");
        n_obstacles++;
        obstacle = std::shared_ptr<Obstacle>(new TruePositive(obstacle_id, new_position, p, certainty_distance, tf_listener) );
    }
    else
    {
        //ROS_INFO_STREAM("CREATE FALSE POSITIVE..");
        n_non_obstacles++;
        obstacle = std::shared_ptr<Obstacle>(new FalsePositive(obstacle_id, new_position, p, certainty_distance) );
    }

    if(!obstacle->is_false_positive() && publish_to_gazebo)
    {
        geometry_msgs::Pose2D msg;
        msg.x = new_position.x;
        msg.y = new_position.y;

        new_pose_publishers[obstacle_id].publish(msg);
    }

    n.setParam("/n_total_obstacles", n_obstacles);
    n.setParam("/n_total_non_obstacles", n_non_obstacles);
    obstacle->set_velocity(v_x,v_y);
    return obstacle;
}

int main(int argc, char **argv)
{
    srand(0);

    double p_obstacle = 1;
    double certainty_distance_offset = 10.0;

    ROS_INFO_STREAM("Launch obstacle control..");

    ros::init(argc, argv, "obstacle_control_control");
    ros::NodeHandle n;
    tf::TransformListener tf_listener;

    //int N = 1; // number of obsatcles
    int N=6;
    int Nn;
    n.getParam("n_obstacles", Nn);
    n.getParam("p_obstacle", p_obstacle);
    n.getParam("road_width", lane_width);
    n.getParam("certainty_distance_offset", certainty_distance_offset);



    for(auto i = 0; i < N; ++i)
    {
        new_pose_publishers.push_back(n.advertise<geometry_msgs::Pose2D>("/lgp_obstacle_" + std::to_string(i) + "/pose_reset", 1000));
    }

    // std::vector<ros::Subscriber> obs_subs;
    // for(auto i = 0; i < 3; ++i)
    // {
    //     obs_subs.push_back(n.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_ros/obs" + std::to_string(i) + "/pose", 10,obs_callback));
    // }
    ros::Publisher marker_publisher = n.advertise<visualization_msgs::MarkerArray>("/lgp_obstacle_belief/marker_array", 10);
    ros::Publisher marker_inneed_publisher = n.advertise<visualization_msgs::MarkerArray>("/lgp_obstacle_belief/marker_array_inneed", 10);
    ros::Publisher passed_count_pub = n.advertise<std_msgs::Int32>("passed_count", 1);

    ros::Rate loop_rate(20);

    // loop variables
    ObstacleObserver observer(tf_listener, N);

    boost::function<void(const nav_msgs::Odometry::ConstPtr& msg)> odometry_callback =
        boost::bind(&ObstacleObserver::odometry_callback, &observer, _1);
    auto odo_subscriber = n.subscribe("/lgp_car/odometry", 1, odometry_callback);
    // auto odo_subscriber = n.subscribe("/tianracer_01/odom", 1, odometry_callback);
    // boost::function<void(const geometry_msgs::PoseStamped::ConstPtr& msg)> ego_pose_callback =
    //     boost::bind(&ObstacleObserver::pose_callback, &observer, _1);
    // auto ego_subscriber = n.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_ros/ego_lgp/pose", 1, ego_pose_callback);

    Position2D car_position{0}, previous_position{0};
    double speed = 0;
    //car_position = observer.get_car_position();
    //std::vector<double> mid_lane_y = {-11.25, -7.5, -3.75, 0, 3.75, -11.25, -7.5, -3.75, 0, 3.75};
    //std::vector<double> mid_lane_x={-20+car_position.x, 40+car_position.x, 100+car_position.x, 130+car_position.x, 10+car_position.x, 75+car_position.x, 150+car_position.x, 190+car_position.x, 220+car_position.x, 160+car_position.x};

    std::default_random_engine generator;
    std::normal_distribution<double> distribution_x(0.0,0.0);
    std::normal_distribution<double> distribution_y(0.0,0.0);
    std::vector<bool>passed{false,false,false,false,false,false,false,false,false,false};
    std::vector<bool>changed{false,false,false,false,false,false,false,false,false,false};
    int passed_count=0;
    std_msgs::Int32 msg_count;
    ros::Time time_now=ros::Time::now(); 
    ros::Time time_last=ros::Time::now(); 
    //bool changed;
    while(ros::ok())
    {       
        time_now=ros::Time::now(); 
            std::vector<double>adjusted_dev_x;
            std::vector<double>adjusted_dev_y;
        // observe car and reset if necessary
        try
        {
            //ROS_INFO_STREAM("................");

            car_position = observer.get_car_position();
            // std::cout<<"car_position_x="<<car_position.x<<" car_position_y="<<car_position.y<<std::endl;

            // purge old
            for(auto i = 0; i < N; ++i)
            {
                if(observer.obstacle(indexes[i]).get())
                {
                    auto obstacle_position = observer.obstacle(indexes[i])->get_position();
                    auto obstacle_velocity= observer.obstacle(indexes[i])->get_velocity();
                    if(obstacle_position.x-car_position.x<2.)
                    {
                    
                    double new_position_x=obstacle_position.x+obstacle_velocity.x*(time_now-time_last).toSec();
                    double new_position_y=obstacle_position.y+obstacle_velocity.y*(time_now-time_last).toSec();
                    // mid_lane_x[indexes[i]]=obstacle_position.x+velocities_x[indexes[i]]*(time_now-time_last).toSec();
                    // mid_lane_y[indexes[i]]=obstacle_position.y+velocities_y[indexes[i]]*(time_now-time_last).toSec();
                    
                    observer.obstacle(indexes[i])->set_position(new_position_x,new_position_y);
                    }
                    obstacle_position = observer.obstacle(indexes[i])->get_position();
                    const auto signed_dist_to_obstacle = obstacle_position.x - car_position.x;
                    // mid_lane_x[indexes[i]]=obstacle_position.x;
                    // mid_lane_y[indexes[i]]=obstacle_position.y;

                    // mid_lane_x_ori[indexes[i]]=mid_lane_x_ori;
                    // mid_lane_y_ori[indexes[i]]=obstacle_position.y;
                    if(obstacle_position.x-car_position.x<0&&passed[indexes[i]]==false)
                    {
                        passed_count++;
                        passed[indexes[i]]=true;
                    }
                    if(obstacle_position.x < car_position.x-5)
                    {
                        observer.erase_obstacle(indexes[i]);
                        //passed_count++;
                        passed[indexes[i]]=false;
                        //mid_lane_x[indexes[i]]=150+5+car_position.x;
                        //mid_lane_x[indexes[i]]+=150+5-50-6;
                        mid_lane_x_ori[indexes[i]]+=60;
                        mid_lane_x[indexes[i]]=mid_lane_x_ori[indexes[i]];
                        mid_lane_y[indexes[i]]=mid_lane_y_ori[indexes[i]];
                        // mid_lane_x[indexes[i]]+=60;
                        //ROS_INFO_STREAM("erased "<<i);
                        //if(indexes[i]==2)
                        //{
                            changed[indexes[i]]=true;
                        //}
                        continue;
                    }
                    if((obstacle_position.x < car_position.x+20) && (indexes[i]==2||indexes[i]==3||indexes[i]==5))
                    {
                        if(changed[indexes[i]])
                        {std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_int_distribution<int> distribution(0, 1);

                        int random_number = distribution(gen);

                        std::cout << random_number << std::endl;
                        changed[indexes[i]]=false;
                        // if(random_number==0)
                        // {observer.erase_obstacle(indexes[i]);
                        // //passed_count++;
                        // passed[indexes[i]]=false;
                        // //mid_lane_x[indexes[i]]=150+5+car_position.x;
                        // //mid_lane_x[indexes[i]]+=150+5-50-6;
                        // mid_lane_x[indexes[i]]+=60;
                        // changed[indexes[i]]=true;}
                        }
                        //ROS_INFO_STREAM("erased "<<i);
                    }

//                    ROS_INFO_STREAM("speed:" << speed << " signed_dist_to_obstacle:" << signed_dist_to_obstacle);

                    speed = (car_position.x - previous_position.x) * 10;
                    // if(speed < 0.1 && signed_dist_to_obstacle < 1.5) // avoid car stuck to obstacle
                    // {
                        // ROS_WARN_STREAM("Erase obstacle because car seems stuck!!!!");
                        // observer.erase_obstacle(indexes[i]);
                        //passed_count++;
                        // mid_lane_x_ori[indexes[i]]+=60;
                        // mid_lane_x_ori[indexes[i]]+=15*(i%3);
                    // }
                    // observer.erase_obstacle(i);
                }
            }

            previous_position = car_position;

            // recreate new
            for(auto i = 0; i < N; ++i)
            {
                if(!observer.obstacle(indexes[i]).get())
                {   //ROS_INFO_STREAM("Draw new "<<i);
                    //mid_lane_x[i]+=150;
                    //auto obstacle = draw_new_obstacle(i, observer.ref_x(i), p_obstacle, certainty_distance_offset, observer, n, new_pose_publishers, tf_listener);
                    // auto obstacle = draw_new_obstacle(indexes[i], mid_lane_x[indexes[i]],mid_lane_y[indexes[i]], p_obstacle, certainty_distance_offset, observer, n, new_pose_publishers, tf_listener);
                    auto obstacle = draw_new_obstacle(indexes[i], mid_lane_x_ori[indexes[i]],mid_lane_y_ori[indexes[i]], velocities_x[indexes[i]],velocities_y[indexes[i]], p_obstacle, certainty_distance_offset, observer, n, new_pose_publishers, tf_listener);
                    //ROS_INFO_STREAM("Drawed new "<<i);
                    observer.set_obstacle(indexes[i], obstacle);
                    
                }
            }
            for (auto i = 0; i<N; ++i)
            {if(!observer.obstacle(i).get())
            {
                adjusted_dev_x.push_back(0.);
                adjusted_dev_y.push_back(0.);
                continue;
            }
                double std_dev_x=distribution_x(generator);
                double std_dev_y=distribution_y(generator);
                adjusted_dev_x.push_back(observer.adjust_std_devs(std_dev_x,i));
                adjusted_dev_y.push_back(observer.adjust_std_devs(std_dev_y,i));
                //std::cout<<"dev_x"<<observer.adjust_std_devs(std_dev_x,i)<<"dev_y"<<observer.adjust_std_devs(std_dev_y,i)<<std::endl;
                //adjusted_dev_x.push_back(0.0);
                //adjusted_dev_y.push_back(0.0);
            }
            // publish observation
            //ROS_INFO_STREAM("Finish Draw");
            observer.reset_visibility();
            indexes=observer.sort_obstacles(Nn);
            //auto markers = observer.observe_obstacles(adjusted_dev_x,adjusted_dev_y,indexes);
            auto markers_inneed= observer.observe_obstacles_indexes(indexes,adjusted_dev_x,adjusted_dev_y);

            marker_publisher.publish(markers_inneed);
            marker_inneed_publisher.publish(markers_inneed);
            msg_count.data=passed_count;
            passed_count_pub.publish(msg_count);
            observer.reset_visibility();
            time_last=ros::Time::now();
            //ROS_INFO_STREAM("Obstacle Updated");
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }

        ros::spinOnce();

        loop_rate.sleep();
    }

    return 0;
}
