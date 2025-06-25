#include <eigen3/Eigen/Dense>
#include <memory>
#include <math.h>
#include "ros/ros.h"
#include <ros/package.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/Twist.h"

#include <common/utility.h>
#include <komo/komo_factory.h>
#include <nodes/obstacle_common.h>
#include <komo/utility_komo.h>
#include <KOMO/komo.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include<std_msgs/Int32.h>

#include <random>


namespace
{
class TrajEvaluator
{
public:
    TrajEvaluator(int steps_per_phase, double road_width, double v_desired)
        : steps_per_phase_(steps_per_phase)
    {
        komo_ = komo_factory_.create_komo(2, steps_per_phase);
        objectives_ = komo_factory_.ground_komo(komo_, {}, road_width, v_desired);

        komo_->reset(0);
    }

    void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        // retrieve pose
        odometry_.x = msg->pose.pose.position.x;
        odometry_.y = msg->pose.pose.position.y;
        odometry_.yaw = get_yaw_from_quaternion(msg->pose.pose.orientation);

        // retrieve speeds
        odometry_.v = msg->twist.twist.linear.x;
        odometry_.vy = msg->twist.twist.linear.y;
        odometry_.omega = msg->twist.twist.angular.z;

        odo_received_ = true;
    }
    int passed_count;
        void count_callback(const std_msgs::Int32::ConstPtr& msg)
    {
      passed_count=msg->data;
    }


    void trajectory_callback(const nav_msgs::Path::ConstPtr& msg)
    {
        trajectory_ = std::vector<Pose2D>(msg->poses.size());

        for(auto i = 0; i < msg->poses.size(); ++i)
        {
            auto pose = msg->poses[i];
            trajectory_[i] = Pose2D{pose.pose.position.x, pose.pose.position.y, get_yaw_from_quaternion(pose.pose.orientation)};
        }
    }

    double car_x() const { return odometry_.x; }
    double car_y() const { return odometry_.y; }
    double car_vx() const { return odometry_.v; }
    double car_vy() const { return odometry_.vy; }
    double car_yaw() const { return odometry_.yaw; }
    double car_omega() const { return odometry_.omega; }


    double evaluate()
    {   
        //// return early if not enough info
        if( trajectory_.size() == 0 || ! odo_received_ )
        {
            return 0.0;
        }

        // get 0-0
        Pose2D current = {odometry_.x, odometry_.y, odometry_.yaw};

        // project on trajectory
        int index = -1;
        double mu = -1;

        const auto projected = project_on_trajectory(current, trajectory_, index, mu);

        //std::cout << "index:" << index << std::endl;

        if(index == -1 || index == 0)
        {
            // ROS_WARN_STREAM("wrong projection");
            return 0.0;
        }

        update_komo(trajectory_[index - 1], komo_->configurations(0));
        update_komo(trajectory_[index], komo_->configurations(1));
        update_komo(trajectory_[index + 1], komo_->configurations(2));

        auto Gs = get_traj_start(komo_->configurations, 0, 2);
        auto cost = traj_cost(Gs, {objectives_.acc_, objectives_.ax_, objectives_.vel_});
        acceleration=data_value(Gs, objectives_.acc_);


        static int n = 0;
        ++n;

        if(n >=100)
            cost_evaluator_.acc(cost.total);

        return cost_evaluator_.average();
    }

    void update_komo(const Pose2D & pose, rai::KinematicWorld* kin) const
    {
        kin->q(0) = pose.x;
        kin->q(1) = pose.y;
        kin->q(2) = pose.yaw;

        kin->calc_Q_from_q();
        kin->calc_fwdPropagateFrames();
    }
        std::vector<double> acceleration;

    double obs_x;
    double obs_y;
    void obstacle_callback(const visualization_msgs::MarkerArray::ConstPtr& msg)
{

      const auto&m = msg->markers[2 * 0 + 1];

      /// position and geometry
      obs_x=m.pose.position.x;
      obs_y= m.pose.position.y;
}
private:
    // state
    bool odo_received_;
    OdometryState odometry_;
    std::vector<Pose2D> trajectory_;
    KomoFactory komo_factory_;
    std::shared_ptr< KOMO > komo_;
    Objectives objectives_;
    Evaluator cost_evaluator_;
    

    // params
    const int steps_per_phase_;
};
}


int main(int argc, char **argv)
{
    ROS_INFO_STREAM("Launch evaluation node..");

    int steps_per_phase = 1;
    int trajectory_index = 0;
    double road_width = 3.5;
    double v_desired = 10;

    // ros init
    ros::init(argc, argv, "evaluation");
    ros::NodeHandle n;
    n.getParam("/traj_planner/steps_per_phase", steps_per_phase);
    n.getParam("/traj_controller/trajectory_index", trajectory_index);
    n.getParam("/road_width", road_width);
    n.getParam("/v_desired", v_desired);

    // logging
    std::ofstream ofs(filename(n));
    std::ofstream ofs_passed(filename_(n));
    std::ofstream ofs_time(filename__(n));

    // evaluation
    TrajEvaluator evaluator(steps_per_phase, road_width, v_desired);

    boost::function<void(const nav_msgs::Path::ConstPtr& msg)> trajectory_callback =
            boost::bind(&TrajEvaluator::trajectory_callback, &evaluator, _1);
    auto trajectory_subscriber = n.subscribe("/traj_planner/trajectory_" + std::to_string(trajectory_index), 100, trajectory_callback);

    boost::function<void(const nav_msgs::Odometry::ConstPtr& msg)> odometry_callback =
            boost::bind(&TrajEvaluator::odometry_callback, &evaluator, _1);
    auto odo_subscriber = n.subscribe("/lgp_car/odometry", 1000, odometry_callback);
    // boost::function<void(const visualization_msgs::MarkerArray::ConstPtr& msg)> obstacle_callback_tree =
    //         boost::bind(&TrajEvaluator::obstacle_callback, &evaluator, _1);
                // auto odo_subscriber = n.subscribe("/tianracer_01/odom", 1000, odometry_callback);
    boost::function<void(const visualization_msgs::MarkerArray::ConstPtr& msg)> obstacle_callback_tree =
            boost::bind(&TrajEvaluator::obstacle_callback, &evaluator, _1);
    auto obstacle_tree = n.subscribe("/lgp_obstacle_belief/marker_array_inneed", 1, obstacle_callback_tree);
        boost::function<void(const std_msgs::Int32::ConstPtr& msg)> count_callback =
            boost::bind(&TrajEvaluator::count_callback, &evaluator, _1);
    // auto passed_count_sub = n.subscribe("passed_count",1,count_callback);

    ros::Rate loop_rate(10);
    double car_vx_last, car_vy_last, car_omega_last;
    double acc_x, acc_y, acc_x_last, acc_y_last;
    double jerk_x, jerk_y;
    int i = 0;
    int it=0;
    bool timing=false;
    bool time_logged=false;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (ros::ok())
    {  
      auto car_yaw = evaluator.car_yaw();  
      auto avg = evaluator.evaluate();
      auto car_x = evaluator.car_x();
      auto car_y = evaluator.car_y();
      auto car_vx = evaluator.car_vx()*cos(car_yaw);
      //auto car_vy = evaluator.car_vy();
      auto car_vy = evaluator.car_vx()*sin(car_yaw);
      
      auto car_omega = evaluator.car_omega();
      auto distance_obs=sqrt(pow(car_x-evaluator.obs_x,2)+pow(car_y-evaluator.obs_y,2));
      auto distance_obs_long=car_x-evaluator.obs_x;
      auto distance_obs_lateral=car_y-evaluator.obs_y;
    //double distance_obs=0;
      //double distance_obs_long=0;
      //double distance_obs_lateral=0;
                it++;
      if(it>0)
      {
        acc_x=(car_vx-car_vx_last)/0.1;
        acc_y=(car_vy-car_vy_last)/0.1;

      }
    if(it>1)
      {
        jerk_x=(acc_x-acc_x_last)/0.1;
        jerk_y=(acc_y-acc_y_last)/0.1;
        it--;
      }

      ros::spinOnce();

      
      //auto acc=evaluator.acceleration;
      //if(acc[0]&&acc[1])
      
      //{ROS_INFO_STREAM("Acceleration x:"<< acc[0] << " y:" << acc[1]);}
      // logging
    //   std::cout<< "car_x: " << car_x << " car_y: " << car_y << " car_vx: " << car_vx << " car_vy: " << car_vy << " acc_x: " << acc_x << " acc_y: " << acc_y << " jerk_x: " << jerk_x << " jerk_y: " << jerk_y << std::endl;
      if(car_x>=-50&&car_x<=50&&!(car_x==0&&car_y==0&&car_vx==0&&car_vy==0))
      {
        if(!timing)
        {
          start_time = std::chrono::high_resolution_clock::now();
          timing=true;
        }
        ++i;
        log_to_file(ofs, n, car_x,  car_y,  car_vx,  car_vy,  acc_x, acc_y, jerk_x, jerk_y, car_yaw, car_omega,distance_obs,distance_obs_long,distance_obs_lateral);
      if(i%600==0)
      {
        //   log_to_file(ofs_passed, n, evaluator.passed_count);
      }
      if(i && i%100 == 0)
      {
        double time;
        n.getParam("/planning_time",time);
         ROS_INFO_STREAM("[cost evaluation] cost:" << avg << " " << i << " evaluations :)"<<"time" <<time << " car_x: " << car_x << " car_y: " << car_y << " car_vx: " << car_vx << " car_vy: " << car_vy << " acc_x: " << acc_x << " acc_y: " << acc_y << " jerk_x: " << jerk_x << " jerk_y: " << jerk_y);

         if(i%100 == 0)
          ;  // log_to_file(ofs, n, car_x, i, avg);
            
      }

      loop_rate.sleep();}
      if(car_x>=50&&!time_logged)
      {
        auto stop_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count() / 1000.0;
        log_to_file(ofs_time, n, duration);
        std::cout << "Time taken to reach 50m: " << duration << " seconds" << std::endl;
        time_logged=true;
      }
    car_vx_last=car_vx;
      car_vy_last=car_vy;
      car_omega_last=car_omega;
      if(i>0)
      {
        acc_x_last=acc_x;
        acc_y_last=acc_y;
      }
    }
    return 0;
}
