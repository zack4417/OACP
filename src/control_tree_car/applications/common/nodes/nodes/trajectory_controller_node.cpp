#include <eigen3/Eigen/Dense>

#include "ros/ros.h"
#include <ros/package.h>

#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseStamped.h"
#include "ackermann_msgs/AckermannDrive.h"
#include "ackermann_msgs/AckermannDriveStamped.h"
#include <common/utility.h>
#include <tf/transform_broadcaster.h>

using MsgsType = std::tuple<geometry_msgs::Twist, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>;
using MsgsType_2 = std::tuple<ackermann_msgs::AckermannDriveStamped, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>;

namespace
{
geometry_msgs::PoseStamped to_pose_msg(const Pose2D& pose)
{
    tf2::Quaternion q; q.setRPY(0, 0, pose.yaw);
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = pose.x;
    pose_msg.pose.position.y = pose.y;
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();

    return pose_msg;
}

nav_msgs::Odometry to_odometry_msg(const Pose2D& pose, geometry_msgs::Twist& twist_msg)
{
    tf2::Quaternion q; q.setRPY(0, 0, pose.yaw);
    nav_msgs::Odometry odometry_msg;
    odometry_msg.header.stamp = ros::Time::now();
    odometry_msg.header.frame_id = "map";
    odometry_msg.child_frame_id = "lgp_car";
    odometry_msg.pose.pose.position.x = pose.x;
    odometry_msg.pose.pose.position.y = pose.y;
    odometry_msg.pose.pose.orientation.x = q.x();
    odometry_msg.pose.pose.orientation.y = q.y();
    odometry_msg.pose.pose.orientation.z = q.z();
    odometry_msg.pose.pose.orientation.w = q.w();
    odometry_msg.twist.twist.linear.x=twist_msg.linear.x;
    odometry_msg.twist.twist.angular.z=twist_msg.angular.z;
    // std::cout<<"x: "<<pose.x<<"y: "<<pose.y<<"orientation: "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<std::endl;
    return odometry_msg;
}

nav_msgs::Odometry init_odometry_msg()
{
    tf2::Quaternion q; q.setRPY(0, 0, 0);
    nav_msgs::Odometry odometry_msg;
    odometry_msg.header.stamp = ros::Time::now();
    odometry_msg.header.frame_id = "map";
    odometry_msg.child_frame_id = "lgp_car";
    odometry_msg.pose.pose.position.x = -20;
    odometry_msg.pose.pose.position.y = 0;
    odometry_msg.pose.pose.orientation.x = q.x();
    odometry_msg.pose.pose.orientation.y = q.y();
    odometry_msg.pose.pose.orientation.z = q.z();
    odometry_msg.pose.pose.orientation.w = q.w();
    odometry_msg.twist.twist.linear.x=0;
    odometry_msg.twist.twist.angular.z=0;

    return odometry_msg;
}

nav_msgs::Odometry odo_odometry_msg(OdometryState odo)
{
    tf2::Quaternion q; q.setRPY(0, 0, odo.yaw);
    nav_msgs::Odometry odometry_msg;
    odometry_msg.header.stamp = ros::Time::now();
    odometry_msg.header.frame_id = "map";
    odometry_msg.child_frame_id = "lgp_car";
    odometry_msg.pose.pose.position.x = odo.x;
    odometry_msg.pose.pose.position.y = odo.y;
    odometry_msg.pose.pose.orientation.x = q.x();
    odometry_msg.pose.pose.orientation.y = q.y();
    odometry_msg.pose.pose.orientation.z = q.z();
    odometry_msg.pose.pose.orientation.w = q.w();
    odometry_msg.twist.twist.linear.x=0;
    odometry_msg.twist.twist.angular.z=0;

    return odometry_msg;
}

}

class TrajectoryController
{
public:
    TrajectoryController(int steps_per_phase)
        : steps_per_phase_(steps_per_phase)
    {

    }

    void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        // retrieve pose
        odometry_.x = msg->pose.pose.position.x;
        odometry_.y = msg->pose.pose.position.y;
        odometry_.yaw = get_yaw_from_quaternion(msg->pose.pose.orientation);

        // retrieve speeds
        odometry_.v = msg->twist.twist.linear.x;
        odometry_.omega = msg->twist.twist.angular.z;

        odo_received_ = true;
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

    MsgsType create_control() const
    {   
        const auto nominal_v = 7; // scale w according to the ratio v / nominal_v
        static double d_int = 0;

        //// return early if not enough info
        if( trajectory_.size() == 0 || ! odo_received_ )
        {
            return std::tuple<geometry_msgs::Twist, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>(geometry_msgs::Twist(), geometry_msgs::PoseStamped(), geometry_msgs::PoseStamped());
        }

        // get 0-0
        Pose2D current = {odometry_.x, odometry_.y, odometry_.yaw};

        // project on trajectory
        int index = -1;
        double mu = -1;

        const auto projected = project_on_trajectory(current, trajectory_, index, mu);
        // if(index<2)
        // {index=2;}
        if(index <= 0)
        {
            geometry_msgs::Twist twist_msg;
            twist_msg.linear.x = 0;
            twist_msg.angular.z = 0;
            Pose2D pose_ctrl={current.x,current.y,current.yaw};
            // std::cout<<"index "<<index<<std::endl;
            return std::tuple<geometry_msgs::Twist, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>
                    (twist_msg, geometry_msgs::PoseStamped(), to_pose_msg(trajectory_[2]));
        }

        /// v
        int k = index + 1;
        int l = k + 1;
        if(l >= trajectory_.size())
        {
            l = trajectory_.size() - 1;
            k = l - 1;
        }
        double v = sqrt(pow(trajectory_[l].x - trajectory_[k].x, 2) + pow(trajectory_[l].y - trajectory_[k].y, 2)) * steps_per_phase_;
        // double v = sqrt(pow(trajectory_[4].x - trajectory_[3].x, 2) + pow(trajectory_[4].y - trajectory_[3].y, 2)) * steps_per_phase_;

        /// w
        double w_cmd = 0;
        // double omega=(trajectory_[4].yaw - trajectory_[3].yaw)*steps_per_phase_;
        // distance between current pose and its projection on the trajectory
        Eigen::Vector2f u(current.x - projected.x, current.y - projected.y);
        Eigen::Vector2f n(-sin(projected.yaw), cos(projected.yaw));
        const auto d = n.dot(u);
        d_int += d / 30.0;

        //std::cout << "d:" << d << " d_int:" << d_int << std::endl;

        // angle
        double d_yaw = projected.yaw - current.yaw;

        w_cmd = 3 * (-0.1 * d - 0.02 * d_int + d_yaw);
        // w_cmd = w_cmd * (std::max(v, 0.0) / nominal_v);
        w_cmd = w_cmd * (std::max(v, 0.5) / nominal_v);

        // scaling
        const double max_w = 1.3;
        if(std::fabs(w_cmd) > max_w)
        {
            //std::cout << "limit omega!" << w_cmd << std::endl;
            //v *= max_w / std::fabs(w_cmd);
            w_cmd = w_cmd > 0 ? max_w : -max_w;
        }

        ///
        auto target_pose_msg = to_pose_msg(projected);

        geometry_msgs::Twist twist_msg;
        if(v>10)
        {
            v=10;  
        }
        if(odometry_.x<-50)
        {
            v=5;
        }
        twist_msg.linear.x = v;
        // std::cout<<"v: "<<v<<std::endl;
        twist_msg.angular.z = w_cmd;
        // twist_msg.angular.z=omega*0.25;
        // Pose2D pose_ctrl={current.x+v*0.25*cos(current.yaw),current.y+v*0.25*sin(current.yaw),current.yaw+omega*0.25};
        // std::cout<<"vx="<<v*0.25*cos(current.yaw)<<"vy="<<v*0.25*sin(current.yaw)<<std::endl;
        // std::cout<<"current x: "<<current.x<<" current y: "<<current.y<<std::endl;
        // std::cout<<" target x"<<trajectory_[3].x<<" target y"<<trajectory_[3].y<<std::endl;
        // std::cout<<" target x"<<trajectory_[4].x<<" target y"<<trajectory_[4].y<<std::endl;
        // std::cout<<" target x"<<trajectory_[5].x<<" target y"<<trajectory_[5].y<<std::endl;
        // std::cout<<" target x"<<trajectory_[6].x<<" target y"<<trajectory_[6].y<<std::endl;
        // std::cout<<" target x"<<trajectory_[7].x<<" target y"<<trajectory_[7].y<<std::endl;
        return std::make_tuple(twist_msg, target_pose_msg, to_pose_msg(trajectory_[2]));
    }

        MsgsType_2 create_control_AckermannDrive() const
    {   
        const auto nominal_v = 1; // scale w according to the ratio v / nominal_v
        static double d_int = 0;

        //// return early if not enough info
        if( trajectory_.size() == 0 || ! odo_received_ )
        {
            return std::tuple<ackermann_msgs::AckermannDriveStamped, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>(ackermann_msgs::AckermannDriveStamped(), geometry_msgs::PoseStamped(), geometry_msgs::PoseStamped());
        }

        // get 0-0
        Pose2D current = {odometry_.x, odometry_.y, odometry_.yaw};

        // project on trajectory
        int index = -1;
        double mu = -1;

        const auto projected = project_on_trajectory(current, trajectory_, index, mu);
        // if(index<2)
        // {index=2;}
        if(index == -1)
        {

            ackermann_msgs::AckermannDriveStamped ackermann_msg;
            ackermann_msg.drive.speed = 0;
            ackermann_msg.drive.steering_angle = 0;
            // geometry_msgs::Twist twist_msg;
            // twist_msg.linear.x = 0;
            // twist_msg.angular.z = 0;
            Pose2D pose_ctrl={current.x,current.y,current.yaw};
            // std::cout<<"index "<<index<<std::endl;
            return std::tuple<ackermann_msgs::AckermannDriveStamped, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>
                    (ackermann_msg, geometry_msgs::PoseStamped(), to_pose_msg(trajectory_[2]));
        }

        /// v
        int k = index + 1;
        int l = k + 1;
        if(l >= trajectory_.size())
        {
            l = trajectory_.size() - 1;
            k = l - 1;
        }
        double v = sqrt(pow(trajectory_[l].x - trajectory_[k].x, 2) + pow(trajectory_[l].y - trajectory_[k].y, 2)) * steps_per_phase_;
        // double v = sqrt(pow(trajectory_[4].x - trajectory_[3].x, 2) + pow(trajectory_[4].y - trajectory_[3].y, 2)) * steps_per_phase_;

        /// w
        double steering_angle_cmd = 0;
        double w_cmd = 0;
        // double omega=(trajectory_[4].yaw - trajectory_[3].yaw)*steps_per_phase_;
        // distance between current pose and its projection on the trajectory
        Eigen::Vector2f u(current.x - projected.x, current.y - projected.y);
        Eigen::Vector2f n(-sin(projected.yaw), cos(projected.yaw));
        const auto d = n.dot(u);
        d_int += d / 30.0;

        //std::cout << "d:" << d << " d_int:" << d_int << std::endl;

        // angle
        double d_yaw = projected.yaw - current.yaw;

        w_cmd = 3 * (-0.1 * d - 0.02 * d_int + d_yaw);
        // w_cmd = w_cmd * (std::max(v, 0.0) / nominal_v);
        w_cmd = w_cmd * (std::max(v, 0.5) / nominal_v);

        // scaling
        const double max_w = 1.;
        if(std::fabs(w_cmd) > max_w)
        {
            //std::cout << "limit omega!" << w_cmd << std::endl;
            //v *= max_w / std::fabs(w_cmd);
            w_cmd = w_cmd > 0 ? max_w : -max_w;
        }

        ///
        // auto target_pose_msg = to_pose_msg(projected);

        // geometry_msgs::Twist twist_msg;
        // twist_msg.linear.x = v;
        // // std::cout<<"v: "<<v<<std::endl;
        // twist_msg.angular.z = w_cmd;

        // Convert angular velocity to steering angle
        const double wheelbase = 0.5; // 小车的轴距，需要根据实际情况调整
        steering_angle_cmd = atan2(wheelbase * w_cmd, v);

        // Limit steering angle
        const double max_steering_angle = 0.5; // 最大转向角，需要根据实际情况调整
        if (std::fabs(steering_angle_cmd) > max_steering_angle) {
            steering_angle_cmd = steering_angle_cmd > 0 ? max_steering_angle : -max_steering_angle;
        }

        auto target_pose_msg = to_pose_msg(projected);

        ackermann_msgs::AckermannDriveStamped ackermann_msg;
        ackermann_msg.drive.speed = v;
        ackermann_msg.drive.steering_angle = steering_angle_cmd;

        return std::make_tuple(ackermann_msg, target_pose_msg, to_pose_msg(trajectory_[2]));
    }


    MsgsType create_control_nose_based() const
    {
        const auto d_nose = 10.0;
        const auto nominal_v = 5; // scale w according to the ratio v / nominal_v

        //// return early if not enough info
        if( trajectory_.size() == 0 || ! odo_received_ )
        {
            return std::tuple<geometry_msgs::Twist, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>(geometry_msgs::Twist(), geometry_msgs::PoseStamped(), geometry_msgs::PoseStamped());
        }

        // get 0-0
        Pose2D current = {odometry_.x, odometry_.y, odometry_.yaw};

        // get nose
        Pose2D current_nose = {odometry_.x + cos(odometry_.yaw) * d_nose, odometry_.y + sin(odometry_.yaw) * d_nose, odometry_.yaw};

        // project nose on trajectory
        int index = -1;
        int index_nose = -1;
        double mu = -1;

        const auto projected = project_on_trajectory(current, trajectory_, index, mu);
        const auto projected_nose = project_on_trajectory(current_nose, trajectory_, index_nose, mu);

        //ROS_ERROR_STREAM("index:" << index << " size:" << trajectory_.size());
        if(index == -1)
        {
            //ROS_ERROR_STREAM("ego pose doesn't project correctly to planned trajectory");

            //ROS_ERROR_STREAM("current pose:" << current.x << " " << current.y);

//            for(auto p : trajectory_)
//            {
//                ROS_ERROR_STREAM(p.x << " " << p.y);
//            }

            geometry_msgs::Twist twist_msg;
            twist_msg.linear.x = 0;
            twist_msg.angular.z = 0;
            // std::cout<<"index "<<index<<std::endl;
            return std::tuple<geometry_msgs::Twist, geometry_msgs::PoseStamped, geometry_msgs::PoseStamped>
                    (twist_msg, geometry_msgs::PoseStamped(), to_pose_msg(trajectory_[2]));
        }

        /// v
        //double v_index = sqrt(pow(trajectory_[index+1].x - trajectory_[index].x, 2) + pow(trajectory_[index+1].y - trajectory_[index].y, 2)) * steps_per_phase_;
        //double v_index_1 = sqrt(pow(trajectory_[index+2].x - trajectory_[index+1].x, 2) + pow(trajectory_[index+2].y - trajectory_[index+1].y, 2)) * steps_per_phase_;
        //double v = v_index * (1 - mu) + mu * v_index_1;
        int k = index + 1;
        int l = k + 1;
        if(l >= trajectory_.size())
        {
            l = trajectory_.size() - 1;
            k = l - 1;
        }
        double v = sqrt(pow(trajectory_[l].x - trajectory_[k].x, 2) + pow(trajectory_[l].y - trajectory_[k].y, 2)) * steps_per_phase_;
        //double v = sqrt(pow(trajectory_[index+1].x - trajectory_[index].x, 2) + pow(trajectory_[index+1].y - trajectory_[index].y, 2)) * steps_per_phase_;

        /// w
        double w_cmd = 0;
        if(index_nose != -1)
        {
            // distance between corrected nose and current nose
            Eigen::Vector2f u(current_nose.x - projected_nose.x, current_nose.y - projected_nose.y);
            Eigen::Vector2f n(-sin(projected_nose.yaw), cos(projected_nose.yaw));
            auto d = n.dot(u);

            w_cmd = -0.25 * d;
        }

        w_cmd = w_cmd * (v / nominal_v);

        const double max_w = 8;
        if(std::fabs(w_cmd) > max_w)
        {
            //std::cout << "limit omega!" << w_cmd << std::endl;
            //v *= max_w / std::fabs(w_cmd);
            w_cmd = w_cmd > 0 ? max_w : -max_w;
        }

        ///
        auto target_pose_msg = to_pose_msg(projected_nose);

        geometry_msgs::Twist twist_msg;
        twist_msg.linear.x = v;
        twist_msg.angular.z = w_cmd;

        return std::make_tuple(twist_msg, target_pose_msg, to_pose_msg(trajectory_[3]));
    }
OdometryState odometry_;
private:
    // state
    bool odo_received_;
    
    std::vector<Pose2D> trajectory_;

    // params
    const int steps_per_phase_;
};

struct LowPassFilter
{
    geometry_msgs::Twist low_pass_filter_v(const geometry_msgs::Twist & twist, double dt)
    {
        geometry_msgs::Twist filtered = twist;

//        const double dv = twist.linear.x - last_v_;
//        const auto a = dv / dt;

//        // bound acc
//        v_filtered_ = twist.linear.x;

//        if(a > 0 && a > 8.0)
//        {
//            v_filtered_ = last_v_ + 8.0 * dt;
//        }

//        if(a < 0 && a < -8.0)
//        {
//            v_filtered_ = last_v_ - 8.0 * dt;
//        }

//        std::cout << "v filtered:" << v_filtered_ << std::endl;

        v_filtered_ = 0.7 * v_filtered_ + 0.3 * twist.linear.x;

        filtered.linear.x = v_filtered_;

        return filtered;
    }

    double v_filtered_ = 0; // low pass filtering
    double last_v_ = 0;
};



int main(int argc, char **argv)
{
    ROS_INFO_STREAM("Launch trajectory controller..");

    int steps_per_phase = 1;
    int trajectory_index = 0;
    bool low_pass_filter = false;
    bool nose_tracking = false;
    bool ackermann = false;

    LowPassFilter v_filter;

    // ros init
    ros::init(argc, argv, "trajectory_controller");
    ros::NodeHandle n;
    n.getParam("/traj_planner/steps_per_phase", steps_per_phase);
    n.getParam("/traj_controller/trajectory_index", trajectory_index);
    n.getParam("/traj_controller/low_pass_filter", low_pass_filter);
    n.getParam("/traj_controller/nose_tracking", nose_tracking);

    // ros::Publisher ctrl_ackermann=n.advertise<ackermann_msgs::AckermannDrive>("tianracer/ackermann_cmd",10);
    ros::Publisher ctrl_ackermann=n.advertise<ackermann_msgs::AckermannDriveStamped>("tianracer_01/ackermann_cmd_stamped",10);
    ros::Publisher ctrl_publisher = n.advertise<geometry_msgs::Twist>("/lgp_car/vel_cmd", 100);
    ros::Publisher start_planning_publisher = n.advertise<geometry_msgs::PoseStamped>("/lgp_car/start_planning_pose", 100);
    ros::Publisher target_pose_publisher = n.advertise<geometry_msgs::PoseStamped>("/lgp_car/target_pose", 100);
    ros::Publisher odometry_publisher = n.advertise<nav_msgs::Odometry>("/lgp_car/odometry", 1);

    TrajectoryController controller(steps_per_phase);

    boost::function<void(const nav_msgs::Path::ConstPtr& msg)> trajectory_callback =
            boost::bind(&TrajectoryController::trajectory_callback, &controller, _1);

    auto trajectory_subscriber = n.subscribe("/traj_planner/trajectory_" + std::to_string(trajectory_index), 100, trajectory_callback);

    boost::function<void(const nav_msgs::Odometry::ConstPtr& msg)> odometry_callback =
            boost::bind(&TrajectoryController::odometry_callback, &controller, _1);
    auto odo_subscriber = n.subscribe("/lgp_car/odometry", 1, odometry_callback);
    // auto odo_subscriber = n.subscribe("/tianracer_01/odom", 1, odometry_callback);

    ros::Rate loop_rate(20);

    std::shared_ptr<tf::TransformBroadcaster>  tf_broadcaster_ = std::make_shared<tf::TransformBroadcaster>();
    // odometry_publisher.publish(init_odometry_msg());
    loop_rate.sleep();
    ros::spinOnce();
    while (ros::ok())
    {  
        ros::spinOnce();

        geometry_msgs::TransformStamped tf;
        tf.header.stamp = ros::Time::now();
        tf.header.frame_id = "map";
        tf.child_frame_id = "lgp_car";
        tf.transform.translation.x = controller.odometry_.x;
        tf.transform.translation.y = controller.odometry_.y;
        tf.transform.translation.z = 0;
        auto q=get_quaternion_from_yaw(controller.odometry_.yaw);
        tf.transform.rotation.x = q.x;
        tf.transform.rotation.y = q.y;
        tf.transform.rotation.z = q.z;
        tf.transform.rotation.w = q.w;
        tf_broadcaster_->sendTransform(tf);


    if(ackermann)
    {
        MsgsType_2 msgs;
        msgs = controller.create_control_AckermannDrive();      
        auto ctrl = std::get<0>(msgs);
      const auto & target = std::get<1>(msgs); // odo proj onto traj
      const auto & start = std::get<2>(msgs); // planning start
      ctrl_ackermann.publish(ctrl);
            target_pose_publisher.publish(target);
      start_planning_publisher.publish(start);
    }
    else{
      MsgsType msgs;
      if(nose_tracking)
          msgs = controller.create_control_nose_based();
      else
          msgs = controller.create_control();

      auto ctrl = std::get<0>(msgs);
      const auto & target = std::get<1>(msgs); // odo proj onto traj
      const auto & start = std::get<2>(msgs); // planning start
    //   const auto & odo=std::get<3>(msgs);

      if(low_pass_filter)
          ctrl = v_filter.low_pass_filter_v(ctrl, 1.0 / 30);


      ctrl_publisher.publish(ctrl);
      target_pose_publisher.publish(target);
      start_planning_publisher.publish(start);
    }
    //   odometry_publisher.publish(odo);
    // odometry_publisher.publish(init_odometry_msg());

      ros::spinOnce();

      loop_rate.sleep();
    }

    return 0;
}
