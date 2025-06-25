#pragma once

#include <KOMO/komo.h>

#include <eigen3/Eigen/Dense>

uint getFrameIndex(const rai::KinematicWorld& G, const std::string& object_name);

//struct Vector2D
//{
//    double x;
//    double y;
//};

//struct Pose2D
//{
//    double x;
//    double y;
//    double yaw;
//};

//bool near(const Pose2D & a, const Pose2D & b, double eps = 0.0001);
//Pose2D project_on_trajectory(const Pose2D & p, std::vector<Pose2D> trajectory, int & index, double & mu);
//Pose2D project_on_trajectory(const Pose2D & p, std::vector<Pose2D> trajectory);
//std::vector<Pose2D> reinit_trajectory_from(const Pose2D & p, std::vector<Pose2D> trajectory, int prefix_size);
//Pose2D to_2d_pose(rai::KinematicWorld * configuration);
//std::vector< Pose2D > to_2d_trajectory(const KOMO & komo);
//Pose2D to_1d_pose(rai::KinematicWorld * configuration);
//std::vector< Pose2D > to_1d_trajectory(const KOMO & komo);
