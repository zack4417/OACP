#pragma once

#include <visualization_msgs/Marker.h>
#include <ros/node_handle.h>

void log_to_file(std::ofstream & ofs, ros::NodeHandle & n, double car_x, int i, double cost);

std::string filename(ros::NodeHandle & n);
std::string filename_(ros::NodeHandle & n);
std::string filename__(ros::NodeHandle & n);
visualization_msgs::Marker create_obstacle_marker(double x, double y, double sx, double sy, double sz, double alpha, int id, double r, double g, double b,double a);
visualization_msgs::Marker create_collision_marker(double x, double y, double sx, double sy, double sz, double alpha, int id,double lat_dist_to_obstacle);
void log_to_file(std::ofstream & ofs, ros::NodeHandle & n, double car_x, double car_y, double car_vx, double car_vy, double acc_x,double acc_y,double jerk_x,double jerk_y,double yaw,double omega,double distance_obs, double distance_obs_long, double distance_obs_lateral);
void log_to_file(std::ofstream & ofs, ros::NodeHandle & n, int passed_count);
void log_to_file(std::ofstream & ofs, ros::NodeHandle & n, double time);
visualization_msgs::Marker create_ellipsoid_collision_marker(double x, double y, double sx, double sy, double sz, double alpha, int id);
//visualization_msgs::Marker create_parked_car_collision_marker(double x, double y, double sx, double sy, double sz, double alpha, int id);
//std::vector<nav_msgs::Path> transform(const std::vector<nav_msgs::Path> & trajectories, double tx);
