#include <nodes/obstacle_common.h>
#include <common/utility.h>

#include <fstream>
#include <math.h>
#include <ros/package.h>
#include <ros/console.h>
#include <math.h>
#include <chrono>

void log_to_file(std::ofstream & ofs, ros::NodeHandle & n, double car_x, int i, double cost)
{
    int n_obstacles;
    int n_non_obstacles;
    double p_obstacle;
    double planning_time = 0;

    n.getParam("/n_total_obstacles", n_obstacles);
    n.getParam("/n_total_non_obstacles", n_non_obstacles);
    n.getParam("/p_obstacle", p_obstacle);
    n.getParam("/planning_time", planning_time);

    std::stringstream ss;

    ss << "N:" << n_obstacles << " M:" << n_non_obstacles << " M+N:"<< n_obstacles + n_non_obstacles << " ~p:" << double(n_obstacles) / (n_non_obstacles+n_obstacles) << " Travelled dist:" << car_x << " Probability:" << p_obstacle << " Avg cost:" << cost << " time(ms):" << planning_time * 1000 << " " << i << " iterations";

    ROS_INFO_STREAM(ss.str());
    //ofs << ss.str() << std::endl;
}

void log_to_file(std::ofstream & ofs, ros::NodeHandle & n, double car_x, double car_y, double car_vx, double car_vy, double acc_x,double acc_y,double jerk_x,double jerk_y,double yaw,double omega,double distance_obs, double distance_obs_long, double distance_obs_lateral)
{
    
    double planning_time = 0;
    n.getParam("/planning_time", planning_time);
    std::stringstream ss;

    ss << car_x<<" "<<car_y<<" "<<yaw<<" "<<car_vx<<" "<<car_vy<<" "<<omega<<" "<<car_vx<<" "<<car_vy<<" "<<acc_x<<" "<<acc_y<<" "<<jerk_x<<" "<<jerk_y<<" "<<distance_obs<<" "<<abs(distance_obs_long)<<" "<<abs(distance_obs_lateral)<<" "<<planning_time * 1000<<" "<<0<<" "<<0;

    //ROS_INFO_STREAM(ss.str());
    ofs << ss.str() << std::endl;
}

void log_to_file(std::ofstream & ofs, ros::NodeHandle & n,int passed_count)
{
    
    double planning_time = 0;
    n.getParam("/planning_time", planning_time);
    std::stringstream ss;

    ss << passed_count<<std::endl;

    //ROS_INFO_STREAM(ss.str());
    ofs << ss.str() << std::endl;
}

void log_to_file(std::ofstream & ofs, ros::NodeHandle & n,double time)
{
    
    double planning_time = 0;
    n.getParam("/planning_time", planning_time);
    std::stringstream ss;

    ss << time<<std::endl;

    //ROS_INFO_STREAM(ss.str());
    ofs << ss.str() << std::endl;
}

std::string filename(ros::NodeHandle & n)
{
    double p_obstacle = 0.1;
    int n_obstacles = 1;
    double certainty_distance_offset = 10.0;
    bool tree = true;

    n.getParam("tree", tree);
    n.getParam("p_obstacle", p_obstacle);
    n.getParam("n_obstacles", n_obstacles);
    n.getParam("certainty_distance_offset", certainty_distance_offset);
    n.getParam("tree", tree);
    n.getParam("tree", tree);

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    //ss << "/home/camille/Phd/Paper/ICRA-2021/plots/gen_obstacles/data-" << std::to_string(p_obstacle) << "-" << (tree ? "tree" : "linear") << "-" << n_obstacles << "-" << "-offset-" << certainty_distance_offset << "-" << t << ".txt";
    ss << "/home/zmz/treempc/src/icra2021/logs/data-"<<t<< ".txt";
    return ss.str();
}

std::string filename_(ros::NodeHandle & n)
{
    double p_obstacle = 0.1;
    int n_obstacles = 1;
    double certainty_distance_offset = 10.0;
    bool tree = true;

    n.getParam("tree", tree);
    n.getParam("p_obstacle", p_obstacle);
    n.getParam("n_obstacles", n_obstacles);
    n.getParam("certainty_distance_offset", certainty_distance_offset);
    n.getParam("tree", tree);
    n.getParam("tree", tree);

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    //ss << "/home/camille/Phd/Paper/ICRA-2021/plots/gen_obstacles/data-" << std::to_string(p_obstacle) << "-" << (tree ? "tree" : "linear") << "-" << n_obstacles << "-" << "-offset-" << certainty_distance_offset << "-" << t << ".txt";
    ss << "/home/zmz/treempc/src/icra2021/logs/data-"<<t<< "passed_obstacles.txt";
    return ss.str();
}

std::string filename__(ros::NodeHandle & n)
{
    double p_obstacle = 0.1;
    int n_obstacles = 1;
    double certainty_distance_offset = 10.0;
    bool tree = true;

    n.getParam("tree", tree);
    n.getParam("p_obstacle", p_obstacle);
    n.getParam("n_obstacles", n_obstacles);
    n.getParam("certainty_distance_offset", certainty_distance_offset);
    n.getParam("tree", tree);
    n.getParam("tree", tree);

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    //ss << "/home/camille/Phd/Paper/ICRA-2021/plots/gen_obstacles/data-" << std::to_string(p_obstacle) << "-" << (tree ? "tree" : "linear") << "-" << n_obstacles << "-" << "-offset-" << certainty_distance_offset << "-" << t << ".txt";
    ss << "/home/zmz/treempc/src/icra2021/logs/data-"<<t<< "timing.txt";
    return ss.str();
}

visualization_msgs::Marker create_obstacle_marker(double x, double y, double sx, double sy, double sz, double alpha, int id, double r, double g, double b,double a)
{
    visualization_msgs::Marker marker;

    marker.header.frame_id = "map";
    marker.id = std::hash<std::string>()("obstacle_" + std::to_string(id));
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = 0.5 * sz;
    marker.scale.x = sx; // diameter
    marker.scale.y = sy;
    marker.scale.z = sz;
    //marker.pose.orientation = get_quaternion_from_yaw(x);
    //marker.pose.orientation = get_quaternion_from_yaw(x);
    marker.color.a = a;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;

    return marker;
}

visualization_msgs::Marker create_collision_marker(double x, double y, double sx, double sy, double sz, double alpha, int id, double lat_dist_to_obstacle)
{
    visualization_msgs::Marker marker;

    const double m = 0.5; // margin
    double b = sy;
    double a = (sx/2)*(sx/2)/(1+1/b);
    //double d = sx * sx / (8 * m) - sy / 2 + m / 2;
    double d = 1.5;
    //double diameter = 2 * (d + sy / 2 + m); //4.55*2
    double diameter=2.2;

    marker.header.frame_id = "map";
    marker.id = std::hash<std::string>()("collision_" + std::to_string(id));
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = x;
    /*if(lat_dist_to_obstacle<0.2)
    {
        marker.pose.position.y = y-d;
    }
    else{
        marker.pose.position.y = y+d;
    }*/
    //marker.pose.position.y = y + (y > 0 ? d : -d);
    marker.pose.position.y = y;
    marker.pose.position.z = 0.5 * 0.01;
    marker.scale.x = diameter; // diameter
    marker.scale.y = diameter;
    //marker.scale.x = 2*a; // diameter
    //marker.scale.y = 2*a;
    marker.scale.z = 0.01;
    marker.color.a = 1.; //(use by optimization)
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    return marker;
}

visualization_msgs::Marker create_ellipsoid_collision_marker(double x, double y, double sx, double sy, double sz, double alpha, int id)
{
    visualization_msgs::Marker marker;

    const double m = 0.5; // margin
    double d = sx * sx / (8 * m) - sy / 2 + m / 2;
    double diameter = 2 * (d + sy / 2 + m);

    marker.header.frame_id = "map";
    marker.id = std::hash<std::string>()("collision_" + std::to_string(id));
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = x;
    marker.pose.position.y = y + (y > 0 ? d : -d);
    marker.pose.position.z = 0.5 * 0.01;
    marker.scale.x = diameter; // diameter
    marker.scale.y = diameter;
    marker.scale.z = 0.01;
    marker.color.a = alpha > 0.01 ? alpha : 0.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

//    //std::cout << "diameter:" << diameter << std::endl;

    return marker;
}

//nav_msgs::Path transform(const nav_msgs::Path & trajectory, double tx)
//{
//    nav_msgs::Path transformed;
//    transformed.header = trajectory.header;

//    for(auto i = 0; i < trajectory.poses.size(); ++i)
//    {
//        const auto pose = trajectory.poses[i];
//        const auto x = pose.pose.position.x;
//        const auto y = pose.pose.position.y;
//        const auto yaw = get_yaw_from_quaternion(pose.pose.orientation);

//        geometry_msgs::PoseStamped transformed_pose;
//        //transformed_pose.header = pose.header;
//        transformed_pose.pose.position.x = x + tx * cos(yaw);
//        transformed_pose.pose.position.y = y + tx * sin(yaw);
//        transformed_pose.pose.orientation = pose.pose.orientation;

//        transformed.poses.push_back(transformed_pose);
//    }

//    return transformed;
//}

//std::vector<nav_msgs::Path> transform(const std::vector<nav_msgs::Path> & trajectories, double tx)
//{
//    std::vector<nav_msgs::Path> transformed;
//    transformed.reserve(trajectories.size());

//    for(const auto & trajectory: trajectories)
//    {
//        transformed.push_back(transform(trajectory, tx));
//    }

//    return transformed;
//}
