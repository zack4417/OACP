/*
 * Contact Information:
 * Lei Zheng
 * Email: zack44170625@gmail.com
 * Affiliation: HKUST/CMU
 */
#define _USE_MATH_DEFINES

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <oacp/Controls.h>
#include <oacp/States.h>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include "oacp/oacp.h"

using std::placeholders::_1;

class MinimalPublisher {
 public:
  MinimalPublisher();

 private:
  void TopicCallback(const oacp::States::ConstPtr& msg);
  void VelocityBoundary(const oacp::States::ConstPtr& msg);
  void TimerCallback(const ros::TimerEvent&);
  void SafetyAdjustment();
  void CandidateEvaluation();
  void Goalfilter();
  bool IsOccluded(int obs_id);

  ros::Subscriber subscription_;
  ros::Publisher publisher_;
  ros::Timer timer_;
  size_t count_;

 
  Eigen::ArrayXXf lane_, tot_time_, vx_g_, x_g_, y_g_, y_delta_, tot_time_up_;
  Eigen::ArrayXXf x_obs_temp_, y_obs_temp_, vx_obs_, vy_obs_, lon_delta_, meta_cost_;
  Eigen::ArrayXXf safety_barriers_, distance_obs_, distance_obs_long_;
  Eigen::ArrayXXf distance_obs_lateral_, v_cost_;
  Eigen::ArrayXXf sigmoid_boundary_, sigmoid_boundary1_;
  optim::four_var PPP_, PPP_up_;
  optim::probData pto_data_;
  float v_des_, x_init_, y_init_, v_x_, v_y_, v_init_, ax_init_, ay_init_;
  float psi_init_, psidot_init_, total_time_, speed_, avg_time_, avg_speed_;
  float gamma_, prev_v_send_, prev_w_send_, w0_, w1_, w2_, w3_, safe_lon_, safe_lat_;
  bool Gotit_, warm_;
  float min_val_ = 1000000;
  float track_steps_ = 2;
  int index_, cnt_, loop_;
  float y_goal_last_;
  float t1_, t2_, y_proj_start_, y_proj_end_, y_min_, y_max_;
  clock_t start_, end_;
  std::ofstream outdata_, outdata2_, outdata3_, outdata4_, data_recode_;
  std::ofstream data_recode_receding_, data_recode_safety_;
  Eigen::Array<bool, Eigen::Dynamic, 1> mask_max_, mask_min_;
};

MinimalPublisher::MinimalPublisher() : count_(0) {
  ros::NodeHandle nh;
  subscription_ = nh.subscribe("ego_vehicle_obs", 10, 
                              &MinimalPublisher::TopicCallback, this);
  publisher_ = nh.advertise<oacp::Controls>("ego_vehicle_cmds", 10);
  timer_ = nh.createTimer(ros::Duration(0.01), 
                         &MinimalPublisher::TimerCallback, this);

  cnt_ = 1;
  ROS_INFO("NODES ARE UP");

  Gotit_ = false;
  warm_ = true;
  YAML::Node map = YAML::LoadFile("src/oacp/config.yaml");

  std::string setting = map["setting"].as<std::string>();
  pto_data_.alpha_admm = 1.0;
  pto_data_.num_goal = map["configuration"][setting]["goal"].as<float>();
  pto_data_.gamma = map["configuration"][setting]["gamma"].as<float>();
  safe_lon_ = map["configuration"][setting]["safe_lon"].as<float>();
  safe_lat_ = map["configuration"][setting]["safe_lat"].as<float>();

  w0_ = map["configuration"][setting]["weights"][0].as<float>();
  w1_ = map["configuration"][setting]["weights"][1].as<float>();
  w2_ = map["configuration"][setting]["weights"][2].as<float>();
  w3_ = map["configuration"][setting]["weights"][3].as<float>();

  x_g_ = Eigen::ArrayXXf(pto_data_.num_goal, 1);
  y_g_ = Eigen::ArrayXXf(pto_data_.num_goal, 1);
  vx_g_ = Eigen::ArrayXXf(pto_data_.num_goal, 1);
  y_delta_ = Eigen::ArrayXXf(pto_data_.num_goal, 1);
  meta_cost_ = Eigen::ArrayXXf(pto_data_.num_goal, 9);
  meta_cost_ = -1;

  for (int i = 0; i < pto_data_.num_goal; ++i) {
    x_g_(i) = map["configuration"][setting]["x_g"][i].as<float>();
    y_g_(i) = map["configuration"][setting]["y_g"][i].as<float>();
    y_delta_(i) = map["configuration"][setting]["y_delta"][i].as<float>();
    meta_cost_(i, 0) = i + 1;
    ROS_INFO_STREAM("y_g_(" << i << "): " << y_g_(i));
    vx_g_(i) = map["configuration"][setting]["v_g"][i].as<float>();
  }
  y_goal_last_ = y_g_(0);

  pto_data_.longitudinal_min = 
      map["configuration"][setting]["pos_limits"][0].as<float>();
  pto_data_.longitudinal_max = 
      map["configuration"][setting]["pos_limits"][1].as<float>();
  pto_data_.lateral_min = 
      map["configuration"][setting]["pos_limits"][2].as<float>();
  pto_data_.lateral_max = 
      map["configuration"][setting]["pos_limits"][3].as<float>();
  avg_speed_ = 0.0;
  speed_ = 0.0;
  total_time_ = 0.0;
  avg_time_ = 0.0;
  index_ = 0;
  loop_ = 0;
  prev_v_send_ = 0.0;
  prev_w_send_ = 0.0;

  x_init_ = 0.0;
  y_init_ = -2;
  v_init_ = 0;
  ax_init_ = 0;
  ay_init_ = 0;
  psi_init_ = 0.0;
  psidot_init_ = 0.0;

  pto_data_.t_fin = 4;
  pto_data_.num = 40;
  pto_data_.t = pto_data_.t_fin / pto_data_.num;
  pto_data_.weight_smoothness = 100;
  pto_data_.weight_smoothness_psi = 150.0;
  pto_data_.weight_vel_tracking = 50;
  pto_data_.weight_lane_tracking = 100;

  pto_data_.maxiter = 200;
  pto_data_.num_obs = 4;
  pto_data_.num_consensus = 
      map["configuration"][setting]["consensus"].as<int>();
  pto_data_.v_max = map["configuration"][setting]["v_max"].as<float>();
  pto_data_.vx_max = pto_data_.v_max;
  pto_data_.vx_min = 0;
  pto_data_.vxc1_max = pto_data_.vxc_max = pto_data_.v_max;
  pto_data_.vxc_min = 0;
  pto_data_.vy_max = 4;
  pto_data_.vy_min = -4;
  pto_data_.ax_max = 3;
  pto_data_.ax_min = -4;
  pto_data_.ay_max = 0.1;
  pto_data_.ay_min = -0.1;
  pto_data_.jx_max = 6.0;
  pto_data_.jy_max = 1.0;
  pto_data_.kappa = 5;
  pto_data_.a_obs_vec = safe_lon_ * 
      Eigen::ArrayXf::LinSpaced(pto_data_.num, 1.0f, 1.0f)
          .replicate(pto_data_.num_obs, pto_data_.num_goal).transpose();
  pto_data_.b_obs_vec = safe_lat_ * 
      Eigen::ArrayXf::LinSpaced(pto_data_.num, 1.0f, 1.0f)
          .replicate(pto_data_.num_obs, pto_data_.num_goal).transpose();

  pto_data_.rho_ineq = 1.0;
  pto_data_.rho_psi = 1.0;
  pto_data_.rho_nonhol = 1.0;
  pto_data_.rho_obs = 1.0;

  tot_time_ = Eigen::ArrayXXf(pto_data_.num, 1);
  tot_time_.col(0).setLinSpaced(pto_data_.num, 0.0, pto_data_.t_fin);

  PPP_ = optim::ComputeBernstein(tot_time_, pto_data_.t_fin, pto_data_.num);
  pto_data_.nvar = PPP_.a.cols();

  tot_time_up_ = Eigen::ArrayXXf(static_cast<int>(pto_data_.t_fin / 0.01), 1);
  tot_time_up_.col(0).setLinSpaced(pto_data_.t_fin / 0.01, 0.0, pto_data_.t_fin);
  PPP_up_ = optim::ComputeBernstein(tot_time_up_, pto_data_.t_fin, 
                                  pto_data_.t_fin / 0.01);
  pto_data_.Pdot_upsample = PPP_up_.b;

  pto_data_.cost_smoothness = pto_data_.weight_smoothness * 
      PPP_.c.transpose().matrix() * PPP_.c.matrix();
  pto_data_.cost_smoothness_psi = pto_data_.weight_smoothness_psi * 
      PPP_.c.transpose().matrix() * PPP_.c.matrix();
  pto_data_.cost_tracking_lateral = pto_data_.weight_lane_tracking * 
      Eigen::MatrixXf::Ones(pto_data_.nvar, pto_data_.nvar);
  pto_data_.cost_tracking_vel = pto_data_.weight_vel_tracking * 
      Eigen::MatrixXf::Ones(pto_data_.nvar, pto_data_.nvar);

  pto_data_.vx_des = vx_g_.transpose().replicate(pto_data_.num, 1);
  pto_data_.y_des = y_g_.transpose().replicate(
      pto_data_.num - pto_data_.num_consensus - track_steps_, 1);

  pto_data_.A_tracking_lateral = PPP_.a.middleRows(
      pto_data_.num_consensus + track_steps_, 
      pto_data_.num - (pto_data_.num_consensus + track_steps_));
  pto_data_.A_tracking_vel = PPP_.b;
  pto_data_.A_eq_x = Eigen::ArrayXXf(2, pto_data_.nvar);
  pto_data_.A_eq_y = Eigen::ArrayXXf(3, pto_data_.nvar);
  pto_data_.A_eq_psi = Eigen::ArrayXXf(4, pto_data_.nvar);
  pto_data_.A_eq_x << PPP_.a.row(0), PPP_.b.row(0);
  pto_data_.A_eq_y << PPP_.a.row(0), PPP_.b.row(0), PPP_.a.row(PPP_.a.rows() - 1);
  pto_data_.A_eq_psi << PPP_.a.row(0), PPP_.b.row(0), 
                      PPP_.a.row(PPP_.a.rows() - 1), PPP_.b.row(PPP_.b.rows() - 1);
  pto_data_.A_nonhol = PPP_.b;
  pto_data_.A_psi = PPP_.a;

  pto_data_.v_ref = Eigen::ArrayXXf::Ones(pto_data_.num, 1) * 15;
  pto_data_.A_consensus_x = optim::stackVertically3(
      PPP_.a.topRows(pto_data_.num_consensus),
      PPP_.b.topRows(pto_data_.num_consensus),
      PPP_.c.topRows(pto_data_.num_consensus));
  pto_data_.A_consensus_y = optim::stackVertically3(
      PPP_.a.topRows(pto_data_.num_consensus),
      PPP_.b.topRows(pto_data_.num_consensus),
      PPP_.c.topRows(pto_data_.num_consensus));
  pto_data_.A_consensus_psi = PPP_.a.topRows(pto_data_.num_consensus);

  pto_data_.A_lateral_long = optim::stack(PPP_.a, -PPP_.a, 'v');
  pto_data_.A_vel = optim::stack(PPP_.b, -PPP_.b, 'v');
  pto_data_.A_acc = optim::stack(PPP_.c, -PPP_.c, 'v');
  pto_data_.A_jerk = optim::stack(PPP_.d, -PPP_.d, 'v');

  pto_data_.b_lateral_ineq = optim::stack(
      pto_data_.lateral_max * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      -pto_data_.lateral_min * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');
  pto_data_.b_longitudinal_ineq = optim::stack(
      pto_data_.longitudinal_max * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      -pto_data_.longitudinal_min * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');
  pto_data_.b_vx_ineq = optim::stack(
      optim::stack(pto_data_.vxc_max * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, 1),
                  -pto_data_.vxc_min * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, 1), 'v'),
      optim::stack(pto_data_.vx_max * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal - 1),
                  -pto_data_.vx_min * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal - 1), 'v'),
      'h');
  pto_data_.b_vy_ineq = optim::stack(
      pto_data_.vy_max * Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      -pto_data_.vy_min * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');
  pto_data_.b_ax_ineq = optim::stack(
      pto_data_.ax_max * Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      -pto_data_.ax_min * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');
  pto_data_.b_ay_ineq = optim::stack(
      pto_data_.ay_max * Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      -pto_data_.ay_min * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');
  pto_data_.b_jx_ineq = optim::stack(
      pto_data_.jx_max * Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      pto_data_.jx_max * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');
  pto_data_.b_jy_ineq = optim::stack(
      pto_data_.jy_max * Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal),
      pto_data_.jy_max * 
          Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal), 'v');

  pto_data_.s_x_ineq_old = pto_data_.s_x_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_y_ineq_old = pto_data_.s_y_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_vx_ineq_old = pto_data_.s_vx_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_vy_ineq_old = pto_data_.s_vy_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_ax_ineq_old = pto_data_.s_ax_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_ay_ineq_old = pto_data_.s_ay_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_jx_ineq_old = pto_data_.s_jx_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;
  pto_data_.s_jy_ineq_old = pto_data_.s_jy_ineq = 
      Eigen::ArrayXXf::Ones(2 * pto_data_.num, pto_data_.num_goal) * 0;

  pto_data_.s_consensus_x = 
      Eigen::ArrayXXf::Ones(pto_data_.A_consensus_x.rows(), pto_data_.num_goal) * 0;
  pto_data_.s_consensus_y = 
      Eigen::ArrayXXf::Ones(pto_data_.A_consensus_y.rows(), pto_data_.num_goal) * 0;
  pto_data_.s_consensus_psi = 
      Eigen::ArrayXXf::Ones(pto_data_.A_consensus_psi.rows(), pto_data_.num_goal) * 0;

  pto_data_.A_obs = optim::stack(PPP_.a, PPP_.a, 'v');
  for (int i = 0; i < pto_data_.num_obs - 2; ++i) {
    pto_data_.A_obs = optim::stack(pto_data_.A_obs, PPP_.a, 'v');
  }
  x_obs_temp_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  y_obs_temp_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  vx_obs_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  vy_obs_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  safety_barriers_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  distance_obs_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  distance_obs_long_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  distance_obs_lateral_ = Eigen::ArrayXXf(pto_data_.num_obs, 1);
  sigmoid_boundary_ = sigmoid_boundary1_ = Eigen::ArrayXXf(pto_data_.num, 1);
  lon_delta_ = x_g_;

  data_recode_.open(map["configuration"][setting]["file"].as<std::string>());
  data_recode_receding_.open(
      map["configuration"][setting]["file_mpc"].as<std::string>());
  data_recode_safety_.open(
      map["configuration"][setting]["file_safety"].as<std::string>());
}

bool MinimalPublisher::IsOccluded(int obs_id) {
  struct Building {
    float x_start, y_start;
    float x_end, y_end;
    float x_lane;
  };
  const Building buildings[2] = {
      {-4.5f, 32.0f, -4.5f, 12.0f, 0.0f},
      {-4.5f, -38.5f, -4.5f, -8.5f, 3.75f}};

  if (x_init_ > -4.5f) {
    return false;
  }

  for (const auto& b : buildings) {
    if (std::abs(x_init_ - b.x_start) < 1e-5f) continue;

    t1_ = (b.x_lane - x_init_) / (b.x_start - x_init_);
    t2_ = (b.x_lane - x_init_) / (b.x_end - x_init_);

    y_proj_start_ = y_init_ + t1_ * (b.y_start - y_init_);
    y_proj_end_ = y_init_ + t2_ * (b.y_end - y_init_);

    y_min_ = std::min(y_proj_start_, y_proj_end_);
    y_max_ = std::max(y_proj_start_, y_proj_end_);

    if (y_obs_temp_(obs_id) >= y_min_ && y_obs_temp_(obs_id) <= y_max_) {
      return true;
    }
  }
  return false;
}

void MinimalPublisher::SafetyAdjustment() {
  safety_barriers_ = ((x_init_ - x_obs_temp_).square() / (2.91f * 2.91f) +
                    (y_init_ - y_obs_temp_).square() / (2.0f * 2.0f) - 1)
                        .matrix();
  for (int i = 0; i < pto_data_.num_obs; ++i) {
    if ((y_init_ > -5 && y_init_ < -2) && 
        (y_obs_temp_(i) > -5 && y_obs_temp_(i) < -2)) {
      safety_barriers_(0) = std::abs(x_init_ - x_obs_temp_(i));
      safety_barriers_(1) = v_x_;
      safety_barriers_(2) = vx_obs_(i);
      break;
    } else {
      safety_barriers_(0) = 0;
    }
  }

  distance_obs_ = sqrt((x_init_ - x_obs_temp_).square() + 
                     (y_init_ - y_obs_temp_).square())
                     .matrix();
  distance_obs_long_ = sqrt((x_init_ - x_obs_temp_).square()).matrix();
  distance_obs_lateral_ = sqrt((y_init_ - y_obs_temp_).square()).matrix();

  for (int obs = 0; obs < pto_data_.num_obs; ++obs) {
    for (int goal = 0; goal < pto_data_.num_goal; ++goal) {
      Eigen::ArrayXf a_values = safe_lon_ * 
          Eigen::ArrayXf::LinSpaced(pto_data_.num, 1.2f, 1.0f) * (1 - 0.05 * obs);
      Eigen::ArrayXf b_values = safe_lat_ * 
          Eigen::ArrayXf::LinSpaced(pto_data_.num, 1.1f, 1.0f) * (1 - 0.03 * obs);

      const int col_index = obs * pto_data_.num;
      const bool is_occluded_obstacle = IsOccluded(obs);
      if ((is_occluded_obstacle || 
          ((goal == 0 && obs >= 3) || (goal == 1 && obs >= 2))) &&
          distance_obs_(obs) > optim::generate_random_threshold(10.0, 0)) {
        a_values = 1e-10 * Eigen::ArrayXf::LinSpaced(pto_data_.num, 1.2f, 1.0f);
        b_values = 1e-10 * Eigen::ArrayXf::LinSpaced(pto_data_.num, 1.1f, 1.0f);
      }
      pto_data_.a_obs_vec.block(goal, col_index, 1, pto_data_.num) = 
          a_values.transpose();
      pto_data_.b_obs_vec.block(goal, col_index, 1, pto_data_.num) = 
          b_values.transpose();
    }
  }
}

void MinimalPublisher::VelocityBoundary(
    const oacp::States::ConstPtr& msg) {
  // ROS_INFO("Received a vel boundary msg");
  pto_data_.vxc_max = msg->vxc_max;
  pto_data_.vxc1_max = msg->vxc1_max;
  pto_data_.vxc_min = msg->vxc_min;
  pto_data_.b_vx_ineq = optim::stack(
      optim::stack(pto_data_.vxc_max * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, 1),
                  -pto_data_.vxc_min * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, 1), 'v'),
      optim::stack(pto_data_.vxc1_max * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal - 1),
                  -pto_data_.vxc_min * 
                  Eigen::ArrayXXf::Ones(pto_data_.num, pto_data_.num_goal - 1), 'v'),
      'h');
}

void MinimalPublisher::TopicCallback(const oacp::States::ConstPtr& msg) {
  x_init_ = msg->x[0];
  y_init_ = msg->y[0];
  v_init_ = std::sqrt(msg->vx[0] * msg->vx[0] + msg->vy[0] * msg->vy[0]);
  psi_init_ = msg->psi[0];
  psidot_init_ = msg->psidot;
  v_x_ = msg->vx[0];
  v_y_ = msg->vy[0];
  x_obs_temp_ << msg->x[1], msg->x[2], msg->x[3], msg->x[4];
  y_obs_temp_ << msg->y[1], msg->y[2], msg->y[3], msg->y[4];
  vx_obs_ << msg->vx[1], msg->vx[2], msg->vx[3], msg->vx[4];
  vy_obs_ << msg->vy[1], msg->vy[2], msg->vy[3], msg->vy[4];
  VelocityBoundary(msg);
  SafetyAdjustment();
  if (loop_ == 0) prev_v_send_ = v_init_;
  Gotit_ = true;
}

void MinimalPublisher::CandidateEvaluation() {
  for (int i = 0; i < pto_data_.num_goal; ++i) {
    for (int j = 1; j < pto_data_.num; ++j) {
      const float weight = (j <= 5) ? 1.0f : std::exp(-(j - 5) / 4.0f);
      meta_cost_(i, 5) += weight * std::abs(pto_data_.v(i, j) - vx_g_(i));
      meta_cost_(i, 6) += weight * std::abs(pto_data_.xdddot(i, j));
      meta_cost_(i, 7) += weight * std::abs(pto_data_.y(i, j) - y_g_(i));
    }
    meta_cost_(i, 8) = pto_data_.res_obs.row(i).matrix().lpNorm<2>();
  }
}

void MinimalPublisher::Goalfilter() {
  Eigen::ArrayXXf numbers(pto_data_.num_obs, pto_data_.num);
  for (int i = 0; i < pto_data_.num_obs; ++i) {
    numbers.row(i).setLinSpaced(pto_data_.num, 0, pto_data_.num);
  }
  pto_data_.x_obs = (Eigen::ArrayXXf::Ones(pto_data_.num_obs, pto_data_.num))
                       .colwise() *
                   x_obs_temp_.col(0) +
                   (numbers.colwise() * vx_obs_.col(0) * pto_data_.t);
  pto_data_.y_obs = (Eigen::ArrayXXf::Ones(pto_data_.num_obs, pto_data_.num))
                       .colwise() *
                   y_obs_temp_.col(0) +
                   (numbers.colwise() * vy_obs_.col(0) * pto_data_.t);

  pto_data_.x_obs_fin = pto_data_.x_obs.col(pto_data_.x_obs.cols() - 1);
  pto_data_.y_obs_fin = pto_data_.y_obs.col(pto_data_.y_obs.cols() - 1);
  x_g_ = lon_delta_ + x_init_;

  y_g_ = y_goal_last_ * Eigen::ArrayXXf::Ones(y_g_.rows(), y_g_.cols()) + y_delta_;
  mask_max_ = y_g_ > pto_data_.lateral_max;
  mask_min_ = y_g_ < pto_data_.lateral_min;
  y_g_ = mask_max_.select(y_goal_last_ - y_delta_, y_g_);
  y_g_ = mask_min_.select(y_goal_last_ - y_delta_, y_g_);
  y_g_ = y_g_.max(pto_data_.lateral_min).min(pto_data_.lateral_max);
  pto_data_.y_des = y_g_.transpose().replicate(
      pto_data_.num - pto_data_.num_consensus - track_steps_, 1);

  for (int i = 0; i < pto_data_.num_goal; ++i) {
    for (int j = 0; j < pto_data_.num_obs; ++j) {
      const float test = optim::safety_dist(x_g_(i, 0), y_g_(i, 0),
                                pto_data_.x_obs_fin(j, 0),
                                pto_data_.y_obs_fin(j, 0));
      if (test < 0) {
        const float test_new = optim::safety_dist(
            x_g_(i, 0) - 5, y_g_(i, 0), pto_data_.x_obs_fin(j, 0),
            pto_data_.y_obs_fin(j, 0));
        x_g_(i, 0) = (test_new > test) ? x_g_(i, 0) - 5 : x_g_(i, 0) - 10;
        break;
      }
    }
  }
}

void MinimalPublisher::TimerCallback(const ros::TimerEvent&) {
  oacp::Controls message;
  if (!Gotit_) return;

  Goalfilter();
  ++cnt_;
  start_ = clock();

  pto_data_ = optim::OACP(pto_data_, PPP_, x_g_, y_g_, x_init_, y_init_, v_init_,
                         ax_init_, ay_init_, psi_init_, psidot_init_, warm_);
  end_ = clock();

  CandidateEvaluation();
  min_val_ = 100000000;
  index_ = 0;

  if (std::abs(psi_init_) > 30 * M_PI / 180 && loop_ > 1) {
    for (int i = 0; i < pto_data_.num_goal; ++i) {
      meta_cost_(i, 7) = std::abs(y_g_(i) - y_goal_last_);
      const float cost = meta_cost_(i, 7);
      if (cost < min_val_) {
        min_val_ = cost;
        index_ = i;
      }
    }
  } else {
    for (int i = 0; i < pto_data_.num_goal; ++i) {
      meta_cost_(i, 1) =
          (meta_cost_(i, 5) - meta_cost_.col(5).minCoeff()) /
          (meta_cost_.col(5).maxCoeff() - meta_cost_.col(5).minCoeff());
      meta_cost_(i, 2) =
          (meta_cost_(i, 6) - meta_cost_.col(6).minCoeff()) /
          (meta_cost_.col(6).maxCoeff() - meta_cost_.col(6).minCoeff());
      meta_cost_(i, 3) =
          (meta_cost_(i, 7) - meta_cost_.col(7).minCoeff()) /
          (meta_cost_.col(7).maxCoeff() - meta_cost_.col(7).minCoeff());
      meta_cost_(i, 5) = std::abs(y_g_(i) - y_goal_last_) / 8;
      meta_cost_(i, 4) =
          (meta_cost_(i, 8) - meta_cost_.col(8).minCoeff()) /
          (meta_cost_.col(8).maxCoeff() - meta_cost_.col(8).minCoeff());
      const float cost = w0_ * meta_cost_(i, 1) + w1_ * meta_cost_(i, 2) +
                         w2_ * meta_cost_(i, 3) + w3_ * meta_cost_(i, 4) +
                         0 * meta_cost_(i, 5);
      if (cost < min_val_) {
        min_val_ = cost;
        index_ = i;
      }
    }
  }

  y_goal_last_ = y_g_(index_);
  message.w = pto_data_.psidot(index_, 1);
  message.v = pto_data_.v(index_, 1);
  if (loop_ <= 2) {
    message.v = v_init_;
    message.w = 0;
  }
  ax_init_ = pto_data_.xddot(index_, 1);
  ay_init_ = pto_data_.yddot(index_, 1);
  message.jx = pto_data_.xdddot(index_, 1);
  message.jy = pto_data_.ydddot(index_, 1);
  message.index = index_;
  message.goals = pto_data_.num_goal;
  ++loop_;

  speed_ += message.v;
  total_time_ += static_cast<double>(end_ - start_) / CLOCKS_PER_SEC;
  avg_speed_ = speed_ / loop_;
  avg_time_ = total_time_ / loop_;

  data_recode_ << x_init_ << " " << y_init_ << " " << psi_init_ << " " 
               << message.v << " " << message.v << " " << message.w << " "
               << pto_data_.xdot(index_, 1) << " " << pto_data_.ydot(index_, 1) << " "
               << pto_data_.xddot(index_, 1) << " " << pto_data_.yddot(index_, 1) << " "
               << pto_data_.xdddot(index_, 1) << " " << pto_data_.ydddot(index_, 1) << " "
               << distance_obs_(0) << " " << distance_obs_long_(0) << " " 
               << distance_obs_lateral_(0) << " "
               << static_cast<double>(end_ - start_) / CLOCKS_PER_SEC << " " 
               << loop_ << " " << index_ << "\t" << y_goal_last_ << std::endl;

  data_recode_safety_ << safety_barriers_(0) << " " << safety_barriers_(1) << " "
                     << safety_barriers_(2) << " " << loop_ + 1 << " " << 0 
                     << std::endl;

  for (int j = 0; j < pto_data_.num; ++j) {
    data_recode_receding_ << index_ << 0 << "\t" << pto_data_.x(0, j) << "\t"
                         << pto_data_.y(0, j) << "\t" << pto_data_.v(0, j) << "\t"
                         << pto_data_.xddot(0, j) << "\t" << pto_data_.ydot(0, j)
                         << "\t" << pto_data_.xdot(0, j) << "\t"
                         << pto_data_.yddot(0, j) << std::endl;
  }
  for (int j = 0; j < pto_data_.num; ++j) {
    data_recode_receding_ << index_ << 0 << "\t" << pto_data_.x(1, j) << "\t"
                         << pto_data_.y(1, j) << "\t" << pto_data_.v(1, j) << "\t"
                         << pto_data_.xddot(1, j) << "\t" << pto_data_.ydot(1, j)
                         << "\t" << pto_data_.xdot(1, j) << "\t"
                         << pto_data_.yddot(1, j) << std::endl;
  }

  prev_v_send_ = message.v;
  prev_w_send_ = message.w;

  for (int i = 0; i < pto_data_.num_goal; ++i) {
    for (int j = 0; j < pto_data_.num; ++j) {
      geometry_msgs::Pose pose;
      pose.position.x = pto_data_.x(i, j);
      pose.position.y = pto_data_.y(i, j);
      message.batch.poses.push_back(pose);
    }
  }

  ROS_INFO("Average time = %f Average speed = %f", avg_time_, avg_speed_);
  publisher_.publish(message);
  Gotit_ = false;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "minimal_publisher");
  MinimalPublisher minimal_publisher;
  ros::spin();
  return 0;
}