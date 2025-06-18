/*
 * Contact Information:
 * Lei Zheng
 * Email: zack44170625@gmail.com
 * Affiliation: HKUST/CMU
 */
#define _USE_MATH_DEFINES
#include <memory>
#include <iostream>  
#include <ctime>
#include <fstream>
#include <cstdlib>
#include <iomanip> 
#include <cmath>
#include "yaml-cpp/yaml.h"
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include "consensus_bphto/consensus_bphto.h" 
#include <consensus_bphto/Controls.h>
#include <consensus_bphto/States.h>
using namespace std;
using namespace optim;
using std::placeholders::_1; 

class MinimalPublisher {
  public: 
        ArrayXXf lane, tot_time, vx_g, x_g, y_g, y_delta, tot_time_up;
        ArrayXXf x_obs_temp, y_obs_temp, vx_obs, vy_obs, lon_delta, meta_cost,safety_barriers, distance_obs, distance_obs_long, distance_obs_lateral;
        ArrayXXf v_cost;
        ArrayXXf sigmoid_boundary, sigmoid_boundary1;
        four_var PPP, PPP_up;
        probData pto_data;
        float v_des, x_init, y_init, v_x, v_y, v_init, ax_init, ay_init, psi_init, psidot_init, total_time, speed, avg_time, avg_speed;
        float gamma, prev_v_send, prev_w_send, w0, w1, w2, w3, safe_lon, safe_lat;
        bool Gotit, warm;
        float min = 1000000;
        float track_steps = 2; //OCC, in a same lane, this can be small like 5
        int index, cnt, loop;
        float  y_goal_last;
        float t1, t2 , y_proj_start, y_proj_end, y_min, y_max;
        clock_t start, end;
        ofstream outdata, outdata2, outdata3, outdata4, data_recode, data_recode_receding, data_recode_safety;
        Eigen::Array<bool, Eigen::Dynamic, 1> mask_max, mask_min; 
        MinimalPublisher(); 
  private:
    void TopicCallback(const consensus_bphto::States::ConstPtr& msg);
    void VelocityBoundary(const consensus_bphto::States::ConstPtr& msg);
    void TimerCallback(const ros::TimerEvent&);
    void SafetyAdjustment();
    void CandidateEvaluation();
    void Goalfilter();
    void GoalPoint();
    void GetRanks();
    bool IsOccluded(int obs_id);
    ros::Subscriber subscription_;
    // ros::Subscriber subscription_vx_bound_;
    ros::Publisher publisher_;
    ros::Timer timer_;
    size_t count_;
}; 

// Implementation of constructor
MinimalPublisher::MinimalPublisher(): count_(0) { 
        ros::NodeHandle nh;
        subscription_ = nh.subscribe("ego_vehicle_obs", 10, &MinimalPublisher::TopicCallback, this);
        // subscription_vx_bound_ = nh.subscribe("ego_vehicle_vel_bound", 10, &MinimalPublisher::TopicVelCallback, this);
        publisher_ = nh.advertise<consensus_bphto::Controls>("ego_vehicle_cmds", 10);
        timer_ = nh.createTimer(ros::Duration(0.01), &MinimalPublisher::TimerCallback, this);
 
        cnt = 1;
        ROS_INFO("NODES ARE UP");
 
        Gotit = false;
        warm = true;
        YAML::Node map = YAML::LoadFile("src/consensus_bphto/config.yaml");

        // YAML::Node map1 = YAML::LoadFile("src/consensus_bphto/config.yaml"); 
        string setting = map["setting"].as<string>();
        pto_data.alpha_admm = 1.0; //  1.5
        pto_data.num_goal = map["configuration"][setting]["goal"].as<float>();
        pto_data.gamma = map["configuration"][setting]["gamma"].as<float>();
        safe_lon = map["configuration"][setting]["safe_lon"].as<float>();
        safe_lat = map["configuration"][setting]["safe_lat"].as<float>();
        // cout << "gamma num_goal yaml" <<  " " << pto_data.num_goal << " " << endl;
        // cout << "gamma value yaml" <<  " " << pto_data.gamma << " " << endl;

        w0 = map["configuration"][setting]["weights"][0].as<float>();
        w1 = map["configuration"][setting]["weights"][1].as<float>();
        w2 = map["configuration"][setting]["weights"][2].as<float>();
        w3 = map["configuration"][setting]["weights"][3].as<float>();        
 
        x_g = ArrayXXf(pto_data.num_goal, 1);
        y_g = ArrayXXf(pto_data.num_goal, 1);
        vx_g = ArrayXXf(pto_data.num_goal, 1);
        y_delta = ArrayXXf(pto_data.num_goal, 1);

        meta_cost = ArrayXXf(pto_data.num_goal, 9);
        meta_cost = -1;

		for(int i = 0; i < pto_data.num_goal; ++i) {
            x_g(i) = map["configuration"][setting]["x_g"][i].as<float>();
            y_g(i) = map["configuration"][setting]["y_g"][i].as<float>();
            y_delta(i) = map["configuration"][setting]["y_delta"][i].as<float>();
            meta_cost(i, 0) = i+1;
            cout << "y_g(i)" <<  " " << y_g(i) << " " << endl;
            vx_g(i) = map["configuration"][setting]["v_g"][i].as<float>();
        }
        y_goal_last = y_g(0);
        // cout << "y_goal_last_init" <<  " " << y_goal_last << " " << endl;

        pto_data.longitudinal_min = map["configuration"][setting]["pos_limits"][0].as<float>();
        pto_data.longitudinal_max = map["configuration"][setting]["pos_limits"][1].as<float>();
        pto_data.lateral_min = map["configuration"][setting]["pos_limits"][2].as<float>();
        pto_data.lateral_max = map["configuration"][setting]["pos_limits"][3].as<float>();        
        avg_speed = 0.0;
        speed = 0.0;
        total_time = 0.0;
        avg_time = 0.0;
        index = 0;
        loop = 0;
        prev_v_send = 0.0;
        prev_w_send = 0.0;

        x_init = 0.0;
        y_init = -2;
        v_init = 0;
        ax_init = 0;        
        ay_init = 0;
        psi_init = 0.0;
        psidot_init = 0.0;

        pto_data.t_fin = 4; 
        pto_data.num = 40; 
        pto_data.t = pto_data.t_fin/pto_data.num; 
        pto_data.weight_smoothness = 100; // 
        pto_data.weight_smoothness_psi = 150.0; // 
        pto_data.weight_vel_tracking =  50; // lon vel tracking 150
        pto_data.weight_lane_tracking =  100; // lateral position tracking

        pto_data.maxiter = 200;
        pto_data.num_obs = 4;
        pto_data.num_consensus = map["configuration"][setting]["consensus"].as<int>();
        pto_data.v_max = map["configuration"][setting]["v_max"].as<float>(); 
        pto_data.vx_max = pto_data.v_max;        
        pto_data.vx_min = 0;
        pto_data.vxc1_max = pto_data.vxc_max = pto_data.v_max;        
        pto_data.vxc_min = 0;
        pto_data.vy_max = 4;        
        pto_data.vy_min = -4;
        pto_data.ax_max = 3;
        pto_data.ax_min = -4;
        pto_data.ay_max = 0.1;
        pto_data.ay_min = -0.1; 
        pto_data.jx_max = 6.0; //racing static
        pto_data.jy_max = 1.0;
        pto_data.kappa = 5;
        pto_data.a_obs_vec = safe_lon * ArrayXf::LinSpaced(pto_data.num, 1.0f, 1.0f).replicate(pto_data.num_obs,  pto_data.num_goal).transpose();
     	pto_data.b_obs_vec = safe_lat * ArrayXf::LinSpaced(pto_data.num, 1.0f, 1.0f).replicate(pto_data.num_obs, pto_data.num_goal).transpose();
          
        //size (num_goal , pto_data.num_obs*pto_data.num 3 200
        pto_data.rho_ineq = 1.0;
        pto_data.rho_psi = 1.0;
        pto_data.rho_nonhol = 1.0;
        pto_data.rho_obs = 1.0;  

        tot_time = ArrayXXf(pto_data.num, 1);
        tot_time.col(0).setLinSpaced(pto_data.num, 0.0, pto_data.t_fin);

        PPP = ComputeBernstein(tot_time, pto_data.t_fin, pto_data.num);
        pto_data.nvar = PPP.a.cols();
        // pto_data.nvar  = 11
        //__________________________________________________________________________
        tot_time_up = ArrayXXf((int)(pto_data.t_fin/0.01), 1);
        tot_time_up.col(0).setLinSpaced(pto_data.t_fin/0.01, 0.0, pto_data.t_fin);
        
        PPP_up = ComputeBernstein(tot_time_up, pto_data.t_fin, pto_data.t_fin/0.01);
        pto_data.Pdot_upsample = PPP_up.b;
        //__________________________________________________________________________
         
        // pto_data.cost_smoothness =   0 * v_cost.transpose().matrix() * v_cost.matrix() + pto_data.weight_smoothness * PPP.c.transpose().matrix() * PPP.c.matrix();
        pto_data.cost_smoothness =  pto_data.weight_smoothness * PPP.c.transpose().matrix() * PPP.c.matrix();
        pto_data.cost_smoothness_psi = pto_data.weight_smoothness_psi * PPP.c.transpose().matrix() * PPP.c.matrix();
        pto_data.cost_tracking_lateral = pto_data.weight_lane_tracking * ones(pto_data.nvar, pto_data.nvar);
        pto_data.cost_tracking_vel = pto_data.weight_vel_tracking * ones(pto_data.nvar, pto_data.nvar);
       
        pto_data.vx_des = vx_g.transpose().replicate(pto_data.num, 1);
        pto_data.y_des = y_g.transpose().replicate(pto_data.num - pto_data.num_consensus - track_steps, 1);
        // pto_data.y_des = y_g.transpose().replicate(5, 1);
        pto_data.A_tracking_lateral = PPP.a.middleRows(pto_data.num_consensus + track_steps, pto_data.num - (pto_data.num_consensus + track_steps));
        pto_data.A_tracking_vel = PPP.b;
        pto_data.A_eq_x = ArrayXXf(2, pto_data.nvar);
        pto_data.A_eq_y = ArrayXXf(3, pto_data.nvar);
        pto_data.A_eq_psi = ArrayXXf(4, pto_data.nvar);
        pto_data.A_eq_x << PPP.a.row(0), PPP.b.row(0);// xy_init, vxy_init
        pto_data.A_eq_y << PPP.a.row(0), PPP.b.row(0), PPP.a.row(PPP.a.rows() - 1);// xy_init, vxy_init,  xy_fin (PPP.a.rows() = pto_data.num)
        pto_data.A_eq_psi << PPP.a.row(0), PPP.b.row(0), PPP.a.row(PPP.a.rows() - 1), PPP.b.row(PPP.b.rows() - 1); // psi init  psi_dot init psi fin  psi_dot fin 
        pto_data.A_nonhol = PPP.b;
        pto_data.A_psi = PPP.a;

		pto_data.v_ref = ArrayXXf :: Ones(pto_data.num, 1) * 15;
        // @ consensus Constraints  		
        pto_data.A_consensus_x = stackVertically3(PPP.a.topRows(pto_data.num_consensus), PPP.b.topRows(pto_data.num_consensus), PPP.c.topRows(pto_data.num_consensus));
        pto_data.A_consensus_y = stackVertically3(PPP.a.topRows(pto_data.num_consensus), PPP.b.topRows(pto_data.num_consensus), PPP.c.topRows(pto_data.num_consensus));
        // pto_data.A_consensus_y = PPP.a.topRows(pto_data.num_consensus);        
        // pto_data.A_consensus_x = stack(PPP.a.topRows(pto_data.num_consensus), PPP.b.topRows(pto_data.num_consensus), 'v');
        // pto_data.A_consensus_y = stack(PPP.a.topRows(pto_data.num_consensus), PPP.b.topRows(pto_data.num_consensus), 'v');
        pto_data.A_consensus_psi = PPP.a.topRows(pto_data.num_consensus);
        // 
 
        // @ inequality Constraints  
        pto_data.A_lateral_long = stack(PPP.a, -PPP.a, 'v');
        pto_data.A_vel = stack(PPP.b, -PPP.b, 'v');	
        pto_data.A_acc = stack(PPP.c, -PPP.c, 'v');	
		pto_data.A_jerk = stack(PPP.d, -PPP.d, 'v');	
        // shape A_psi 50	11
        // shape A acc 100	11
        // shape A_lateral_long 100	11 

        pto_data.b_lateral_ineq = stack(pto_data.lateral_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.lateral_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
		pto_data.b_longitudinal_ineq = stack(pto_data.longitudinal_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.longitudinal_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
 		// pto_data.b_vx_ineq = stack(pto_data.vx_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.vx_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
        // Create lon. boundaries for all trajectories
        pto_data.b_vx_ineq = stack(
            stack(pto_data.vxc_max * ArrayXXf::Ones(pto_data.num, 1),
                -pto_data.vxc_min * ArrayXXf::Ones(pto_data.num, 1),'v' ),
            stack(pto_data.vx_max * ArrayXXf::Ones(pto_data.num, pto_data.num_goal - 1),
                -pto_data.vx_min * ArrayXXf::Ones(pto_data.num, pto_data.num_goal - 1), 'v' ),
            'h' );
        pto_data.b_vy_ineq = stack(pto_data.vy_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.vy_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
	    pto_data.b_ax_ineq = stack(pto_data.ax_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.ax_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
		pto_data.b_ay_ineq = stack(pto_data.ay_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.ay_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
		pto_data.b_jx_ineq = stack(pto_data.jx_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  pto_data.jx_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
		pto_data.b_jy_ineq = stack(pto_data.jy_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  pto_data.jy_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');

        pto_data.s_x_ineq_old = pto_data.s_x_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;
		pto_data.s_y_ineq_old = pto_data.s_y_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;
 		pto_data.s_vx_ineq_old = pto_data.s_vx_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;
		pto_data.s_vy_ineq_old = pto_data.s_vy_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;		
	    pto_data.s_ax_ineq_old = pto_data.s_ax_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;
		pto_data.s_ay_ineq_old = pto_data.s_ay_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;		
        pto_data.s_jx_ineq_old = pto_data.s_jx_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;
		pto_data.s_jy_ineq_old = pto_data.s_jy_ineq = ArrayXXf :: Ones(2*pto_data.num, pto_data.num_goal) *0;		
 
        pto_data.s_consensus_x = ArrayXXf :: Ones(pto_data.A_consensus_x.rows(), pto_data.num_goal) *0;
        pto_data.s_consensus_y = ArrayXXf :: Ones(pto_data.A_consensus_y.rows() , pto_data.num_goal) *0;
        pto_data.s_consensus_psi = ArrayXXf :: Ones(pto_data.A_consensus_psi.rows() , pto_data.num_goal) *0;
		// cout << "shape pto_data.s_consensus_psi" << pto_data.s_consensus_psi.rows() << "\t" << pto_data.s_consensus_psi.cols() << endl;	
        // pto_data.A_acc = PPP.c;
        
        pto_data.A_obs = stack(PPP.a, PPP.a, 'v');
        for (int i = 0; i < pto_data.num_obs - 2; ++i)
            pto_data.A_obs = stack(pto_data.A_obs, PPP.a, 'v'); 
        x_obs_temp = ArrayXXf(pto_data.num_obs, 1);
        y_obs_temp = ArrayXXf(pto_data.num_obs, 1);
        vx_obs = ArrayXXf(pto_data.num_obs, 1);
        vy_obs = ArrayXXf(pto_data.num_obs, 1);
        safety_barriers = ArrayXXf(pto_data.num_obs, 1); 
        distance_obs = ArrayXXf(pto_data.num_obs, 1);
        distance_obs_long = ArrayXXf(pto_data.num_obs, 1);
        distance_obs_lateral = ArrayXXf(pto_data.num_obs, 1); 
        sigmoid_boundary = sigmoid_boundary1 = ArrayXXf(pto_data.num, 1);
        lon_delta = x_g;  
        data_recode.open(map["configuration"][setting]["file"].as<string>());
        data_recode_receding.open(map["configuration"][setting]["file_mpc"].as<string>());
        data_recode_safety.open(map["configuration"][setting]["file_safety"].as<string>());
     } 

 
bool MinimalPublisher::IsOccluded(int obs_id) { 
    // Define buildings (start/end points)
    const struct Building {
        float x_start, y_start;
        float x_end, y_end;
        float x_lane;
    } buildings[2] = {
        // Left building
        { -4.5f, 32.0f, -4.5f, 12.0f, 0.0f },
        // Right building (corrected coordinates)
        { -4.5f, -38.5f, -4.5f, -8.5f, 3.75f }
    };
    if (x_init > -4.5f) {
        // If ego is outside the building range, no occlusion
        return false;
    }
    for (const auto& b : buildings) {
        // Skip if ego is aligned with building (degenerate case)
        if (std::abs(x_init - b.x_start) < 1e-5f) continue;

        // Calculate projection parameters
        t1 = (b.x_lane - x_init) / (b.x_start - x_init);
        t2 = (b.x_lane - x_init) / (b.x_end - x_init);
        
        // Calculate projected y-coordinates on lane
        y_proj_start = y_init + t1 * (b.y_start - y_init);
        y_proj_end = y_init + t2 * (b.y_end - y_init);
        
        // Determine min/max y-range of occluded area
        y_min = std::min(y_proj_start, y_proj_end);
        y_max = std::max(y_proj_start, y_proj_end);
        // cout << "y_obs_temp(obs_id)" << y_obs_temp(obs_id) << "y_min" << y_min << "y_max" << y_max << endl;
        //  Check if obstacle is in occluded region
        if (y_obs_temp(obs_id) >= y_min && y_obs_temp(obs_id)  <= y_max) {
            return true;
        }
    }
    return false;
}


void MinimalPublisher:: SafetyAdjustment() { 
	safety_barriers = ((x_init - x_obs_temp).square() / (2.91f * 2.91f) + (y_init - y_obs_temp).square() / (2.0f * 2.0f) - 1).matrix();
    for (int i = 0; i <  pto_data.num_obs; ++i){
        if( (y_init > -5 && y_init < -2) && (y_obs_temp(i) > -5 && y_obs_temp(i) < -2)) {
            safety_barriers(0) = abs(x_init - x_obs_temp(i));
        	safety_barriers(1) = v_x;
	        safety_barriers(2) = vx_obs(i);
        	// safety_barriers(3) = x_obs_temp(i);
	        // safety_barriers(4) = y_obs_temp(i);
            break;
        } else{
            safety_barriers(0) = 0;
        	// safety_barriers(1) = 0;
	        // safety_barriers(2) = 0;
        }
    }
    // safety_barriers(0) = abs(x_init - x_obs_temp(0));
	// safety_barriers(1) = abs(y_init - y_obs_temp(0));
	// safety_barriers(2) = abs(x_init - x_obs_temp(1));
	// safety_barriers(3) =  v_x;
	// safety_barriers(4) = vx_obs(0);
    distance_obs =  sqrt((x_init - x_obs_temp).square()  + (y_init - y_obs_temp).square()).matrix();
    distance_obs_long =  sqrt((x_init - x_obs_temp).square()).matrix();
    distance_obs_lateral = sqrt((y_init - y_obs_temp).square()).matrix(); 
    for (int obs = 0; obs < pto_data.num_obs; ++obs) {
        for (int goal = 0; goal < pto_data.num_goal; ++goal) { 
            ArrayXf a_values = safe_lon * ArrayXf::LinSpaced(pto_data.num, 1.2f, 1.0f) * (1-0.05*obs); // pto_data.num * 1
            ArrayXf b_values = safe_lat * ArrayXf::LinSpaced(pto_data.num, 1.1f, 1.0f)* (1-0.03*obs); 
            // contingency trajectory consider all possible obstacles
            int col_index = obs * pto_data.num;  
            //  occlusion handling condition
            bool is_occluded_obstacle = IsOccluded(obs);  // Your occlusion detection 
            if (is_occluded_obstacle ||((goal == 0 && obs >= 3) || (goal == 1 && obs >= 2) ) && distance_obs(obs) > generate_random_threshold(10.0, 0)) { 
                // ignore obstacles 
                // cout << "goal" << goal << " " << "obs"  << obs << " "  << "distance_obs" << distance_obs(obs) << endl;
                a_values = 1e-10 * ArrayXf::LinSpaced(pto_data.num, 1.2f, 1.0f); // pto_data.num * 1
                b_values = 1e-10 * ArrayXf::LinSpaced(pto_data.num, 1.1f, 1.0f);
            }
            // Assign the generated values to the corresponding block in the final array
            pto_data.a_obs_vec.block(goal, col_index, 1, pto_data.num) = a_values.transpose();
            pto_data.b_obs_vec.block(goal, col_index, 1, pto_data.num) = b_values.transpose();
        }
    }  
    // //size (num_goal , pto_data.num_obs*pto_data.num)  
}


 
void MinimalPublisher::VelocityBoundary(const consensus_bphto::States::ConstPtr& msg) { 
    ROS_INFO("Received a vel boundary msg");  
    pto_data.vxc_max  = msg->vxc_max;
    pto_data.vxc1_max  = msg->vxc1_max;
    pto_data.vxc_min = msg->vxc_min;
    // cout << "vxc_max "<< "\t" << pto_data.vxc_max  << "\t"<< pto_data.vxc1_max << "msg->x[0]" << msg->x[0]<< endl; 
    // vx_g(0) = pto_data.vxc_max * 0.8; 
    // vx_g(1) = pto_data.vxc_max; 
    // vx_g(0) = clip3(pto_data.vxc_min,pto_data.vxc_max, vx_g(0));
    // vx_g(1) = clip3(pto_data.vxc_min,pto_data.vxc_max, vx_g(1));
    // pto_data.vx_des = vx_g.transpose().replicate(pto_data.num, 1);
    // // Update the first trajectory boundary
    // pto_data.vx_des = vx_g.transpose().replicate(pto_data.num, 1);
    // pto_data.b_vx_ineq.block(0, 0, pto_data.num * 2, 1) = stack(
    // pto_data.vxc_max * ArrayXXf::Ones(pto_data.num, 1),
    // -pto_data.vxc_min * ArrayXXf::Ones(pto_data.num, 1),
    // 'v');
    // Update all trajectory boundary 

    // // Parameters for the exponential growth function
    // float growth_rate = 0.8; // Speed of transition
    // float initial_offset = 0.1; // Initial proportion of the transition

    // // Precompute scale factors for efficiency
    // float scale = (1 - initial_offset) * (pto_data.vxc_max - v_init);
    // float scale1 = (1 - initial_offset) * (pto_data.vxc1_max - v_init);

    // // Generate the exponential-based velocity boundary
    // for (int t = 0; t < pto_data.num; ++t) {
    //     float exp_value = 1.0 - exp(-growth_rate * t); // Exponential growth
    //     sigmoid_boundary(t, 0) = v_init + initial_offset * (pto_data.vxc_max - v_init) + exp_value * scale;
    //     sigmoid_boundary1(t, 0) = v_init + initial_offset * (pto_data.vxc1_max - v_init) + exp_value * scale1;
    // }

    // // Debug print for the boundary values
    // // cout << "sigmoid_boundary" << sigmoid_boundary << endl;

    // // Integrate the new boundary into the existing constraints
    // pto_data.b_vx_ineq = stack(
    //     stack(sigmoid_boundary,
    //         -pto_data.vxc_min * ArrayXXf::Ones(pto_data.num, 1), 'v'),
    //     stack(sigmoid_boundary1,
    //         -pto_data.vxc_min * ArrayXXf::Ones(pto_data.num, pto_data.num_goal - 1), 'v'),
    //     'h');
 

    pto_data.b_vx_ineq = stack(
        stack(pto_data.vxc_max * ArrayXXf::Ones(pto_data.num, 1),
            -pto_data.vxc_min * ArrayXXf::Ones(pto_data.num, 1),'v' ),
        stack(pto_data.vxc1_max * ArrayXXf::Ones(pto_data.num, pto_data.num_goal - 1),
            -pto_data.vxc_min * ArrayXXf::Ones(pto_data.num, pto_data.num_goal - 1), 'v' ),
        'h' );
}

void MinimalPublisher::TopicCallback(const consensus_bphto::States::ConstPtr& msg) { 
    // ROS_INFO("Received a message");  
    x_init = msg->x[0];
    y_init = msg->y[0];
    v_init = sqrt(msg->vx[0] * msg->vx[0] + msg->vy[0] * msg->vy[0]);
    // if(loop <= 1) {
    //     v_init = v_des;
    // } 
    psi_init = msg->psi[0];
    psidot_init = msg->psidot; 
    v_x = msg->vx[0];
    v_y = msg->vy[0];
    // x_obs_temp << msg->x[1], msg->x[2], msg->x[3], msg->x[4], msg->x[5], msg->x[6];
    // y_obs_temp << msg->y[1], msg->y[2], msg->y[3], msg->y[4], msg->y[5], msg->y[6];
    // vx_obs << msg->vx[1], msg->vx[2], msg->vx[3], msg->vx[4], msg->vx[5], msg->vx[6];
    // vy_obs << msg->vy[1], msg->vy[2], msg->vy[3], msg->vy[4], msg->vy[5], msg->vy[6];
    // x_obs_temp << msg->x[1], msg->x[2], msg->x[3], msg->x[4], msg->x[5];
    // y_obs_temp << msg->y[1], msg->y[2], msg->y[3], msg->y[4], msg->y[5];
    // vx_obs << msg->vx[1], msg->vx[2], msg->vx[3], msg->vx[4], msg->vx[5];
    // vy_obs << msg->vy[1], msg->vy[2], msg->vy[3], msg->vy[4], msg->vy[5]; 
    x_obs_temp << msg->x[1], msg->x[2], msg->x[3], msg->x[4];
    y_obs_temp << msg->y[1], msg->y[2], msg->y[3], msg->y[4];
    vx_obs << msg->vx[1], msg->vx[2], msg->vx[3], msg->vx[4];
    vy_obs << msg->vy[1], msg->vy[2], msg->vy[3], msg->vy[4]; 
    // x_obs_temp << msg->x[1], msg->x[2], msg->x[3];
    // y_obs_temp << msg->y[1], msg->y[2], msg->y[3];
    // vx_obs << msg->vx[1], msg->vx[2], msg->vx[3];
    // vy_obs << msg->vy[1], msg->vy[2], msg->vy[3]; 
    VelocityBoundary(msg);
    SafetyAdjustment(); 
    
    if(loop == 0)
        prev_v_send = v_init; 
    Gotit = true; 
}

void MinimalPublisher :: GetRanks() {
    // // float v_cruise = 15.0;
    // for(int i = 0; i < pto_data.num_goal; ++i) {

    //     meta_cost(i, 1) = (pto_data.v.row(i) - v_des).matrix().lpNorm<2>(); // 5
    //     meta_cost(i, 2) = pto_data.res_obs.row(i).matrix().lpNorm<2>();        // 6
    //     meta_cost(i, 3) = (pto_data.y.row(i) - y_g(i)).matrix().lpNorm<2>(); // 7
    //     meta_cost(i, 4) = (pto_data.v.row(i) - v_des).matrix().lpNorm<2>();                       // 8
    //     meta_cost(i, 5) = -1;
    //     meta_cost(i, 6) = -1;
    //     meta_cost(i, 7) = -1;
    //     meta_cost(i, 8) = -1;
    // }
    // float inf = std::numeric_limits<float>::infinity();
    // for(int i = 0; i < pto_data.num_goal; ++i) {
    //     float min0 = inf, min1 = inf, min2 = inf, min3 = inf; 
    //     int index0 = -1, index1 = -1, index2 = -1, index3 = -1;
    //     for(int j = 0; j < pto_data.num_goal; ++j) {
    //         if(meta_cost(j, 1) < min0 && meta_cost(j, 5) < 0) {
    //             min0 = meta_cost(j, 1);
    //             index0 = j;
    //         }
    //         if(meta_cost(j, 2) < min1 && meta_cost(j, 6) < 0) {
    //             min1 = meta_cost(j, 2);
    //             index1 = j;
    //         }
    //         if(meta_cost(j, 3) < min2 && meta_cost(j, 7) < 0) {
    //             min2 = meta_cost(j, 3);
    //             index2 = j;
    //         }
    //         if(meta_cost(j, 4) < min3 && meta_cost(j, 8) < 0) {
    //             min3 = meta_cost(j, 4);
    //             index3 = j;
    //         }
    //     }
    //     meta_cost(index0, 5) = i+1; // cruise     
    //     meta_cost(index1, 6) = i+1; // optimal
    //     meta_cost(index2, 7) = i+1; // rightmost lane
    //     meta_cost(index3, 8) = i+1; // max average velocity
    // }
} 
void MinimalPublisher :: CandidateEvaluation() {
    // cout << "vx_g: "  << " " << vx_g << " " << endl;
    for(int i = 0; i < pto_data.num_goal; ++i) {
		for (int j = 1; j < pto_data.num; ++j) {
			if (j <=5) {
				meta_cost(i, 5) += 1 * abs(pto_data.v(i, j) - vx_g(i)); // cruise- speed performance
				meta_cost(i, 6) += 1 * abs(pto_data.xdddot(i,j));        // abs(batch_a(i,j)- batch_a(i,j-1))
				meta_cost(i, 7) += 1 * abs(pto_data.y(i,j) - y_g(i)); // lane keep performance
				// meta_cost(i, 8) = (batch_v.row(i) - 24).matrix().lpNorm<2>();                       // 8
				// RCLCPP_ERROR(this->get_logger(),"  meta_cost(i, 6):  = %f,  batch_a(i,j):  = %f,  batch_a(i,j-1):  = %f",  meta_cost(i, 6),batch_a(i,j), batch_a(i,j-1));
			} else {
				meta_cost(i, 5) += exp(-(j-5)/4) * abs(pto_data.v(i, j) - vx_g(i)); // cruise- speed performance
				meta_cost(i, 6) += exp(-(j-5)/4) * abs(pto_data.xdddot(i,j)); // cruise- speed performance
				meta_cost(i, 7) += exp(-(j-5)/4) * abs(pto_data.y(i,j) - y_g(i)); // lane keep performance
			}	
		}
        meta_cost(i, 8) = pto_data.res_obs.row(i).matrix().lpNorm<2>();        // 6 
    } 
}

void MinimalPublisher :: GoalPoint() {
    // // Constant motion model for SVs
    // float t_acc, t_const, t_decc, v_temp, lon_cons, lon_dcc, lon_delta_temp, test;
    // float max_acc; 
    // float gamma = 0.8;
    // t_acc = (pto_data.ax_max - ax_init)/(pto_data.jx_max * gamma);
    // v_temp = v_init + t_acc * (pto_data.ax_max +  ax_init)/2;
    // if (v_temp > (v_init + (v_des - v_init)/2)){ // no need to acc to maximum value
    //     max_acc = sqrt(((pto_data.jx_max * gamma)  * (v_des - v_init) * 2 +  pow(ax_init ,2))/2);
    //     t_acc = (max_acc - ax_init)/(pto_data.jx_max * gamma);
    //     t_decc = max_acc/(pto_data.jx_max * gamma); 
    //     lon_delta_temp =  v_init * t_acc + 0.5 * 0.5* (ax_init + max_acc )* t_acc * t_acc 
    //                     + v_temp * t_decc + 0.5 * 0.5* max_acc * t_decc * t_decc 
    //                     + v_des * (pto_data.t_fin - t_acc - t_decc);
    //     // test =  0.5 * (v_des +v_init) *(t_acc + t_decc) + v_des * (pto_data.t_fin - t_acc - t_decc);
    //     // cout << "phase 13 ax_init" << ax_init  << "v_init:" << v_init << "t_acc:" << t_acc  << "t_decc:" << t_decc << "v_temp:" << v_temp << endl;
    //     // cout << "phase 13 max_acc:"  <<  max_acc << "lon_delta_temp" << lon_delta_temp << "test" << test << endl; 
    // } else {// keep constant acc then decc 
    //     t_acc = (pto_data.ax_max - ax_init)/(pto_data.jx_max * gamma); 
    //     t_decc = (pto_data.ax_max)/(pto_data.jx_max * gamma);
    //     t_const =((v_des - v_init) - 0.5 * (pto_data.ax_max +  ax_init)  * t_acc - 0.5 * pto_data.ax_max * t_decc)/  pto_data.ax_max;
        
    //     lon_delta_temp = v_init * t_acc + 0.5 * 0.5* (ax_init + pto_data.ax_max) * t_acc * t_acc 
    //                     + v_temp * t_const + 0.5 * pto_data.ax_max * t_const * t_const
    //                     + (v_temp + pto_data.ax_max * t_const) * t_decc  + 0.5 * 0.5 * pto_data.ax_max * t_decc * t_decc 
    //                     + v_des * (pto_data.t_fin - t_acc - t_const - t_decc);
    //     // test =  0.5 * (v_des +v_init) *(t_acc + t_decc+ t_const) + v_des * (pto_data.t_fin - t_acc - t_const - t_decc);
    //     // cout << "phase 123 ax_init" << ax_init  << "v_init:" << v_init << "t_acc:" << t_acc << "t_const:" << t_const  << "t_decc:" << t_decc << "v_temp:" << v_temp << endl;
    //     // cout << "phase 123 max_acc:"  << pto_data.ax_max << "lon_delta_temp" << lon_delta_temp << "test" << test << endl; 
    // }
    // for(int i = 0; i < pto_data.num_goal; ++i)
    //     lon_delta.row(i) = lon_delta_temp; 
}
void MinimalPublisher :: Goalfilter() {
    // Constant motion model for SVs
    ArrayXXf numbers(pto_data.num_obs, pto_data.num);
    // ArrayXXf diff_pos_x, diff_pos_y;
    for(int i = 0; i < pto_data.num_obs; ++i)
        numbers.row(i).setLinSpaced(pto_data.num, 0, pto_data.num); 
    pto_data.x_obs = (ones(pto_data.num_obs, pto_data.num)).colwise() * (x_obs_temp).col(0) + (numbers.colwise() * vx_obs.col(0) * pto_data.t);
    pto_data.y_obs = (ones(pto_data.num_obs, pto_data.num)).colwise() * (y_obs_temp).col(0) + (numbers.colwise() * vy_obs.col(0) * pto_data.t);
 
    pto_data.x_obs_fin = pto_data.x_obs.col(pto_data.x_obs.cols() - 1);
    pto_data.y_obs_fin = pto_data.y_obs.col(pto_data.y_obs.cols() - 1); 
    x_g = lon_delta + x_init;
    // cout << " lon_delta:" << lon_delta << endl;
    // x_g =  x_init +  75 * ones(x_g.rows() , x_g.cols());   
    // // static
    // pto_data.lateral_min = -8; //cruise
    // // pto_data.lateral_max = 8;
    // // road construction   //repair
    // if (x_g(0,0) >= 148 and x_g(0,0) <= 500 ) {
    //     // pto_data.lateral_max =  0;
    //     pto_data.lateral_min =  0;
    //     // pto_data.b_lateral_ineq = stack(pto_data.lateral_max * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal),  -pto_data.lateral_min * ArrayXXf :: Ones(pto_data.num,  pto_data.num_goal), 'v');
    // }
 
    y_g = y_goal_last * ones(y_g.rows() , y_g.cols()) + y_delta;
    // clip y_g
    mask_max = y_g > pto_data.lateral_max;
    y_g = mask_max.select(y_goal_last - y_delta, y_g);
    mask_min = y_g < pto_data.lateral_min;
    y_g = mask_min.select(y_goal_last - y_delta, y_g);
    y_g = y_g.max(pto_data.lateral_min).min(pto_data.lateral_max);
    pto_data.y_des = y_g.transpose().replicate(pto_data.num - pto_data.num_consensus - track_steps, 1);
    // pto_data.y_des = y_g.transpose().replicate(5, 1);
    // cout << "y_g" << y_g(0) << "\t"<< y_g(1)  << endl;
    // y_g = y_g.max(pto_data.lateral_min).min(pto_data.lateral_max);  
    //Tell is the goal point is into obs: 
    float test;
    for (int i = 0; i < pto_data.num_goal; ++i) {
        // cout << "i" <<  i << endl; 
        for (int j = 0; j < pto_data.num_obs; ++j) { 
            test = safety_dist(x_g(i,0), y_g(i,0), pto_data.x_obs_fin(j, 0), pto_data.y_obs_fin(j,0));  
            if(test < 0) { 
            //  cout << x_g(i,0) << '\t' << y_g(i,0) << '\t' << pto_data.x_obs_fin(j, 0) << '\t' << pto_data.y_obs_fin(j,0) << endl;
                // cout << "test" <<  test << endl; 
                if (safety_dist(x_g(i,0) - 5, y_g(i,0), pto_data.x_obs_fin(j, 0), pto_data.y_obs_fin(j,0)) > test) {
                    x_g(i,0) =  x_g(i,0) - 5; 
                } else {
                    x_g(i,0) =  x_g(i,0) - 10; 
                } 
                // cout << x_g(i,0) << '\t' << y_g(i,0) << '\t' << pto_data.x_obs_fin(j, 0) << '\t' << pto_data.y_obs_fin(j,0) << endl;
                // cout << "test_new" <<  safety_dist(x_g(i,0), y_g(i,0), pto_data.x_obs_fin(j, 0), pto_data.y_obs_fin(j,0)) << endl; 
                break;
            }
        }  
     }
}
 
void MinimalPublisher :: TimerCallback(const ros::TimerEvent&) {
    auto message = consensus_bphto::Controls();   
    if(Gotit) {  
        // cout << "Gotit msg" << Gotit << endl;     
        // ROS_INFO("Gotit msg: %s", Gotit ? "true" : "false");
        // ROS_WARN("Gotit msg (warning level): %s", Gotit ? "true" : "false");
        // ROS_ERROR("Gotit msg (error level): %s", Gotit ? "true" : "false"); 
        // if(abs(v_init - v_des) > 1 and v_init < v_des) {
        //     GoalPoint();
        // }
        Goalfilter();
        cnt+=1; 
        start = clock();   
        // if(loop == 0) {
        //     pto_data.maxiter = 400;
        // } else {
        //     pto_data.maxiter = 300;
        // }
  
 
        pto_data = OACP(pto_data, PPP, x_g, y_g, x_init, y_init, v_init, ax_init, ay_init, psi_init, psidot_init, warm);
        end = clock(); 
        // GetRanks();
        // cout << "111" << endl;
        CandidateEvaluation(); 
        min = 100000000;
        index = 0; 
		float min = 100000000;
		// RCLCPP_INFO(this->get_logger(),"theta_last_init:  = %f ,y_goal_last = %d , loop = %d ",  psi_init, y_goal_last, loop);
		if (abs(psi_init) > 30 * M_PI/180 and loop > 1) { // keep the target goal to the last timestep to enable the vehicle to be more stable
		    // RCLCPP_ERROR(this->get_logger()," loop = %d  , theta_last_init :  = %f ,y_goal_last = %d", loop, psi_init, y_goal_last);
			for(int i = 0; i <  pto_data.num_goal; ++i) {
				meta_cost(i, 7) = abs(y_g(i) - y_goal_last); // 7
				float cost = meta_cost(i, 7);   // last lane
				if( cost < min) {
					min = cost; 
					index = i;
				}    
			}
        } else {
            // float v_cruise = 15.0;
			for(int i = 0; i <  pto_data.num_goal; ++i) {
				// RCLCPP_INFO(this->get_logger(),"meta_cost(i, 5)_min:  = %f, meta_cost(i, 5)_max:  = %f, meta_cost(i, 5):  = %f",  meta_cost.col(5).matrix().minCoeff(), meta_cost.col(5).matrix().maxCoeff(), meta_cost(i, 5));
				// RCLCPP_INFO(this->get_logger(),"delta_to_min(i, 5):  = %f,  max_min:  = %f",  (meta_cost(i, 5)- meta_cost.col(5).matrix().minCoeff()),(meta_cost.col(5).matrix().maxCoeff() - meta_cost.col(5).matrix().minCoeff()));
				meta_cost(i, 1) = (meta_cost(i, 5)- meta_cost.col(5).matrix().minCoeff()) /(meta_cost.col(5).matrix().maxCoeff() - meta_cost.col(5).matrix().minCoeff());
				meta_cost(i, 2) = (meta_cost(i, 6)- meta_cost.col(6).matrix().minCoeff()) /(meta_cost.col(6).matrix().maxCoeff() - meta_cost.col(6).matrix().minCoeff());
				meta_cost(i, 3) = (meta_cost(i, 7)- meta_cost.col(7).matrix().minCoeff()) /(meta_cost.col(7).matrix().maxCoeff() - meta_cost.col(7).matrix().minCoeff());
				meta_cost(i, 5) = abs(y_g(i) - y_goal_last)/8;
				meta_cost(i, 4) = (meta_cost(i, 8)- meta_cost.col(8).matrix().minCoeff()) /(meta_cost.col(8).matrix().maxCoeff() - meta_cost.col(8).matrix().minCoeff());
			    // RCLCPP_WARN(this->get_logger(),"meta_cost(i, 1):  = %f,  meta_cost(i, 2):  = %f,  meta_cost(i, 3):  = %f,  meta_cost(i, 4):  = %f", meta_cost(i, 1), meta_cost(i, 2), meta_cost(i, 3), meta_cost(i, 4));
                float cost = w0 * meta_cost(i, 1) + w1 * meta_cost(i, 2) + w2 * meta_cost(i, 3) + w3 * meta_cost(i, 4) + 0 *  meta_cost(i, 5); 
									// tracking            comfortable          target lane             safety            policy switch									// cruise            comfortable          target lane             safety            policy switch
				if( cost < min) {
					min = cost; 
					index = i;
				}    
			}
		}

        // cout << "pto_data.v" <<  pto_data.v(index, 0) <<  pto_data.v(index, 1) << endl;
		y_goal_last = y_g(index);
        message.w = pto_data.psidot(index, 1);
        message.v = pto_data.v(index, 1);        
        if(loop <= 2) { // for stable consideration
            message.v = v_init;
            message.w = 0;
        }   
        ax_init =  pto_data.xddot(index, 1);        
        ay_init =  pto_data.yddot(index, 1);
        message.jx = pto_data.xdddot(index, 1);
        message.jy = pto_data.ydddot(index, 1);
        message.index = index;
        message.goals = pto_data.num_goal;
        loop++;
        // cout << pto_data.xdddot(index, 0) << " " << pto_data.xdddot(index, 1)  << " " << pto_data.xddot(index, 1) << " " << pto_data.xdot(index, 1) << endl;
        
        speed += message.v;
        total_time += double(end - start) / double(CLOCKS_PER_SEC);
        avg_speed = speed/loop;
        avg_time = total_time/loop;
	    data_recode << x_init << " " << y_init << " " << psi_init << " " << message.v  << " " << message.v  << " " << message.w
                // << " " << (message.v - prev_v_send)/pto_data.t << " " << (message.w - prev_w_send)/pto_data.t  
                << " " << pto_data.xdot(index, 1)  << " " << pto_data.ydot(index, 1) 
                << " " << pto_data.xddot(index, 1) << " "  << pto_data.yddot(index, 1)
                << " " << pto_data.xdddot(index, 1) << " "  << pto_data.ydddot(index, 1) 
                 << " "  << distance_obs(0) << " "  << distance_obs_long(0) << " "  << distance_obs_lateral(0)
                << " " << double(end - start) / double(CLOCKS_PER_SEC) << " " << loop << " " << index << "\t" << y_goal_last << endl;

		data_recode_safety << safety_barriers(0) << " " << safety_barriers(1) << " "<< safety_barriers(2)  <<  " " << loop+1 << " " << 0 << endl;
        for(int j = 0; j < pto_data.num; ++j) { 
            data_recode_receding << index  << 0   << "\t" <<  pto_data.x(0, j) << "\t" <<  pto_data.y(0, j)  << "\t" <<  pto_data.v(0, j) << "\t" <<  pto_data.xddot(0, j) << "\t" <<  pto_data.ydot(0, j) << "\t" <<  pto_data.xdot(0, j) << "\t" <<  pto_data.yddot(0, j) << endl;    
        }
        for(int j = 0; j < pto_data.num; ++j)  {
            data_recode_receding << index   << 0  << "\t" <<  pto_data.x(1, j) << "\t" <<  pto_data.y(1, j)  << "\t" <<  pto_data.v(1, j) << "\t" <<  pto_data.xddot(1, j) << "\t" <<  pto_data.ydot(1, j) << "\t" <<  pto_data.xdot(1, j) << "\t" <<  pto_data.yddot(1, j) << endl;    
        }
        prev_v_send = message.v;
		prev_w_send = message.w; 
        for(int i = 0; i < pto_data.num_goal; ++i) {
            for(int j = 0; j < pto_data.num; ++j) {
                geometry_msgs::Pose pose;
                pose.position.x = pto_data.x(i, j);
                pose.position.y = pto_data.y(i, j);
                message.batch.poses.push_back(pose);    
            }
        }
        ROS_INFO("Average time = %f Average speed = %f", avg_time, avg_speed);
        publisher_.publish(message);
        Gotit = false;
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "minimal_publisher");
    MinimalPublisher minimal_publisher;
    ros::spin();
    return 0;
}