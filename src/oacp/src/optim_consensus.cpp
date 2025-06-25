/*
 * Contact Information:
 * Lei Zheng
 * Email: zack44170625@gmail.com
 * Affiliation: HKUST/CMU
 */
#include <iostream>
#include <fstream>
#include <ctime>
#include <random>
#include <Eigen/Dense>
#include "oacp/oacp.h"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace Eigen;
YAML::Node map_t = YAML::LoadFile("/home/zmz/OACP/src/oacp/config.yaml");
string setting = map_t["setting"].as<string>();
float rho =  map_t["configuration"][setting]["rho"].as<float>();
float coefficient_rho = map_t["configuration"][setting]["coefficient_rho"].as<float>();
float coefficient_rho_d = map_t["configuration"][setting]["coefficient_rho_d"].as<float>();
namespace optim {
    clock_t start, end;
	ArrayXXf ones(int row, int col) {
		ArrayXXf temp(row, col);
		temp = 1.0;
		return temp;
	}

	ArrayXXf stack(ArrayXXf arr1, ArrayXXf arr2, char ch) {
		if (ch == 'v') {
			ArrayXXf temp(arr1.rows() + arr2.rows(), arr1.cols());
			temp << arr1, arr2;
			return temp;
		}
		else
		{
			ArrayXXf temp(arr1.rows(), arr1.cols() + arr2.cols());
			temp << arr1, arr2;
			return temp;
		}

	}//np.shape(a)== (3,) a = [1,2,3] 
 
    ArrayXXf stackVertically3(const ArrayXXf& a, const ArrayXXf& b, const ArrayXXf& c) {
		// Use the stack function to concatenate vertically
		ArrayXXf temp = stack(a, b, 'v');
		temp = stack(temp, c, 'v');
		return temp;
	}
	double clip3(double min, double max, double number) { 
		if (number > max)
			number = max;
		if (number < min)
			number = min;
		return number;
	}
	ArrayXXf clip2(ArrayXXf min, ArrayXXf max, ArrayXXf arr) {
		for (int k = 0; k < arr.cols() * arr.rows(); k++) {
			if (arr(k) > max(k))
				arr(k) = max(k);
			if (arr(k) < min(k))
				arr(k) = min(k);
		}
		return arr;
	}
	ArrayXXf clip(float min, float max, ArrayXXf arr)
	{
		for (int k = 0; k < arr.cols() * arr.rows(); k++)
		{
			if (arr(k) > max)
				arr(k) = max;
			if (arr(k) < min)
				arr(k) = min;
		}
		return arr;
	}
	ArrayXXf diff(ArrayXXf arr)
	{
		ArrayXXf temp;
		
		if(arr.cols() == 1)
			temp = ArrayXXf(arr.rows() - 1, 1);
		else
			temp = ArrayXXf(1, arr.cols() - 1);

		for (int i = 0; i < temp.rows() * temp.cols(); ++i)
		{
			temp(i) = arr(i + 1) - arr(i);
		}
		return temp;
	}
	ArrayXXf maximum(float val, ArrayXXf arr2)
	{
		ArrayXXf temp(arr2.rows(), arr2.cols());
		temp = val;

		int k = 0;
		for (int i = 0; i < arr2.cols(); ++i)
		{
			for (int j = 0; j < arr2.rows(); ++j)
			{
				if (arr2(k) > val)
					temp(k) = arr2(k);
				k++;
			}
		}
		return temp;
	}
	ArrayXXf minimum(float val, ArrayXXf arr2) {
		ArrayXXf temp(arr2.rows(), arr2.cols());
		temp = val;

		int k = 0;
		for (int i = 0; i < arr2.cols(); ++i) {
			for (int j = 0; j < arr2.rows(); ++j) {
				if (arr2(k) < val)
					temp(k) = arr2(k);
				k++;
			}
		}
		return temp;
	}
	ArrayXXf reshape(ArrayXXf x, uint32_t r, uint32_t c)
	{
		Map<ArrayXXf> rx(x.data(), r, c);
		return rx;
	}

	float binomialCoeff(float n, float k) {
		if (k == 0 || k == n)
			return 1;

		return binomialCoeff(n - 1, k - 1) +
			binomialCoeff(n - 1, k);
	}
    // Function to generate a random number with Gaussian distribution
    double generate_random_threshold(double mean, double stddev) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<> dis(mean, stddev);  
		double number = dis(gen);
        return clip3(mean - stddev, mean + stddev, number);
    }
 
		
	ArrayXXf arctan2(ArrayXXf arr1, ArrayXXf arr2) {
		ArrayXXf temp(arr1.rows(), arr1.cols());

		int k = 0;
		for (int i = 0; i < arr1.cols(); ++i) {
			for (int j = 0; j < arr1.rows(); ++j) {
				temp(k) = atan2(arr1(k), arr2(k));
				k++;
			}
		}
		return temp;
	}

	ArrayXXf cumsum(ArrayXXf arr1, ArrayXXf arr2) {
		float init = arr1(0);
		for (int i = 0; i < arr1.cols(); ++i)
			arr1.col(i) = arr1.col(i) * arr2;
		int k = 1;
		for (int j = 0; j < arr1.cols(); ++j) {
			for (int i = 1; i < arr1.rows(); ++i) {
				arr1(k) = arr1(k) + arr1(k - 1);
				k++;
			}
			k++;
		}
		return arr1;
	}
	void shape(ArrayXXf arr) {
		cout << arr.rows() << " " << arr.cols();
	}
	float euclidean_dist(float x1, float y1, float x2, float y2) {
		return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
	}
	//___
	float safety_dist(float x1, float y1, float x2, float y2) {
		return (pow((x1 - x2)/ 5.6, 2) + pow((y1 - y2)/ 3.0, 2) -1);
	}
  
  	four_var BernsteinCoeffOrder10(float n, float tmin, float tmax, ArrayXXf t_actual, int num) {
		four_var s;
		float l = tmax - tmin;
		ArrayXXf t = (t_actual - tmin) / l;

		ArrayXXf P(num, (int)n + 1), Pdot(num, (int)n + 1), Pddot(num, (int)n + 1), Pdddot(num, (int)n + 1), Pddddot(num, (int)n + 1);

		for (int i = 0; i <= n; ++i) { 
			float coeff = binomialCoeff(n, i); 
			P.col(i) = coeff * pow(1 - t, n - i) * pow(t, i); 
		}  
		  
		Pdot.col(0) = -10.0 * pow(-t + 1, 9);
		Pdot.col(1) = -90.0 * t * pow(-t + 1, 8) + 10.0 * pow(-t + 1, 9);
		Pdot.col(2) = -360.0 * pow(t, 2) * pow(-t + 1, 7) + 90.0 * t * pow(-t + 1, 8);
		Pdot.col(3) = -840.0 * pow(t, 3) * pow(-t + 1, 6) + 360.0 * pow(t, 2) * pow(-t + 1, 7);
		Pdot.col(4) = -1260.0 * pow(t, 4) * pow(-t + 1, 5) + 840.0 * pow(t, 3) * pow(-t + 1, 6);
		Pdot.col(5) = -1260.0 * pow(t, 5) * pow(-t + 1, 4) + 1260.0 * pow(t, 4) * pow(-t + 1, 5);
		Pdot.col(6) = -840.0 * pow(t, 6) * pow(-t + 1, 3) + 1260.0 * pow(t, 5) * pow(-t + 1, 4);
		Pdot.col(7) = -360.0 * pow(t, 7) * pow(-t + 1, 2) + 840.0 * pow(t, 6) * pow(-t + 1, 3);
		Pdot.col(8) = 45.0 * pow(t, 8) * (2 * t - 2) + 360.0 * pow(t, 7) * pow(-t + 1, 2);
		Pdot.col(9) = -10.0 * pow(t, 9) + 9 * pow(t, 8) * (-10.0 * t + 10.0);
		Pdot.col(10) = 10.0 * pow(t, 9);

		Pddot.col(0) = 90.0 * pow(-t + 1, 8.0);
		Pddot.col(1) = 720.0 * t * pow(-t + 1, 7) - 180.0 * pow(-t + 1, 8);
		Pddot.col(2) = 2520.0 * pow(t, 2) * pow(-t + 1, 6) - 1440.0 * t * pow(-t + 1, 7) + 90.0 * pow(-t + 1, 8);
		Pddot.col(3) = 5040.0 * pow(t, 3) * pow(-t + 1, 5) - 5040.0 * pow(t, 2) * pow(-t + 1, 6) + 720.0 * t * pow(-t + 1, 7);
		Pddot.col(4) = 6300.0 * pow(t, 4) * pow(-t + 1, 4) - 10080.0 * pow(t, 3) * pow(-t + 1, 5) + 2520.0 * pow(t, 2) * pow(-t + 1, 6);
		Pddot.col(5) = 5040.0 * pow(t, 5) * pow(-t + 1, 3) - 12600.0 * pow(t, 4) * pow(-t + 1, 4) + 5040.0 * pow(t, 3) * pow(-t + 1, 5);
		Pddot.col(6) = 2520.0 * pow(t, 6) * pow(-t + 1, 2) - 10080.0 * pow(t, 5) * pow(-t + 1, 3) + 6300.0 * pow(t, 4) * pow(-t + 1, 4);
		Pddot.col(7) = -360.0 * pow(t, 7) * (2 * t - 2) - 5040.0 * pow(t, 6) * pow(-t + 1, 2) + 5040.0 * pow(t, 5) * pow(-t + 1, 3);
		Pddot.col(8) = 90.0 * pow(t, 8) + 720.0 * pow(t, 7) * (2 * t - 2) + 2520.0 * pow(t, 6) * pow(-t + 1, 2);
		Pddot.col(9) = -180.0 * pow(t, 8) + 72 * pow(t, 7) * (-10.0 * t + 10.0);
		Pddot.col(10) = 90.0 * pow(t, 8);

		Pdddot.col(0) = -720.0 * pow(-t + 1, 7);
		Pdddot.col(1) = 2160 * pow(1 - t, 7) - 5040 * pow(1 - t, 6) * t;
		Pdddot.col(2) = -15120 * pow(1-t, 5) * pow(t, 2) + 15120 * pow(1 - t, 6) * t - 2160 * pow(1 - t, 7);
		Pdddot.col(3) = -720 * pow(t-1, 4) * (120 * pow(t, 3) - 108 * pow(t, 2) + 24 * t - 1);
		Pdddot.col(4) = 5040 * pow(t-1, 3) * t * (30 * pow(t, 3) - 36*pow(t, 2) + 12 * t - 1);
		Pdddot.col(5) = -15120 * pow(t-1, 2) * pow(t, 2) * (12 * pow(t, 3) - 18*pow(t, 2) + 8 * t - 1);
		Pdddot.col(6) = 5040 * (t - 1) * pow(t, 3) * (30 * pow(t, 3)-54 * pow(t, 2) + 30 * t - 5);
		Pdddot.col(7) = -720 * pow(t, 7) + 10080 * (1-t) * pow(t, 6) - 45360 * pow(1 - t, 2) * pow(t, 5) + 25200 * pow(1-t, 3) * pow(t, 4) - 2520 * pow(t, 6) * (2 * t - 2);
		Pdddot.col(8) = 2160 * pow(t, 7) - 5040 * (1 - t) * pow(t, 6) + 15120 * pow(1-t, 2) * pow(t, 5) + 5040 * pow(t, 6) * (2 * t - 2);
		Pdddot.col(9) = 504 * (10 - 10 * t) * pow(t, 6) - 2160 * pow(t, 7);
		Pdddot.col(10) = 720.0 * pow(t, 7);
 
		s.a = P;
		s.b = Pdot / l;
		s.c = Pddot / (l * l);
		s.d = Pdddot / (l * l * l);
  
		return s;
	}
	 
	probData solve(probData pto_data, ArrayXXf P, ArrayXXf Pdot, ArrayXXf Pddot, ArrayXXf Pdddot, ArrayXXf x_init, ArrayXXf x_fin, ArrayXXf y_init, ArrayXXf y_fin, ArrayXXf v_init,  ArrayXXf ax_init, ArrayXXf ay_init,  ArrayXXf psi_init, ArrayXXf psidot_init, ArrayXXf psi_fin, ArrayXXf psidot_fin, ArrayXXf x_obs, ArrayXXf y_obs, bool warm ) {
        pto_data.rho_consensus_psi = rho- 0;        
		pto_data.rho_consensus_x = rho - 0;
		pto_data.rho_consensus_y = rho - 0; 
	    pto_data.rho_psi = rho- 0;
        pto_data.rho_obs = rho + 0;
		pto_data.rho_lateral_long = rho- 0;
		pto_data.rho_vel = rho- 0;
		pto_data.rho_acc = rho- 0;
		pto_data.rho_jerk = rho- 0;
		ArrayXXf res_psi_new;
		ArrayXXf vx_init, vy_init, b_eq_x, b_eq_y, b_eq_psi,
		res_psi, res_nonhol, res_obs, res_vel, res_acc, res_lateral, res_longitudinal, res_jx_vec, res_jy_vec, res_jerk_vec, 
		res_consensus_x, res_consensus_y, res_consensus_psi,
		res_eq_x, res_nonhol_x, res_eq_y, res_nonhol_y, res_x_obs_vec, res_y_obs_vec, 
		wc_alpha_longitude, wc_alpha_lateral, ws_alpha_lateral, res_longitudinal_vec, res_lateral_vec, res_latlon_vec,
		wc_alpha_ax, ws_alpha_ay, res_vx_vec, res_vy_vec, res_ax_vec, res_ay_vec, psi_temp, c_psi, c_x, c_y, c_psi_old, c_x_old, c_y_old,
		b_lateral_min, b_lateral_max;
		  
		vx_init = v_init * cos(psi_init);
		vy_init = v_init * sin(psi_init);
 
		b_eq_x = ArrayXXf(pto_data.num_goal, 2);
		b_eq_x << x_init, vx_init;

		b_eq_y = ArrayXXf(pto_data.num_goal, 3);
		b_eq_y << y_init, vy_init, y_fin;

		b_eq_psi = ArrayXXf(pto_data.num_goal, 4);
		b_eq_psi << psi_init, psidot_init, psi_fin, psidot_fin;
		
		res_psi = ones(pto_data.maxiter, 1);
		res_nonhol = ones(pto_data.maxiter, 1);
		res_lateral = ones(pto_data.maxiter, 1);
		res_longitudinal = ones(pto_data.maxiter, 1);
		res_obs = ones(pto_data.maxiter, 1);
		res_vel = ones(pto_data.maxiter, 1);
		res_acc = ones(pto_data.maxiter, 1);
 
		pto_data.v = ones(pto_data.num_goal, pto_data.num).colwise() * v_init.col(0) ;
		pto_data.psi = ones(pto_data.num_goal, pto_data.num).colwise() * psi_init.col(0);
		
		pto_data.xdot = pto_data.v * cos(pto_data.psi);
		pto_data.ydot = pto_data.v * sin(pto_data.psi);
		
		int i = 0;
		// for(int i = 0; i < pto_data.maxiter; ++i)
		do {
			if(i >= pto_data.maxiter)
				break;
			psi_temp = arctan2(pto_data.ydot, pto_data.xdot);
			three_varr s_xy, s_psi; 
			s_psi = compute_psi(pto_data, P, Pdot, Pddot, psi_temp, b_eq_psi);
			c_psi = s_psi.a;
			res_psi(i) = s_psi.b.matrix().lpNorm<2>();
			pto_data = s_psi.c;
			// cout<< "res_psi" << i << '\t' << res_psi(i)  << endl;
			// break;
			start = clock();
			s_xy = compute_xy(pto_data, P, Pdot, Pddot, Pdddot, b_eq_x, b_eq_y, x_obs, y_obs);	
			end = clock();
		    // cout<< "time: xy"<< " "  << double(end - start) / double(CLOCKS_PER_SEC)<< endl;
			c_x = s_xy.a;
			c_y = s_xy.b;
			pto_data = s_xy.c; 
			pto_data.v = sqrt(pow(pto_data.xdot, 2) + pow(pto_data.ydot, 2));
			pto_data.v = clip(0.0,  pto_data.v_max , pto_data.v);
			pto_data.v.col(0) = pto_data.v_init.col(0); 
			res_nonhol_x = pto_data.xdot - pto_data.v * cos(pto_data.psi); 
			res_nonhol_y = pto_data.ydot - pto_data.v * sin(pto_data.psi);
            // consensus psi
			// Update consensus variable s   
			// Calculate and expand averages, then extract the first Nc rows to s_consensus
			pto_data.s_consensus_psi = pto_data.psi.transpose().matrix().rowwise().mean().replicate(1, pto_data.psi.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);
 		    pto_data.s_consensus_x.block(0, 0, pto_data.num_consensus, pto_data.num_goal)  = pto_data.x.transpose().matrix().rowwise().mean().replicate(1, pto_data.x.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);
		    pto_data.s_consensus_x.block(pto_data.num_consensus, 0, pto_data.num_consensus, pto_data.num_goal)  = pto_data.xdot.transpose().matrix().rowwise().mean().replicate(1, pto_data.xdot.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);
		    pto_data.s_consensus_x.block(2 *pto_data.num_consensus, 0, pto_data.num_consensus, pto_data.num_goal)  = pto_data.xddot.transpose().matrix().rowwise().mean().replicate(1, pto_data.xddot.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);
  			pto_data.s_consensus_y.block(0, 0, pto_data.num_consensus, pto_data.num_goal)  = pto_data.y.transpose().matrix().rowwise().mean().replicate(1, pto_data.y.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);
		    pto_data.s_consensus_y.block(pto_data.num_consensus, 0, pto_data.num_consensus, pto_data.num_goal)  = pto_data.ydot.transpose().matrix().rowwise().mean().replicate(1, pto_data.ydot.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);
		    pto_data.s_consensus_y.block(2 *pto_data.num_consensus, 0, pto_data.num_consensus, pto_data.num_goal)  = pto_data.yddot.transpose().matrix().rowwise().mean().replicate(1, pto_data.yddot.cols()).topRows(pto_data.num_consensus).leftCols(pto_data.num_goal);

	 
			// obs
			ArrayXXf wc_alpha(pto_data.num_goal, pto_data.num_obs*pto_data.num);
			ArrayXXf ws_alpha(pto_data.num_goal, pto_data.num_obs*pto_data.num);
			ArrayXXf randm, randm2(1, pto_data.num_obs*pto_data.num), randm3(1, pto_data.num_obs*pto_data.num);
			for(int i = 0; i < pto_data.num_goal; ++i) {
				randm = (-pto_data.x_obs).rowwise() + pto_data.x.row(i);
				randm2 = reshape(randm.transpose(), 1, pto_data.num_obs*pto_data.num);  
				wc_alpha.row(i) = randm2;

				randm = (-pto_data.y_obs).rowwise() + pto_data.y.row(i);
				randm3 = reshape(randm.transpose(), 1, pto_data.num_obs*pto_data.num);
				ws_alpha.row(i) = randm3;
			} 
			pto_data.alpha_obs = arctan2(ws_alpha * pto_data.a_obs_vec, wc_alpha * pto_data.b_obs_vec);
			// num_traj / num_obs*num_horizon
			ArrayXXf c1_d, c2_d, d_temp, long_pos;
			c1_d = pow(pto_data.a_obs_vec, 2) * pow(cos(pto_data.alpha_obs), 2) + pow(pto_data.b_obs, 2) * pow(sin(pto_data.alpha_obs), 2);
			c2_d = pto_data.a_obs_vec * wc_alpha * cos(pto_data.alpha_obs) + pto_data.b_obs * ws_alpha*sin(pto_data.alpha_obs);
			d_temp = c2_d/c1_d;

			pto_data.d_obs = d_temp.max(1.0 + (1.0 - pto_data.gamma_matrices) * (pto_data.d_obs_old - 1.0));		
			// pto_data.d_obs = maximum(1, d_temp); 
 		    // cout << "gamma value" <<  " " << pto_data.gamma << " " << endl; 
			res_x_obs_vec = wc_alpha-pto_data.a_obs_vec*pto_data.d_obs*cos(pto_data.alpha_obs);
        	res_y_obs_vec = ws_alpha-pto_data.b_obs_vec*pto_data.d_obs*sin(pto_data.alpha_obs);

			//lateral road boundaries
   			// position_box_constraints_res
 			long_pos =  pto_data.A_lateral_long.matrix() * c_x.transpose().matrix(); 
 		  	//  ADMM
			pto_data.s_x_ineq = (- pto_data.alpha_admm * ((pto_data.A_lateral_long.matrix() * c_x.transpose().matrix()).array() - pto_data.b_longitudinal_ineq)
								+ (1-pto_data.alpha_admm)*pto_data.s_x_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.
 
		    b_lateral_min = pto_data.lateral_min * ArrayXXf :: Ones(pto_data.num, pto_data.num_goal);
		    b_lateral_max = pto_data.lateral_max * ArrayXXf :: Ones(pto_data.num, pto_data.num_goal);
		 
			pto_data.b_lateral_ineq = stack(b_lateral_max, -b_lateral_min, 'v');  
			// pto_data.s_y_ineq = ((- pto_data.A_lateral_long.matrix() * c_y.transpose().matrix()).array() + pto_data.b_lateral_ineq).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.
			// ADMM
			pto_data.s_y_ineq = (- pto_data.alpha_admm * ((pto_data.A_lateral_long.matrix() * c_y.transpose().matrix()).array() - pto_data.b_lateral_ineq)
								+ (1-pto_data.alpha_admm)*pto_data.s_y_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.
		    // cout << "A_lateral_long*c_x value" <<  " " << (- pto_data.A_lateral_long.matrix() * c_x.transpose().matrix()).array() << " " << endl;

			res_lateral_vec = (pto_data.A_lateral_long.matrix() * c_y.transpose().matrix()).array() - pto_data.b_lateral_ineq + pto_data.s_y_ineq;
			res_longitudinal_vec = (pto_data.A_lateral_long.matrix() * c_x.transpose().matrix()).array() - pto_data.b_longitudinal_ineq + pto_data.s_x_ineq;

		   	// velocity constraints_res
			pto_data.s_vy_ineq = (- pto_data.alpha_admm * ((pto_data.A_vel.matrix() * c_y.transpose().matrix()).array() - pto_data.b_vy_ineq)
								+ (1-pto_data.alpha_admm)*pto_data.s_vy_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.

			pto_data.s_vx_ineq = (- pto_data.alpha_admm * ((pto_data.A_vel.matrix() * c_x.transpose().matrix()).array() - pto_data.b_vx_ineq)
								+ (1-pto_data.alpha_admm)*pto_data.s_vx_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.
  
			res_vy_vec = (pto_data.A_vel.matrix() * c_y.transpose().matrix()).array() - pto_data.b_vy_ineq + pto_data.s_vy_ineq;
			res_vx_vec = (pto_data.A_vel.matrix() * c_x.transpose().matrix()).array() - pto_data.b_vx_ineq + pto_data.s_vx_ineq; 
		   	// acceleration constraints_res
			pto_data.s_ay_ineq = (- pto_data.alpha_admm * ((pto_data.A_acc.matrix() * c_y.transpose().matrix()).array() - pto_data.b_ay_ineq)
								+ (1-pto_data.alpha_admm)*pto_data.s_ay_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.

			pto_data.s_ax_ineq = (- pto_data.alpha_admm * ((pto_data.A_acc.matrix() * c_x.transpose().matrix()).array() - pto_data.b_ax_ineq)
						+ (1-pto_data.alpha_admm)*pto_data.s_ax_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.
  
			res_ay_vec = (pto_data.A_acc.matrix() * c_y.transpose().matrix()).array() - pto_data.b_ay_ineq + pto_data.s_ay_ineq;
			res_ax_vec = (pto_data.A_acc.matrix() * c_x.transpose().matrix()).array() - pto_data.b_ax_ineq + pto_data.s_ax_ineq; 
			// jerk_box_constraints_res  
		    // d ADMM
			pto_data.s_jy_ineq = (- pto_data.alpha_admm * ((pto_data.A_jerk.matrix() * c_y.transpose().matrix()).array() - pto_data.b_jy_ineq)
								+ (1-pto_data.alpha_admm)*pto_data.s_jy_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.

			pto_data.s_jx_ineq = (- pto_data.alpha_admm * ((pto_data.A_jerk.matrix() * c_x.transpose().matrix()).array() - pto_data.b_jx_ineq)
						+ (1-pto_data.alpha_admm)*pto_data.s_jx_ineq_old).max(0.0); //if the value inside the parentheses is less than 0.0, it will be replaced with 0.0. If it is greater than or equal to 0.0, it will remain the same.
  
			res_jy_vec = (pto_data.A_jerk.matrix() * c_y.transpose().matrix()).array() - pto_data.b_jy_ineq + pto_data.s_jy_ineq;
			res_jx_vec = (pto_data.A_jerk.matrix() * c_x.transpose().matrix()).array() - pto_data.b_jx_ineq + pto_data.s_jx_ineq;
		    // cout << "shape" << res_jy_vec.matrix().rows() << "\t" << res_jy_vec.matrix().cols() << endl;	100/5 
			res_nonhol_x = pto_data.xdot - pto_data.v * cos(pto_data.psi); 
			res_nonhol_y = pto_data.ydot - pto_data.v * sin(pto_data.psi);  
			res_psi_new = (pto_data.A_psi.matrix() * c_psi.transpose().matrix()).transpose().array() - arctan2(pto_data.ydot, pto_data.xdot);
 		    // cout<< "0011" << endl; 
			pto_data.consensus_actual_psi =  pto_data.psi.transpose().matrix().topRows(pto_data.num_consensus);
			// pto_data.consensus_actual_x =  stack(pto_data.x.transpose().matrix().topRows(pto_data.num_consensus),pto_data.xdot.transpose().matrix().topRows(pto_data.num_consensus),'v');
			// pto_data.consensus_actual_y =  stack(pto_data.y.transpose().matrix().topRows(pto_data.num_consensus),pto_data.ydot.transpose().matrix().topRows(pto_data.num_consensus),'v');
			pto_data.consensus_actual_x =  stackVertically3(pto_data.x.transpose().matrix().topRows(pto_data.num_consensus),pto_data.xdot.transpose().matrix().topRows(pto_data.num_consensus), pto_data.xddot.transpose().matrix().topRows(pto_data.num_consensus));
			pto_data.consensus_actual_y =  stackVertically3(pto_data.y.transpose().matrix().topRows(pto_data.num_consensus),pto_data.ydot.transpose().matrix().topRows(pto_data.num_consensus), pto_data.yddot.transpose().matrix().topRows(pto_data.num_consensus));
 			res_consensus_psi = pto_data.consensus_actual_psi - pto_data.s_consensus_psi;
			res_consensus_x = pto_data.consensus_actual_x - pto_data.s_consensus_x;
			res_consensus_y = pto_data.consensus_actual_y - pto_data.s_consensus_y;
			// N_consensus/*5 
   			// cout << "res of s_consensus: (" << i << ", " << res_consensus_psi.row(0).matrix().norm() << "\t "  << res_consensus_psi.row(1).matrix().norm()  
			// 			<< "\t "  << res_consensus_psi.row(2).matrix().norm()  << "\t "  << res_consensus_psi.row(3).matrix().norm() << ")" << endl;

			// A_psi 50* 11 
 			pto_data.lamda_psi = pto_data.lamda_psi - pto_data.rho_psi * res_psi_new.transpose().matrix().array();
		    // cout<< "res_psi" << '\t' << res_psi_new.matrix().lpNorm<2>()  << endl;
 			pto_data.lamda_obsx = pto_data.lamda_obsx - pto_data.rho_obs *  res_x_obs_vec.transpose().matrix().array();
			pto_data.lamda_obsy = pto_data.lamda_obsy - pto_data.rho_obs *  res_y_obs_vec.transpose().matrix().array();
			pto_data.lamda_x = pto_data.lamda_x
 								-pto_data.rho_lateral_long * (pto_data.A_lateral_long.transpose().matrix() * (pto_data.alpha_admm * res_longitudinal_vec.matrix()+ (1-pto_data.alpha_admm)* (pto_data.s_x_ineq - pto_data.s_x_ineq_old).matrix())).transpose().array()
								-pto_data.rho_vel * (pto_data.A_vel.transpose().matrix() * (pto_data.alpha_admm * res_vx_vec.matrix() + (1-pto_data.alpha_admm)* (pto_data.s_vx_ineq - pto_data.s_vx_ineq_old).matrix())).transpose().array()
								-pto_data.rho_acc * (pto_data.A_acc.transpose().matrix() * (pto_data.alpha_admm * res_ax_vec.matrix() + (1-pto_data.alpha_admm)* (pto_data.s_ax_ineq - pto_data.s_ax_ineq_old).matrix())).transpose().array()
								-pto_data.rho_jerk * (pto_data.A_jerk.transpose().matrix() * (pto_data.alpha_admm * res_jx_vec.matrix() + (1-pto_data.alpha_admm)* (pto_data.s_jx_ineq - pto_data.s_jx_ineq_old).matrix())).transpose().array();
 			pto_data.lamda_y = pto_data.lamda_y
  								-pto_data.rho_lateral_long * (pto_data.A_lateral_long.transpose().matrix() * (pto_data.alpha_admm * res_lateral_vec.matrix()+ (1-pto_data.alpha_admm)* (pto_data.s_y_ineq - pto_data.s_y_ineq_old).matrix())).transpose().array()
							    -pto_data.rho_vel * (pto_data.A_vel.transpose().matrix() * (pto_data.alpha_admm * res_vy_vec.matrix() + (1-pto_data.alpha_admm)* (pto_data.s_vy_ineq - pto_data.s_vy_ineq_old).matrix())).transpose().array()
							    -pto_data.rho_acc * (pto_data.A_acc.transpose().matrix() * (pto_data.alpha_admm * res_ay_vec.matrix() + (1-pto_data.alpha_admm)* (pto_data.s_ay_ineq - pto_data.s_ay_ineq_old).matrix())).transpose().array()
								-pto_data.rho_jerk * (pto_data.A_jerk.transpose().matrix() * (pto_data.alpha_admm * res_jy_vec.matrix() + (1-pto_data.alpha_admm)* (pto_data.s_jy_ineq - pto_data.s_jy_ineq_old).matrix())).transpose().array();
 			pto_data.lamda_consensus_psi = pto_data.lamda_consensus_psi - pto_data.rho_consensus_psi* res_consensus_psi.matrix().array();			 
 			pto_data.lamda_consensus_x = pto_data.lamda_consensus_x - pto_data.rho_consensus_x* res_consensus_x.matrix().array();			 
 			pto_data.lamda_consensus_y = pto_data.lamda_consensus_y - pto_data.rho_consensus_y* res_consensus_y.matrix().array();			 
 	
			res_obs(i) = stack(res_x_obs_vec, res_y_obs_vec, 'v').matrix().lpNorm<2>();
			res_vel(i) = stack(res_vx_vec, res_vy_vec, 'v').matrix().lpNorm<2>();
			res_acc(i) = stack(res_ax_vec, res_ay_vec, 'v').matrix().lpNorm<2>();
			res_nonhol(i) = stack(res_nonhol_x, res_nonhol_y, 'v').matrix().lpNorm<2>();
			res_lateral(i) = res_lateral_vec.matrix().lpNorm<2>() ;
			res_longitudinal(i) = res_longitudinal_vec.matrix().lpNorm<2>();
			// res_jerk_vec(i) = stack(res_jx_vec, res_jy_vec, 'v').matrix().lpNorm<2>();
			// stack(res_jx_vec, res_jy_vec, 'v').matrix().lpNorm<2>(); 
			pto_data.d_obs_old = pto_data.d_obs;
		    pto_data.s_x_ineq_old = pto_data.s_x_ineq;
		    pto_data.s_y_ineq_old = pto_data.s_y_ineq;
			pto_data.s_vx_ineq_old = pto_data.s_vx_ineq;
			pto_data.s_vy_ineq_old = pto_data.s_vy_ineq;
			pto_data.s_ax_ineq_old = pto_data.s_ax_ineq;
			pto_data.s_ay_ineq_old = pto_data.s_ay_ineq;
			pto_data.s_jx_ineq_old = pto_data.s_jx_ineq;
			pto_data.s_jy_ineq_old = pto_data.s_jy_ineq;
	 	    

			if (res_obs(i) <= 0.1 && res_vel(i) <= 0.05 && res_acc(i) <= 0.1 && res_nonhol(i) <= 0.1 && res_lateral(i) <= 0.1 && res_longitudinal(i) <= 0.1 && res_jy_vec.matrix().lpNorm<2>()  <= 0.1 &&  res_jx_vec.matrix().lpNorm<2>()  <= 0.1) {
				cout  << i  << "\t" << "converge break:" << "\t" << res_obs(i) <<  " " << res_vel(i) <<  " " << res_acc(i) << " " << res_nonhol(i) << endl;
		        cout  << i  << "\t" << "converge break:" << "\t"<< res_jx_vec.matrix().lpNorm<2>() <<  " "<< res_jy_vec.matrix().lpNorm<2>() <<  " "<< res_lateral(i) <<  " " << res_longitudinal(i) << endl;
				break;
			}
		  
			pto_data.rho_lateral_long = pto_data.rho_lateral_long * coefficient_rho;
			pto_data.rho_vel = pto_data.rho_vel * coefficient_rho;
			pto_data.rho_acc = pto_data.rho_acc * coefficient_rho;
			pto_data.rho_jerk = pto_data.rho_jerk * coefficient_rho;
			c_psi_old = c_psi; // 5* 11
			c_x_old = c_x; // 5* 11
			c_y_old = c_y; // 5* 11
			i++;
		} while(1);//res_obs(i-1) >= 0.1 || res_acc(i - 1) >= 0.001 || res_nonhol(i - 1) >= 0.1);
 
		pto_data.res_vel = res_vel;
		pto_data.res_acc = res_acc;
		pto_data.res_nonhol = res_nonhol;
		pto_data.res_obs = stack(res_x_obs_vec, res_y_obs_vec, 'h'); 

		pto_data.v_controls = sqrt(pow((pto_data.Pdot_upsample.matrix() * c_x.transpose().matrix()).transpose().array(), 2)
								+ pow((pto_data.Pdot_upsample.matrix() * c_y.transpose().matrix()).transpose().array(), 2));
		pto_data.w_controls = (pto_data.Pdot_upsample.matrix() * c_psi.transpose().matrix()).transpose();
		return pto_data;
	}
	three_varr compute_xy(probData pto_data, ArrayXXf P, ArrayXXf Pdot, ArrayXXf Pddot, ArrayXXf Pdddot, ArrayXXf b_eq_x, ArrayXXf b_eq_y, ArrayXXf x_obs, ArrayXXf y_obs) {
		three_varr s;

		ArrayXXf temp_x_obs, temp_y_obs, b_obs_x, b_obs_y, b_nonhol_x, b_nonhol_y,   
				b_longitudinal_ineq, b_lateral_ineq, b_vx_ineq, b_vy_ineq, b_ax_ineq, b_ay_ineq, b_jy_ineq, b_jx_ineq, 
				cost_x, cost_y, cost_mat_inv_x,  cost_mat_inv_y, lincost_x, lincost_y, sol_x, sol_y, primal_sol_x, primal_sol_y;
		MatrixXf cost_mat_x, cost_mat_y;
  
		temp_x_obs = pto_data.d_obs * cos(pto_data.alpha_obs) * pto_data.a_obs_vec;
		temp_y_obs = pto_data.d_obs * sin(pto_data.alpha_obs) * pto_data.b_obs_vec;
		
		ArrayXXf randm1(pto_data.num*pto_data.num_obs, 1), randm2(pto_data.num*pto_data.num_obs, 1);
		
		randm1 = reshape(pto_data.x_obs.transpose(), pto_data.num*pto_data.num_obs, 1);
		randm2 = reshape(pto_data.y_obs.transpose(), pto_data.num*pto_data.num_obs, 1);

		b_obs_x = temp_x_obs.rowwise() + randm1.col(0).transpose();  
		b_obs_y = temp_y_obs.rowwise() + randm2.col(0).transpose();
		//   5 250 
 		//  pto_data.A_obs.matrix() 
		// 250 11
		// start = clock(); 
		b_nonhol_x = pto_data.v * cos(pto_data.psi);
		b_nonhol_y = pto_data.v * sin(pto_data.psi); 

		b_lateral_ineq = pto_data.b_lateral_ineq - pto_data.s_y_ineq;
		b_longitudinal_ineq = pto_data.b_longitudinal_ineq - pto_data.s_x_ineq;
   
		b_vy_ineq = pto_data.b_vy_ineq - pto_data.s_vy_ineq;
		b_vx_ineq = pto_data.b_vx_ineq - pto_data.s_vx_ineq;
        
		b_ay_ineq = pto_data.b_ay_ineq - pto_data.s_ay_ineq;
		b_ax_ineq = pto_data.b_ax_ineq - pto_data.s_ax_ineq;
     
		b_jy_ineq = pto_data.b_jy_ineq - pto_data.s_jy_ineq;
		b_jx_ineq = pto_data.b_jx_ineq - pto_data.s_jx_ineq;
  
		cost_x = pto_data.cost_smoothness 
			    + pto_data.weight_vel_tracking*(pto_data.A_tracking_vel.transpose().matrix()* pto_data.A_tracking_vel.matrix()).array()
				+ pto_data.rho_psi*(pto_data.A_nonhol.transpose().matrix() * pto_data.A_nonhol.matrix()).array()
				+ pto_data.rho_lateral_long*(pto_data.A_lateral_long.transpose().matrix() * pto_data.A_lateral_long.matrix()).array()
				+ pto_data.rho_vel*(pto_data.A_vel.transpose().matrix() * pto_data.A_vel.matrix()).array()
				+ pto_data.rho_acc*(pto_data.A_acc.transpose().matrix() * pto_data.A_acc.matrix()).array()
				+ pto_data.rho_jerk*(pto_data.A_jerk.transpose().matrix() * pto_data.A_jerk.matrix()).array()
				+ pto_data.rho_obs*(pto_data.A_obs.transpose().matrix() * pto_data.A_obs.matrix()).array()
			    + pto_data.rho_consensus_x*(pto_data.A_consensus_x.transpose().matrix() * pto_data.A_consensus_x.matrix()).array();//Q_{x,y}
		cost_y = pto_data.cost_smoothness 
				+ pto_data.weight_lane_tracking*(pto_data.A_tracking_lateral.transpose().matrix()* pto_data.A_tracking_lateral.matrix()).array()
				+ pto_data.rho_psi*(pto_data.A_nonhol.transpose().matrix() * pto_data.A_nonhol.matrix()).array()
				+ pto_data.rho_lateral_long*(pto_data.A_lateral_long.transpose().matrix() * pto_data.A_lateral_long.matrix()).array()
				+ pto_data.rho_vel*(pto_data.A_vel.transpose().matrix() * pto_data.A_vel.matrix()).array()
				+ pto_data.rho_acc*(pto_data.A_acc.transpose().matrix() * pto_data.A_acc.matrix()).array()
				+ pto_data.rho_jerk*(pto_data.A_jerk.transpose().matrix() * pto_data.A_jerk.matrix()).array()
				+ pto_data.rho_obs*(pto_data.A_obs.transpose().matrix() * pto_data.A_obs.matrix()).array()
			    + pto_data.rho_consensus_y*(pto_data.A_consensus_y.transpose().matrix() * pto_data.A_consensus_y.matrix()).array();//Q_{x,y}

		// cout << "cost=" << cost.matrix().lpNorm<2>() << endl;
		cost_mat_x = stack(stack(cost_x, pto_data.A_eq_x.transpose(), 'h'), stack(pto_data.A_eq_x, 0.0*ones(pto_data.A_eq_x.rows(), pto_data.A_eq_x.rows()), 'h'), 'v');
		cost_mat_y = stack(stack(cost_y, pto_data.A_eq_y.transpose(), 'h'), stack(pto_data.A_eq_y, 0.0*ones(pto_data.A_eq_y.rows(), pto_data.A_eq_y.rows()), 'h'), 'v');
        // cout << "cost_mat_x size=" << cost_mat_x.rows() << '\t' << cost_mat_x.cols() << endl;
 
		// end = clock();
		// cout<< "time1: "<< " "  << double(end - start) / double(CLOCKS_PER_SEC) << endl;
		MatrixXf I_x(cost_mat_x.rows(), cost_mat_x.cols());
	    I_x.setIdentity(); // Ax = I; x = A-1
	    cost_mat_inv_x = (cost_mat_x).householderQr().solve(I_x);//Q_{x,y}^{-1}
 		MatrixXf I_y(cost_mat_y.rows(), cost_mat_y.cols());
	    I_y.setIdentity(); // Ax = I; x = A-1
	    cost_mat_inv_y = (cost_mat_y).householderQr().solve(I_y);//Q_{x,y}^{-1}
  
		lincost_x =  pto_data.lamda_x
					 + pto_data.weight_vel_tracking * (pto_data.A_tracking_vel.transpose().matrix() * pto_data.vx_des.matrix()).transpose().array()
					 +(pto_data.A_nonhol.transpose().matrix() * pto_data.lamda_psi.matrix()).transpose().array() // 11/50 50/5 ^T =           5*11
					 +(pto_data.A_obs.transpose().matrix() * pto_data.lamda_obsx.matrix()).transpose().array() // 11/50 50/5 ^T =           5*11
 					 + pto_data.rho_psi * (pto_data.A_nonhol.transpose().matrix() * b_nonhol_x.transpose().matrix()).transpose().array()
					 + pto_data.rho_lateral_long * (pto_data.A_lateral_long.transpose().matrix() * b_longitudinal_ineq.matrix()).transpose().array()
					 + pto_data.rho_vel * (pto_data.A_vel.transpose().matrix() * b_vx_ineq.matrix()).transpose().array()
					 + pto_data.rho_acc * (pto_data.A_acc.transpose().matrix() * b_ax_ineq.matrix()).transpose().array()
					 + pto_data.rho_jerk * (pto_data.A_jerk.transpose().matrix() * b_jx_ineq.matrix()).transpose().array()
					 + pto_data.rho_obs * (pto_data.A_obs.transpose().matrix() * b_obs_x.transpose().matrix()).transpose().array()
				     + (pto_data.A_consensus_x.transpose().matrix() * pto_data.lamda_consensus_x.matrix()).transpose().array()
					 + pto_data.rho_consensus_x * (pto_data.A_consensus_x.transpose().matrix() * pto_data.s_consensus_x.matrix()).transpose().array();
  
		lincost_y = pto_data.lamda_y 
					+ pto_data.weight_lane_tracking * (pto_data.A_tracking_lateral.transpose().matrix() * pto_data.y_des.matrix()).transpose().array()
					+ (pto_data.A_nonhol.transpose().matrix() * pto_data.lamda_psi.matrix()).transpose().array() // 11/50 50/5 ^T =           5*11
					+ (pto_data.A_obs.transpose().matrix() * pto_data.lamda_obsy.matrix()).transpose().array() // 11/50 50/5 ^T =           5*11
					+ pto_data.rho_psi * (pto_data.A_nonhol.transpose().matrix() * b_nonhol_y.transpose().matrix()).transpose().array()
					+ pto_data.rho_lateral_long * (pto_data.A_lateral_long.transpose().matrix() * b_lateral_ineq.matrix()).transpose().array()
					+ pto_data.rho_vel * (pto_data.A_vel.transpose().matrix() * b_vy_ineq.matrix()).transpose().array()
					+ pto_data.rho_acc * (pto_data.A_acc.transpose().matrix() * b_ay_ineq.matrix()).transpose().array()
					+ pto_data.rho_jerk * (pto_data.A_jerk.transpose().matrix() * b_jy_ineq.matrix()).transpose().array()
					+ pto_data.rho_obs * (pto_data.A_obs.transpose().matrix() * b_obs_y.transpose().matrix()).transpose().array()
					+ (pto_data.A_consensus_y.transpose().matrix() * pto_data.lamda_consensus_y.matrix()).transpose().array()
					+ pto_data.rho_consensus_y * (pto_data.A_consensus_y.transpose().matrix() * pto_data.s_consensus_y .matrix()).transpose().array();
 		 
		// start = clock(); 
		sol_x = (cost_mat_inv_x.matrix() * stack(lincost_x, b_eq_x, 'h').transpose().matrix()).transpose();
		primal_sol_x = sol_x.leftCols(pto_data.nvar);
		pto_data.x = (P.matrix() * primal_sol_x.transpose().matrix()).transpose();
    	pto_data.xdot = (Pdot.matrix() * primal_sol_x.transpose().matrix()).transpose();
		pto_data.xddot = (Pddot.matrix() * primal_sol_x.transpose().matrix()).transpose();
		pto_data.xdddot = (Pdddot.matrix() * primal_sol_x.transpose().matrix()).transpose();

		sol_y = (cost_mat_inv_y.matrix() * stack(lincost_y, b_eq_y, 'h').transpose().matrix()).transpose();
		primal_sol_y = sol_y.leftCols(pto_data.nvar);
		pto_data.y = (P.matrix() * primal_sol_y.transpose().matrix()).transpose();
    	pto_data.ydot = (Pdot.matrix() * primal_sol_y.transpose().matrix()).transpose();
		pto_data.yddot = (Pddot.matrix() * primal_sol_y.transpose().matrix()).transpose();
		pto_data.ydddot = (Pdddot.matrix() * primal_sol_y.transpose().matrix()).transpose();

		// end = clock();
		// cout<< "time2: "<< " "  << double(end - start) / double(CLOCKS_PER_SEC) << endl;
		s.a = primal_sol_x;
		s.b = primal_sol_y;
		s.c = pto_data;
		return s;
	}
	three_varr compute_psi(probData pto_data, ArrayXXf P, ArrayXXf Pdot, ArrayXXf Pddot, ArrayXXf psi_temp, ArrayXXf b_eq_psi) {
		three_varr s;
		ArrayXXf cost, cost_mat_inv, lincost_psi, sol_psi, primal_sol_psi, res_psi;
		MatrixXf cost_mat; 
 
		cost = pto_data.cost_smoothness_psi + pto_data.rho_psi * (pto_data.A_psi.transpose().matrix() * pto_data.A_psi.matrix()).array()
											+ pto_data.rho_consensus_psi*(pto_data.A_consensus_psi.transpose().matrix() * pto_data.A_consensus_psi.matrix()).array();	
		cost_mat = stack(stack(cost, pto_data.A_eq_psi.transpose(), 'h'), stack(pto_data.A_eq_psi, 0.0*ones(pto_data.A_eq_psi.rows(), pto_data.A_eq_psi.rows()), 'h'), 'v');
     // shap pto_data.cost_smoothness_psi 11/ 11
 		MatrixXf I(cost_mat.rows(), cost_mat.cols());
	    I.setIdentity();
	    cost_mat_inv = (cost_mat).householderQr().solve(I);//Q_{psi}^{-1}
  
		lincost_psi = (pto_data.A_psi.transpose().matrix() * pto_data.lamda_psi.matrix()).transpose().array()
					 + pto_data.rho_psi * (psi_temp.matrix()* pto_data.A_psi.matrix()).array()
					 + (pto_data.A_consensus_psi.transpose().matrix() * pto_data.lamda_consensus_psi.matrix()).transpose().array()
					 + pto_data.rho_consensus_psi * (pto_data.A_consensus_psi.transpose().matrix() * pto_data.s_consensus_psi.matrix()).transpose().array();

		// .lamda_psi  50* 5
		// "cost_mat_inv" 15 15
		// "Apsi"  50  11 
		// "psi_temp"  5 50
		sol_psi = (cost_mat_inv.matrix() * stack(lincost_psi, b_eq_psi, 'h').transpose().matrix()).transpose();
		primal_sol_psi = sol_psi.leftCols(pto_data.nvar);
	    // cout << "ttt" << endl;	

		// lincost_psi 5 11
		// b_eq_psi 5 4
	    // shape P.matrix()50	11
		// shape primal_sol_psi.matrix()5	11
		pto_data.psi = (P.matrix() * primal_sol_psi.transpose().matrix()).transpose();
		pto_data.psidot = (Pdot.matrix() * primal_sol_psi.transpose().matrix()).transpose();
		pto_data.psiddot = (Pddot.matrix() * primal_sol_psi.transpose().matrix()).transpose();
		//  5/50 
		res_psi = (pto_data.A_psi.matrix() * primal_sol_psi.transpose().matrix()).transpose().array() - psi_temp;
		// A_psi 50* 11
	    // cout << "222" << endl;	

		// pto_data.lamda_psi = pto_data.lamda_psi - pto_data.rho_psi * (pto_data.A_psi.transpose().matrix() * res_psi.transpose().matrix()).transpose().array();
		// pto_data.lamda_psi = pto_data.lamda_psi - pto_data.rho_psi * res_psi.transpose().matrix().array();
		// cout << "333" << endl;	
	
		s.a = primal_sol_psi;
		s.b = res_psi;
		s.c = pto_data;
		return s; 
	}
	
	probData OACP(probData pto_data, four_var PPP, ArrayXXf x_g, ArrayXXf y_g, float x_init, float y_init, float v_init, float ax_init,  float ay_init, float psi_init, float psidot_init, bool warm) {
		four_var s;
		// if(warm)
		{
			pto_data = initializeArrays(pto_data);
			pto_data.lamda_x = 0 * ones(pto_data.num_goal, pto_data.nvar);
			pto_data.lamda_y = 0 * ones(pto_data.num_goal, pto_data.nvar);
			pto_data.lamda_obsx = 0 * ones(pto_data.num_obs*pto_data.num, pto_data.num_goal);
			pto_data.lamda_obsy = 0 * ones(pto_data.num_obs*pto_data.num, pto_data.num_goal);  
			pto_data.lamda_psi = 0 * ones(pto_data.num, pto_data.num_goal);
			pto_data.lamda_consensus_x = 0 * ones(pto_data.num_consensus*3, pto_data.num_goal);
			pto_data.lamda_consensus_y = 0 * ones(pto_data.num_consensus*3, pto_data.num_goal);
			pto_data.lamda_consensus_psi = 0 * ones(pto_data.num_consensus, pto_data.num_goal);
			pto_data.d_a = pto_data.a_max * ones(pto_data.num_goal, pto_data.num);
 			pto_data.alpha_a = 0.0 * ones(pto_data.num_goal, pto_data.num);
		} 

		pto_data.x_init = x_init;
		pto_data.y_init = y_init;
		pto_data.x_fin = x_g;
		pto_data.y_fin = y_g;
		pto_data.ax_init = ax_init;
		pto_data.ax_init = ay_init;

		pto_data.psi_init = psi_init * ones(pto_data.num_goal, 1);
		pto_data.psi_fin = 0.0 * ones(pto_data.num_goal, 1);
		pto_data.psidot_init = psidot_init * ones(pto_data.num_goal, 1);
		pto_data.psidot_fin = 0.0 * ones(pto_data.num_goal, 1);
		pto_data.v_init = v_init * ones(pto_data.num_goal, 1);
 
		 
		pto_data.x_guess = ArrayXXf(pto_data.num_goal, pto_data.num);
		pto_data.y_guess = ArrayXXf(pto_data.num_goal, pto_data.num); 
		if (pto_data.x.rows() > 0) {
			// Warm start from column 0 to pto_data.num - 10
			pto_data.x_guess.block(0, 0, pto_data.num_goal, pto_data.num - 10) = pto_data.x.block(0, 1, pto_data.num_goal, pto_data.num - 10);
			pto_data.y_guess.block(0, 0, pto_data.num_goal, pto_data.num - 10) = pto_data.y.block(0, 1, pto_data.num_goal, pto_data.num - 10);
			// Fill the remaining states
			for (int i = 0; i < pto_data.x_guess.rows(); ++i) {
				pto_data.x_guess.row(i).segment(pto_data.num - 10, 10).setLinSpaced(10, pto_data.x_guess(i, pto_data.num - 11), x_g(i));
				pto_data.y_guess.row(i).segment(pto_data.num - 10, 10).setLinSpaced(10, pto_data.y_guess(i, pto_data.num - 11), y_g(i));
			}
			// cout << "warm start" << endl;
		} else {
			for (int i = 0; i < pto_data.x_guess.rows(); ++i) {
				pto_data.x_guess.row(i).setLinSpaced(pto_data.num, x_init, x_g(i));
				pto_data.y_guess.row(i).setLinSpaced(pto_data.num, y_init, y_g(i));
			}
			cout << " not warm start" << endl;
		}
 		// if(warm)	
		pto_data = initialize_guess_alpha(pto_data); 

		pto_data = solve(pto_data, PPP.a, PPP.b, PPP.c, PPP.d, pto_data.x_init, pto_data.x_fin, pto_data.y_init, pto_data.y_fin, pto_data.v_init, pto_data.ax_init, pto_data.ay_init,  pto_data.psi_init, pto_data.psidot_init, pto_data.psi_fin, 
					pto_data.psidot_fin, pto_data.x_obs, pto_data.y_obs, warm);		

		return pto_data;
	}
	probData initialize_guess_alpha(probData pto_data) {
		ArrayXXf c1_d, c2_d, d_temp;
		ArrayXXf wc_alpha(pto_data.num_goal, pto_data.num_obs*pto_data.num);
		ArrayXXf ws_alpha(pto_data.num_goal, pto_data.num_obs*pto_data.num);
		ArrayXXf randm, randm2(1, pto_data.num_obs * pto_data.num), randm3(1, pto_data.num_obs*pto_data.num);
        pto_data.gamma_matrices = ArrayXf::LinSpaced(pto_data.num, static_cast<float>(pto_data.gamma), 1.0f).replicate(pto_data.num_obs, pto_data.num_goal).transpose();
		// cout << "pto_data.gamma_matrices." <<  pto_data.gamma_matrices << endl;
		// cout<< "gamma_matrices rows(): " << pto_data.gamma_matrices.rows() << " cols()" << pto_data.gamma_matrices.cols() << endl;
		for(int i = 0; i < pto_data.num_goal; ++i) {	
			randm = (-pto_data.x_obs).rowwise() + pto_data.x_guess.row(i);
			randm2 = reshape(randm.transpose(), 1, pto_data.num_obs*pto_data.num);
			wc_alpha.row(i) = randm2; 
			randm = (-pto_data.y_obs).rowwise() + pto_data.y_guess.row(i);
			randm3 = reshape(randm.transpose(), 1, pto_data.num_obs*pto_data.num);
			ws_alpha.row(i) = randm3;
		} 
		// cout << " pto_data.a_obs_vec:" <<  pto_data.a_obs_vec << endl;
		// cout << " pto_data.b_obs_vec:" <<  pto_data.b_obs_vec << endl;

		pto_data.alpha_obs = arctan2(ws_alpha * pto_data.a_obs_vec, wc_alpha * pto_data.b_obs_vec);  
		// cout << "pto_data.alpha_obs " << " " << pto_data.alpha_obs << endl;
		// pto_data.alpha_obs = arctan2(ws_alpha * pto_data.a_obs, wc_alpha * pto_data.b_obs); 
		// cout << maximum(1, (wc_alpha + ws_alpha)/(pto_data.a_obs*cos(pto_data.alpha_obs) + pto_data.b_obs*sin(pto_data.alpha_obs))).matrix().lpNorm<2>();
 		// num_goal / num_obs*num_horizon
 		c1_d = pow(pto_data.a_obs_vec, 2) * pow(cos(pto_data.alpha_obs), 2) + pow(pto_data.b_obs_vec, 2) * pow(sin(pto_data.alpha_obs), 2);
		c2_d = pto_data.a_obs_vec * wc_alpha*cos(pto_data.alpha_obs) + pto_data.b_obs_vec * ws_alpha * sin(pto_data.alpha_obs);
		d_temp = c2_d/c1_d;
	    // num_goal / num_obs*num_horizon
		// pto_data.gamma_matrices = ArrayXXf(pto_data.num_goal, pto_data.num_obs * pto_data.num); 
		pto_data.d_obs = maximum(1 + 1e-20, d_temp);
		pto_data.d_obs_old = pto_data.d_obs;
  		return pto_data;
	} 
	four_var ComputeBernstein(ArrayXXf tot_time, float t_fin, int num) {
		four_var PPP;
		PPP = BernsteinCoeffOrder10(10.0, tot_time(0), t_fin, tot_time, num); 
		return PPP;
	}
	probData initializeArrays(probData pto_data) { 
		pto_data.alpha_obs = ArrayXXf(pto_data.num_goal, pto_data.num_obs * pto_data.num);
		pto_data.d_obs = ArrayXXf(pto_data.num_goal, pto_data.num_obs * pto_data.num);
		pto_data.d_obs_old = ArrayXXf(pto_data.num_goal, pto_data.num_obs * pto_data.num);
		pto_data.gamma_matrices = ArrayXXf(pto_data.num_goal, pto_data.num_obs * pto_data.num);
		pto_data.lamda_x = ArrayXXf(pto_data.num_goal, pto_data.nvar);
		pto_data.lamda_y = ArrayXXf(pto_data.num_goal, pto_data.nvar);
		pto_data.lamda_obsx = ArrayXXf(pto_data.num_obs*pto_data.num, pto_data.num_goal);
	    pto_data.lamda_obsy =ArrayXXf(pto_data.num_obs*pto_data.num, pto_data.num_goal); 
		pto_data.lamda_psi = ArrayXXf(pto_data.num, pto_data.num_goal);
		pto_data.lamda_consensus_psi = ArrayXXf(pto_data.num_consensus, pto_data.num_goal);
		pto_data.lamda_consensus_x = ArrayXXf(pto_data.num_consensus*2, pto_data.num_goal);
		pto_data.lamda_consensus_y = ArrayXXf(pto_data.num_consensus*2, pto_data.num_goal);
		pto_data.d_a = ArrayXXf(pto_data.num_goal, pto_data.num);
 		pto_data.alpha_a = ArrayXXf(pto_data.num_goal, pto_data.num);

		pto_data.ax_init = ArrayXXf(pto_data.num_goal, 1);		
		pto_data.ay_init = ArrayXXf(pto_data.num_goal, 1);
		pto_data.x_init = ArrayXXf(pto_data.num_goal, 1);
		pto_data.y_init = ArrayXXf(pto_data.num_goal, 1);
		pto_data.x_fin = ArrayXXf(pto_data.num_goal, 1);
		pto_data.y_fin = ArrayXXf(pto_data.num_goal, 1);
		pto_data.psi_init = ArrayXXf(pto_data.num_goal, 1);
		pto_data.psi_fin = ArrayXXf(pto_data.num_goal, 1);
		pto_data.psidot_init = ArrayXXf(pto_data.num_goal, 1);
		pto_data.psidot_fin = ArrayXXf(pto_data.num_goal, 1);
		pto_data.v_init = ArrayXXf(pto_data.num_goal, 1);

		return pto_data;
	}
}