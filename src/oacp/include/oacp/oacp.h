// Contact Information:
// Name: Lei Zheng
// Email: zack44170625@gmail.com
// Affiliation: HKUST/CMU

#pragma once

#include <eigen3/Eigen/Dense>
#include <random> // For random number generation (e.g., generate_random_threshold)

using namespace Eigen;

namespace optim {

    // --- Data Structures ---

    /**
     * @brief Structure to hold all problem-specific data and parameters for optimization.
     *
     * This structure encapsulates various parameters, matrices, and vectors
     * used throughout the optimization process, including problem dimensions,
     * weights, initial/final conditions, obstacle information, and ADMM-related
     * variables.
     */
    struct probData {
        // Problem Dimensions and Iteration Limits
        int num_goal;          // Number of goals/targets
        int maxiter;           // Maximum number of iterations for optimization
        int num;               // General purpose number (e.g., number of time steps)
        int nvar;              // Number of optimization variables
        int num_obs;           // Number of obstacles
        int num_consensus;     // Number of consensus variables (general)
        int num_consensus_psi; // Number of consensus variables for psi (heading)
        int kappa;             // Some scaling factor or iteration count

        // Optimization Parameters and Weights
        float gamma;               // Step size or penalty parameter
        float alpha_admm;          // Alpha parameter for ADMM
        ArrayXXf gamma_matrices;   // Matrices related to gamma
        ArrayXXf a_obs_vec;        // Safety coefficients vector
        ArrayXXf b_obs_vec;        // Safety coefficients vector
        float weight_smoothness;       // Weight for trajectory smoothness
        float weight_lane_tracking;    // Weight for lane tracking
        float weight_vel_tracking;     // Weight for velocity tracking
        float weight_v;                // Weight for velocity magnitude
        float rho_vel;                 // Penalty parameter for velocity constraint
        float rho_acc;                 // Penalty parameter for acceleration constraint
        float rho_jerk;                // Penalty parameter for jerk constraint
        float rho_lateral_long;        // Penalty parameter for lateral/longitudinal coupling
        float rho_obs;                 // Penalty parameter for obstacle avoidance
        float rho_consensus_psi;       // Penalty parameter for psi consensus
        float rho_consensus_x;         // Penalty parameter for x consensus
        float rho_consensus_y;         // Penalty parameter for y consensus
        float t_fin;                   // Final time
        float t;                       // Current time
        float rho_ineq;                // Penalty parameter for general inequalities
        float rho_psi;                 // Penalty parameter for psi (heading)
        float rho_nonhol;              // Penalty parameter for nonholonomic constraints
        float weight_smoothness_psi;   // Weight for psi smoothness

        // Physical and Kinematic Constraints
        float a_obs;                   // Obstacle 'a' coefficient (scalar)
        float b_obs;                   // Obstacle 'b' coefficient (scalar)
        float longitudinal_min;        // Minimum longitudinal position
        float longitudinal_max;        // Maximum longitudinal position
        float lateral_min;             // Minimum lateral position
        float lateral_max;             // Maximum lateral position
        float a_max;                   // Maximum acceleration
        float v_max;                   // Maximum velocity
        float vx_max, vx_min;          // Max/min longitudinal velocity
        float vxc_max, vxc1_max, vxc_min; // Specific longitudinal velocity constraints
        float vy_max, vy_min;          // Max/min lateral velocity
        float ay_max, ay_min;          // Max/min lateral acceleration
        float ax_max, ax_min;          // Max/min longitudinal acceleration
        float jx_max, jy_max;          // Max longitudinal/lateral jerk

        // Obstacle and Reference Data
        ArrayXXf x_obs_fin, y_obs_fin; // Final obstacle positions
        ArrayXXf x_obs, y_obs;         // Current obstacle positions
        ArrayXXf alpha_obs;            // Alpha variables for obstacle avoidance
        ArrayXXf d_obs, d_obs_old;     // Distance to obstacles (current and old)

        // Initial and Final Conditions (Trajectory)
        ArrayXXf x_init, y_init;             // Initial x, y positions
        ArrayXXf vx_init, vy_init;           // Initial x, y velocities
        ArrayXXf ax_init, ay_init;           // Initial x, y accelerations
        ArrayXXf psi_init, psidot_init;      // Initial heading and heading rate
        ArrayXXf v_init;                     // Initial velocity magnitude
        ArrayXXf x_fin, y_fin;               // Final x, y positions
        ArrayXXf vx_fin, vy_fin;             // Final x, y velocities
        ArrayXXf ax_fin, ay_fin;             // Final x, y accelerations
        ArrayXXf psi_fin, psidot_fin;        // Final heading and heading rate

        // Desired Trajectory/Reference Data
        ArrayXXf y_des;     // Desired lateral position
        ArrayXXf vx_des;    // Desired longitudinal velocity
        ArrayXXf v_ref;     // Reference velocity profile
        ArrayXXf x_ref, y_ref; // Reference x, y positions

        // Cost Components
        ArrayXXf cost_smoothness;      // Cost for trajectory smoothness
        ArrayXXf cost_smoothness_psi;  // Cost for psi smoothness
        ArrayXXf cost_tracking_lateral;// Cost for lateral tracking
        ArrayXXf cost_tracking_vel;    // Cost for velocity tracking
        ArrayXXf lincost_smoothness_psi; // Linear cost for psi smoothness

        // Guess/Warm Start Data
        ArrayXXf x_guess, y_guess; // Initial guess for x, y trajectories

        // ADMM Variables: Lagrangian Multipliers (lambda) and Slack Variables (s)
        ArrayXXf lamda_x, lamda_y;              // Lagrangian multipliers for x, y
        ArrayXXf lamda_obsx, lamda_obsy;        // Lagrangian multipliers for obstacle x, y
        ArrayXXf lamda_consensus_psi;           // Lagrangian multipliers for psi consensus
        ArrayXXf lamda_consensus_x;             // Lagrangian multipliers for x consensus
        ArrayXXf lamda_consensus_y;             // Lagrangian multipliers for y consensus
        ArrayXXf lamda_psi;                     // Lagrangian multipliers for psi

        ArrayXXf s_x_ineq, s_y_ineq;             // Slack variables for x, y inequalities
        ArrayXXf s_vx_ineq, s_vy_ineq;           // Slack variables for vx, vy inequalities
        ArrayXXf s_ax_ineq, s_ay_ineq;           // Slack variables for ax, ay inequalities
        ArrayXXf s_jx_ineq, s_jy_ineq;           // Slack variables for jx, jy inequalities

        ArrayXXf s_x_ineq_old, s_y_ineq_old;             // Old slack variables for x, y inequalities
        ArrayXXf s_vx_ineq_old, s_vy_ineq_old;           // Old slack variables for vx, vy inequalities
        ArrayXXf s_ax_ineq_old, s_ay_ineq_old;           // Old slack variables for ax, ay inequalities
        ArrayXXf s_jx_ineq_old, s_jy_ineq_old;           // Old slack variables for jx, jy inequalities

        ArrayXXf d_lateral, d_v, d_a;           // Auxiliary variables for lateral, velocity, acceleration

        // Trajectory Variables (outputs of optimization)
        ArrayXXf x, y;             // Optimized x, y positions
        ArrayXXf xdot, ydot;       // Optimized x, y velocities
        ArrayXXf xddot, yddot;     // Optimized x, y accelerations
        ArrayXXf xdddot, ydddot;   // Optimized x, y jerk
        ArrayXXf psi, psidot, psiddot; // Optimized heading, heading rate, heading acceleration
        ArrayXXf v;                // Optimized velocity magnitude
        ArrayXXf alpha_lateral, alpha_v, alpha_a; // Alpha variables for various constraints

        // Constraint Residuals
        ArrayXXf res_obs;      // Residual for obstacle avoidance
        ArrayXXf res_vel;      // Residual for velocity constraints
        ArrayXXf res_acc;      // Residual for acceleration constraints
        ArrayXXf res_nonhol;   // Residual for nonholonomic constraints
        ArrayXXf res_eq;       // Residual for equality constraints

        // System Matrices (for constraints)
        ArrayXXf A_eq_x, A_eq_y;         // Equality constraint matrices for x, y
        ArrayXXf B_x_eq, B_y_eq;         // Equality constraint vectors for x, y
        ArrayXXf A_vel, A_acc;           // Velocity and acceleration constraint matrices
        ArrayXXf A_lateral_long;         // Lateral-longitudinal coupling constraint matrix
        ArrayXXf A_jerk;                 // Jerk constraint matrix
        ArrayXXf A_psi;                  // Psi constraint matrix
        ArrayXXf A_nonhol;               // Nonholonomic constraint matrix
        ArrayXXf A_eq_psi;               // Equality constraint matrix for psi
        ArrayXXf A_obs;                  // Obstacle avoidance constraint matrix
        ArrayXXf A_ineq;                 // General inequality constraint matrix
        ArrayXXf b_longitudinal_ineq, b_lateral_ineq; // Longitudinal/lateral inequality vectors
        ArrayXXf b_vx_ineq, b_vy_ineq;   // Vx, Vy inequality vectors
        ArrayXXf b_ax_ineq, b_ay_ineq;   // Ax, Ay inequality vectors
        ArrayXXf b_jx_ineq, b_jy_ineq;   // Jx, Jy inequality vectors

        // Consensus Variables and Matrices
        ArrayXXf A_consensus_psi, s_consensus_psi; // Psi consensus matrix and slack
        ArrayXXf A_consensus_x, s_consensus_x;     // X consensus matrix and slack
        ArrayXXf A_consensus_y, s_consensus_y;     // Y consensus matrix and slack
        ArrayXXf consensus_actual_psi, consensus_actual_x, consensus_actual_y; // Actual consensus values

        // Tracking and Control Related
        ArrayXXf A_tracking_lateral, A_tracking_vel; // Tracking matrices
        ArrayXXf Pdot_upsample;                      // Upsampled Pdot (e.g., for controls)
        ArrayXXf v_controls, w_controls;             // Velocity and angular velocity controls
    };

    /**
     * @brief Structure to hold four ArrayXXf variables.
     */
    struct four_var {
        ArrayXXf a, b, c, d;
    };

    /**
     * @brief Structure to hold three ArrayXXf variables.
     */
    struct three_var {
        ArrayXXf a, b, c;
    };

    /**
     * @brief Structure to hold two ArrayXXf variables and a probData object.
     *
     * Note: Having a probData object inside another struct like this might
     * lead to large data duplication if not handled carefully (e.g., by
     * passing by reference or pointer in functions). Re-evaluate if this
     * structure is truly necessary or if a different approach (e.g., returning
     * `probData` directly and individual `ArrayXXf` values) would be better.
     */
    struct three_varr {
        ArrayXXf a, b;
        probData c; // Consider if this should be a pointer or reference to avoid copying
    };

    /**
     * @brief Structure to hold two ArrayXXf variables.
     */
    struct two_var {
        ArrayXXf a, b;
    };

    // --- Utility Functions ---

    /**
     * @brief Prints the shape (dimensions) of an Eigen ArrayXXf.
     * @param arr The ArrayXXf to inspect.
     */
    void shape(ArrayXXf arr);

    /**
     * @brief Computes the Euclidean distance between two 2D points.
     * @param x1 X-coordinate of the first point.
     * @param y1 Y-coordinate of the first point.
     * @param x2 X-coordinate of the second point.
     * @param y2 Y-coordinate of the second point.
     * @return The Euclidean distance.
     */
    float euclidean_dist(float x1, float y1, float x2, float y2);

    /**
     * @brief Computes a safety distance between two 2D points.
     * (The implementation details for "safety" need to be defined
     * within the function body).
     * @param x1 X-coordinate of the first point.
     * @param y1 Y-coordinate of the first point.
     * @param x2 X-coordinate of the second point.
     * @param y2 Y-coordinate of the second point.
     * @return The calculated safety distance.
     */
    float safety_dist(float x1, float y1, float x2, float y2);

    /**
     * @brief Creates an Eigen ArrayXXf filled with ones.
     * @param row The number of rows.
     * @param col The number of columns.
     * @return An ArrayXXf of specified dimensions filled with ones.
     */
    ArrayXXf ones(int row, int col);

    /**
     * @brief Reshapes an Eigen ArrayXXf to new dimensions.
     * @param x The ArrayXXf to reshape.
     * @param r The new number of rows.
     * @param c The new number of columns.
     * @return The reshaped ArrayXXf.
     */
    ArrayXXf reshape(ArrayXXf x, uint32_t r, uint32_t c);

    /**
     * @brief Clips a single double value between a minimum and maximum.
     * @param min The minimum allowed value.
     * @param max The maximum allowed value.
     * @param number The value to clip.
     * @return The clipped value.
     */
    double clip3(double min, double max, double number);

    /**
     * @brief Clips an Eigen ArrayXXf element-wise between corresponding min and max arrays.
     * @param min An ArrayXXf containing the minimum limits for each element.
     * @param max An ArrayXXf containing the maximum limits for each element.
     * @param arr The ArrayXXf to clip.
     * @return The clipped ArrayXXf.
     */
    ArrayXXf clip2(ArrayXXf min, ArrayXXf max, ArrayXXf arr);

    /**
     * @brief Clips an Eigen ArrayXXf element-wise between a scalar minimum and maximum.
     * @param min The scalar minimum allowed value.
     * @param max The scalar maximum allowed value.
     * @param arr The ArrayXXf to clip.
     * @return The clipped ArrayXXf.
     */
    ArrayXXf clip(float min, float max, ArrayXXf arr);

    /**
     * @brief Computes the difference between consecutive elements of an ArrayXXf.
     * (e.g., [arr[1]-arr[0], arr[2]-arr[1], ...])
     * @param arr The ArrayXXf to differentiate.
     * @return An ArrayXXf containing the differences.
     */
    ArrayXXf diff(ArrayXXf arr);

    /**
     * @brief Computes the element-wise maximum between a scalar and an ArrayXXf.
     * @param val The scalar value.
     * @param arr2 The ArrayXXf.
     * @return An ArrayXXf where each element is the maximum of 'val' and the corresponding element in 'arr2'.
     */
    ArrayXXf maximum(float val, ArrayXXf arr2);

    /**
     * @brief Computes the element-wise minimum between a scalar and an ArrayXXf.
     * @param val The scalar value.
     * @param arr2 The ArrayXXf.
     * @return An ArrayXXf where each element is the minimum of 'val' and the corresponding element in 'arr2'.
     */
    ArrayXXf minimum(float val, ArrayXXf arr2);

    /**
     * @brief Generates an array of evenly spaced numbers over a specified interval.
     * @param t_init The starting value of the sequence.
     * @param t_end The end value of the sequence.
     * @param steps The number of samples to generate.
     * @return An ArrayXXf containing the linearly spaced values.
     */
    ArrayXXf linspace(float t_init, float t_end, float steps);

    /**
     * @brief Computes the binomial coefficient "n choose k".
     * @param n The total number of items.
     * @param k The number of items to choose.
     * @return The binomial coefficient.
     */
    float binomialCoeff(float n, float k);

    /**
     * @brief Generates a random number from a Gaussian (normal) distribution.
     * @param mean The mean of the distribution.
     * @param stddev The standard deviation of the distribution.
     * @return A random number sampled from the specified Gaussian distribution.
     */
    double generate_random_threshold(double mean, double stddev);

    // /**
    //  * @brief Computes a coefficient rho based on primal and dual residuals.
    //  * (Commented out as it was commented in the original, but useful for ADMM adaptive rho)
    //  * @param primal_residual The primal residual.
    //  * @param dual_residual The dual residual.
    //  * @param tau_increase Factor to increase rho if primal residual is too high.
    //  * @param tau_decrease Factor to decrease rho if dual residual is too high.
    //  * @return The computed rho coefficient.
    //  */
    // double compute_coefficient_rho(double primal_residual, double dual_residual, double tau_increase = 2.0, double tau_decrease = 2.0);

    /**
     * @brief Computes the element-wise arctangent of y/x using two ArrayXXf.
     * (Similar to `std::atan2`)
     * @param arr1 The 'y' values.
     * @param arr2 The 'x' values.
     * @return An ArrayXXf with the arctangent results.
     */
    ArrayXXf arctan2(ArrayXXf arr1, ArrayXXf arr2);

    /**
     * @brief Computes the cumulative sum of an ArrayXXf, with an initial value.
     * (The second parameter 'arr2' might indicate an initial sum or
     * could be a typo and should be part of the function's internal logic
     * or a single scalar initial value.)
     * @param arr1 The ArrayXXf to sum.
     * @param arr2 The initial value for the cumulative sum (or another array to sum with).
     * @return An ArrayXXf with the cumulative sums.
     */
    ArrayXXf cumsum(ArrayXXf arr1, ArrayXXf arr2);

    /**
     * @brief Stacks two ArrayXXf horizontally or vertically based on a character flag.
     * @param arr1 The first ArrayXXf.
     * @param arr2 The second ArrayXXf.
     * @param ch 'h' for horizontal stack, 'v' for vertical stack.
     * @return The stacked ArrayXXf.
     */
    ArrayXXf stack(ArrayXXf arr1, ArrayXXf arr2, char ch);

    /**
     * @brief Stacks three ArrayXXf objects vertically.
     * @param a The first ArrayXXf.
     * @param b The second ArrayXXf.
     * @param c The third ArrayXXf.
     * @return The vertically stacked ArrayXXf.
     */
    ArrayXXf stackVertically3(const ArrayXXf& a, const ArrayXXf& b, const ArrayXXf& c);

    /**
     * @brief Stacks four ArrayXXf objects vertically.
     * @param a The first ArrayXXf.
     * @param b The second ArrayXXf.
     * @param c The third ArrayXXf.
     * @param d The fourth ArrayXXf.
     * @return The vertically stacked ArrayXXf.
     */
    
    three_var bernstein_coeff_order10_new(float n, float tmin, float tmax, ArrayXXf t_actual, int num);

    /**
     * @brief Computes Bernstein coefficients of order 10.
     * @param n The degree of the polynomial.
     * @param tmin The minimum time.
     * @param tmax The maximum time.
     * @param t_actual An ArrayXXf of actual time points.
     * @param num A general purpose number (e.g., number of points).
     * @return A `four_var` struct containing the coefficients.
     */
    four_var BernsteinCoeffOrder10(float n, float tmin, float tmax, ArrayXXf t_actual, int num);

    /**
     * @brief Computes Bernstein coefficients of order 8.
     * @param n The degree of the polynomial.
     * @param tmin The minimum time.
     * @param tmax The maximum time.
     * @param t_actual An ArrayXXf of actual time points.
     * @param num A general purpose number (e.g., number of points).
     * @return A `three_var` struct containing the coefficients.
     */
    three_var bernstein_coeff_order8(float n, float tmin, float tmax, ArrayXXf t_actual, int num);

    /**
     * @brief Computes Bernstein coefficients of order 12.
     * @param tmin The minimum time.
     * @param tmax The maximum time.
     * @param t_actual An ArrayXXf of actual time points.
     * @param num A general purpose number (e.g., number of points).
     * @return A `three_var` struct containing the coefficients.
     */
    three_var bernstein_coeff_order12(float tmin, float tmax, ArrayXXf t_actual, int num);

    /**
     * @brief Computes Bernstein polynomial related values given total time, final time, and number of points.
     * @param tot_time An ArrayXXf representing total time points.
     * @param t_fin The final time.
     * @param num A general purpose number (e.g., number of points).
     * @return A `three_var` struct containing computed Bernstein values.
     */
    three_var compute_bernstein(ArrayXXf tot_time, float t_fin, int num);

    /**
     * @brief Computes Bernstein polynomial related values (another version, potentially with more outputs).
     * @param tot_time An ArrayXXf representing total time points.
     * @param t_fin The final time.
     * @param num A general purpose number (e.g., number of points).
     * @return A `four_var` struct containing computed Bernstein values.
     */
    four_var ComputeBernstein(ArrayXXf tot_time, float t_fin, int num);

    // --- Optimization and OACP Functions ---

    /**
     * @brief Solves the optimization problem.
     * @param pto_data The problem data structure.
     * @param P Bernstein basis matrix (position).
     * @param Pdot Bernstein basis matrix (velocity).
     * @param Pddot Bernstein basis matrix (acceleration).
     * @param Pddddddot Bernstein basis matrix (jerk or higher derivative).
     * @param x_init Initial x positions.
     * @param x_fin Final x positions.
     * @param y_init Initial y positions.
     * @param y_fin Final y positions.
     * @param v_init Initial velocities.
     * @param ax_init Initial x accelerations.
     * @param ay_init Initial y accelerations.
     * @param psi_init Initial heading angles.
     * @param psidot_init Initial heading rates.
     * @param psi_fin Final heading angles.
     * @param psidot_fin Final heading rates.
     * @param x_obs Obstacle x coordinates.
     * @param y_obs Obstacle y coordinates.
     * @param warm Flag indicating whether to use a warm start.
     * @return The updated `probData` structure after solving.
     */
    probData solve(probData pto_data, ArrayXXf P, ArrayXXf Pdot, ArrayXXf Pddot, ArrayXXf Pddddddot,
                   ArrayXXf x_init, ArrayXXf x_fin, ArrayXXf y_init, ArrayXXf y_fin,
                   ArrayXXf v_init, ArrayXXf ax_init, ArrayXXf ay_init,
                   ArrayXXf psi_init, ArrayXXf psidot_init, ArrayXXf psi_fin, ArrayXXf psidot_fin,
                   ArrayXXf x_obs, ArrayXXf y_obs, bool warm);

    /**
     * @brief Computes x and y trajectories.
     * @param pto_data The problem data structure.
     * @param P Bernstein basis matrix (position).
     * @param Pdot Bernstein basis matrix (velocity).
     * @param Pddot Bernstein basis matrix (acceleration).
     * @param Pdddot Bernstein basis matrix (jerk).
     * @param b_eq_x Equality constraint vector for x.
     * @param b_eq_y Equality constraint vector for y.
     * @param x_obs Obstacle x coordinates.
     * @param y_obs Obstacle y coordinates.
     * @return A `three_varr` struct containing computed x, y, and updated problem data.
     */
    three_varr compute_xy(probData pto_data, ArrayXXf P, ArrayXXf Pdot, ArrayXXf Pddot,
                          ArrayXXf Pdddot, ArrayXXf b_eq_x, ArrayXXf b_eq_y,
                          ArrayXXf x_obs, ArrayXXf y_obs);

    /**
     * @brief Computes psi (heading) trajectory.
     * @param pto_data The problem data structure.
     * @param P Bernstein basis matrix (position).
     * @param Pdot Bernstein basis matrix (velocity).
     * @param Pddot Bernstein basis matrix (acceleration).
     * @param psi_temp Temporary psi values.
     * @param b_eq_psi Equality constraint vector for psi.
     * @return A `three_varr` struct containing computed psi, psidot, and updated problem data.
     */
    three_varr compute_psi(probData pto_data, ArrayXXf P, ArrayXXf Pdot, ArrayXXf Pddot,
                           ArrayXXf psi_temp, ArrayXXf b_eq_psi);

    /**
     * @brief Implements the OACP loop.
     * @param pto_data The problem data structure.
     * @param PPP A `four_var` struct likely containing Bernstein matrices (P, Pdot, Pddot, Pdddot).
     * @param x_g Goal x positions.
     * @param y_g Goal y positions.
     * @param x_init Current initial x position.
     * @param y_init Current initial y position.
     * @param v_init Current initial velocity magnitude.
     * @param ax_init Current initial x acceleration.
     * @param ay_init Current initial y acceleration.
     * @param psi_init Current initial heading angle.
     * @param psidot_init Current initial heading rate.
     * @param warm Flag indicating whether to use a warm start.
     * @return The updated `probData` structure after one OACP step.
     */
    probData OACP(probData pto_data, four_var PPP, ArrayXXf x_g, ArrayXXf y_g,
                 float x_init, float y_init, float v_init, float ax_init, float ay_init,
                 float psi_init, float psidot_init, bool warm);

    /**
     * @brief Initializes the guess for alpha variables (e.g., for ADMM).
     * @param pto_data The problem data structure.
     * @return The updated `probData` structure with initialized alpha guesses.
     */
    probData initialize_guess_alpha(probData pto_data);

    /**
     * @brief Initializes all ArrayXXf members within the `probData` structure to appropriate sizes.
     * This is crucial before using them to prevent errors.
     * @param pto_data The problem data structure (passed by value to allow modification and return).
     * @return The initialized `probData` structure.
     */
    probData initializeArrays(probData pto_data);

} // namespace optim