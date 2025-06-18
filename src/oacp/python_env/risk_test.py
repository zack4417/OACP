# main.py
from risk_model import VehicleRiskModel  # Import the class from vehicle_risk_model.py

# Define model parameters
v_max = 10  # Maximum phantom vehicle velocity (in m/s)
T_pred = 4  # Prediction horizon (in seconds)
l_w = 3.75   # Lane width (in meters)
Z = 2        # Confidence interval factor for lateral deviation
s_s = -30     # Start of Phantom Vehicle Set (PVS)
s_e = -10      # End of Phantom Vehicle Set (PVS)

# Risk thresholds
c_th_min = 0.2
c_th_max = 50.0
v_occ_min = 1  # Ego agent: Minimum velocity allowed (m/s)
v_occ_max = 6    # Ego agent: Maximum velocity allowed (m/s)

# Initialize the risk model
risk_model = VehicleRiskModel(v_max, T_pred, l_w, Z, s_s, s_e, c_th_min, c_th_max, v_occ_min, v_occ_max)

# Calculate total risk for a sample position and lateral deviation
s_ego = 0  # Longitudinal position of the ego vehicle
d_sv = 0   # maximum Lateral deviation of the SVs 
  
# Example usage 

# ego_pos = (0, 0)  # (xe, ye)
# building_obstacles_pos_start = (-2, -10)  # (xo_start, yo_start)
# building_obstacles_pos_end = (-2, -2)  # (xo_end, yo_end)
# vehicle_obstacles_pos_start = (-2, -14)  # (xo_start, yo_start)
# vehicle_obstacles_pos_end = (-2, -12)  # (xo_end, yo_end
 
ego_pos = (-2, 0)  # (xe, ye)
building_obstacles_pos_start = (0, -8)  # (xo_start, yo_start)
building_obstacles_pos_end = (0, -2)  # (xo_end, yo_end)
vehicle_obstacles_pos_start = (0, -14)  # (xo_start, yo_start)
vehicle_obstacles_pos_end = (0, -12)  # (xo_end, yo_end)
x_lane = 3.75  # Target lane x position

# Call the function to calculate the occlusion area
y_lane_start_1, y_lane_end_1 = risk_model.calculate_occlusion_area(ego_pos[0], ego_pos[1], building_obstacles_pos_start[0], building_obstacles_pos_start[1], building_obstacles_pos_end[0], building_obstacles_pos_end[1], x_lane)
print(f"Occlusion area by buildings on lane: y_start = {y_lane_start_1}, y_end = {y_lane_end_1}")
r_total_building = risk_model.total_risk(s_ego, d_sv, y_lane_start_1, y_lane_end_1)
risk_model.plot_risk_map('building') 
# Call the function to calculate the occlusion area
y_lane_start_2, y_lane_end_2 = risk_model.calculate_occlusion_area(ego_pos[0], ego_pos[1], building_obstacles_pos_start[0], vehicle_obstacles_pos_start[1], vehicle_obstacles_pos_end[0], vehicle_obstacles_pos_end[1], x_lane)
print(f"Occlusion area by vehicles on lane: y_start = {y_lane_start_2}, y_end = {y_lane_end_2}")
r_total_vehicle = risk_model.total_risk(s_ego, d_sv, y_lane_start_2, y_lane_end_2)
risk_model.plot_risk_map('vehicle')  
   
r_total = r_total_building + r_total_vehicle
v_occ_s = risk_model.dynamic_velocity_boundary(r_total)

print(f"Total Risk: {r_total}, r_total_building : {r_total_building} r_total_vehicle : {r_total_vehicle}")
print(f"Safe Velocity Boundary: {v_occ_s} m/s")




 