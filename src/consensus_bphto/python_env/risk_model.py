# vehicle_risk_model.py
# Contact: Lei Zheng - zack44170625@gmail.com
# Vehicle Risk Assessment Model for Autonomous Driving Systems
# Implements longitudinal, lateral, and occlusion risk modeling
# Visualization tools for risk assessment and dynamic velocity boundaries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


class VehicleRiskModel:
    def __init__(self, v_max, T_pred, l_w, s_s, s_e, Z, 
                 c_th_min, c_th_max, v_occ_min, v_occ_max):
        """
        Initialize vehicle risk model parameters.
        
        Parameters:
        v_max (float): Maximum vehicle velocity (m/s)
        T_pred (float): Prediction time horizon (s)
        l_w (float): Lane width (m)
        s_s (float): Start position of Phantom Vehicle Set (PVS) (m)
        s_e (float): End position of Phantom Vehicle Set (PVS) (m)
        Z (float): Height parameter for occlusion calculation
        c_th_min (float): Minimum risk threshold for velocity boundary
        c_th_max (float): Maximum risk threshold for velocity boundary
        v_occ_min (float): Minimum occupancy velocity (m/s)
        v_occ_max (float): Maximum occupancy velocity (m/s)
        """
        self.v_max = v_max
        self.T_pred = T_pred
        self.l_w = l_w
        self.Z = Z
        self.s_s = s_s
        self.s_e = s_e
        self.c_th_min = c_th_min
        self.c_th_max = c_th_max
        self.v_occ_min = v_occ_min
        self.v_occ_max = v_occ_max
        self.r_grid_vehicles = None
        self.r_grid_buildings = None

    def calculate_occlusion_area(self, xe, ye, xo_start, yo_start, 
                                 xo_end, yo_end, x_lane):
        """
        Calculate occlusion area boundaries using similar triangles.
        
        Returns:
        tuple: (y_lane_start, y_lane_end) occlusion boundaries
        """
        y_lane_start = ye - ((xe - x_lane) * (ye - yo_start)) / (xe - xo_start)
        y_lane_end = ye - ((xe - x_lane) * (ye - yo_end)) / (xe - xo_end)
        return y_lane_start, y_lane_end

    def longitudinal_risk(self, s):
        """Calculate longitudinal risk g(s) for position s in Phantom Vehicle Set."""
        # Auto-adjust PVS bounds if invalid
        if self.s_e - self.s_s > self.v_max * self.T_pred:
            self.s_e = self.s_s + self.v_max * self.T_pred
            print("Warning: Clipped s_end to maintain PVS validity")

        # Calculate risk based on position zones
        if self.s_s <= s < self.s_e:
            return 0.5 * (2 * self.v_max - (s - self.s_s) / self.T_pred) * (s - self.s_s)
        elif self.s_e <= s < self.s_s + self.v_max * self.T_pred:
            return 0.5 * (2 * self.v_max - (s - self.s_s) / self.T_pred - 
                         (s - self.s_e) / self.T_pred) * (self.s_e - self.s_s)
        elif self.s_s + self.v_max * self.T_pred <= s < self.s_e + self.v_max * self.T_pred:
            return 0.5 * (self.v_max - (s - self.s_e) / self.T_pred) * \
                (self.s_e - (s - self.v_max * self.T_pred))
        return 0  # Outside risk zones

    def lateral_risk(self, d):
        """Calculate lateral risk w(d) using normal distribution."""
        sigma = self.l_w / (2 * 1.645)  # 90% confidence interval
        return norm.pdf(d, 0, sigma)

    def total_risk(self, s, d, s_s, s_e):
        """Calculate combined risk r(s, d) = g(s) * w(d)."""
        self.s_s, self.s_e = s_s, s_e
        return self.longitudinal_risk(s) * self.lateral_risk(d)

    def dynamic_velocity_boundary(self, r_total):
        """Compute velocity boundary based on risk level."""
        if self.c_th_min <= r_total <= self.c_th_max:
            return self.v_occ_max + (self.v_occ_min - self.v_occ_max) * \
                   (r_total - self.c_th_min) / (self.c_th_max - self.c_th_min)
        return self.v_occ_min if r_total > self.c_th_max else self.v_occ_max

    def plot_risk_map(self, risk_type='vehicle', save_path=None):
        """
        Generate and display risk map.
        
        Parameters:
        risk_type (str): 'vehicle' or 'building' risk visualization
        save_path (str): Optional path to save the figure
        """
        # Create position grids
        s_values = np.linspace(self.s_s, self.s_e + self.v_max * self.T_pred, 100)
        d_values = np.linspace(-self.l_w/2, self.l_w/2, 100)
        s_grid, d_grid = np.meshgrid(s_values, d_values)
        
        # Calculate risk grid
        risk_grid = np.vectorize(self.total_risk)(s_grid, d_grid, self.s_s, self.s_e)
        
        # Store based on risk type
        if risk_type == 'building':
            self.r_grid_buildings = risk_grid
            title = "Building Occlusion Risk Map"
            filename = save_path or 'building_risk_map.png'
        else:  # Default to vehicle risk
            self.r_grid_vehicles = risk_grid
            title = "Vehicle Collision Risk Map"
            filename = save_path or 'vehicle_risk_map.png'

        # Create plot
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(s_grid, d_grid, risk_grid, levels=50, cmap='jet')
        
        plt.colorbar(contour).set_label('Risk Level', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlabel('Longitudinal Position (m)', fontsize=12)
        plt.ylabel('Lateral Deviation (m)', fontsize=12)
        plt.axhline(0, color='w', linestyle='--', linewidth=1.5, label='Lane Center')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved risk map to {filename}")
        plt.show()
    # def plot_risk_map(self, risk_type='vehicle'):
    #         """Generates the risk map based on longitudinal position and lateral deviation."""
    #         s_values = np.linspace(self.s_s, self.s_e + self.v_max * self.T_pred, 100)  # Longitudinal positions
    #         d_values = np.linspace(-self.l_w / 2, self.l_w / 2, 100)  # Lateral deviations
    #         s_grid, d_grid = np.meshgrid(s_values, d_values)

    #         if risk_type == 'building':
    #             self.r_grid_buildings = np.vectorize(self.total_risk)(s_grid, d_grid, self.s_s, self.s_e)
    #             plt.figure(figsize=(10, 4))
    #             # contour = plt.contourf(s_grid, d_grid, self.r_grid_buildings, levels=35, cmap='jet')

    #             # Add vmin/vmax to constrain the color range
    #             contour = plt.contourf(s_grid, d_grid, self.r_grid_buildings, 
    #                       levels=30, cmap='jet',  alpha=0.8, 
    #                       vmin=np.percentile(self.r_grid_buildings, 20),  # 5th percentile
    #                       vmax=np.percentile(self.r_grid_buildings, 100))  # 95th percentile

    #             # For better perception of critical values
    #             # contour = plt.contourf(s_grid, d_grid, self.r_grid_buildings, 
    #             #                     levels=30, cmap='coolwarm')  # Other options: 'RdBu', 'PiYG', 'PRGn'
    #             # desaturated_jet = LinearSegmentedColormap.from_list('coolwarm', plt.cm.jet(np.linspace(0, 1, 256)) * 0.85 + 0.15)  # Reduce saturation by 30%
    #             # contour = plt.contourf(s_grid, d_grid, self.r_grid_buildings, 
    #             #                     levels=30, cmap=desaturated_jet)

    #             # Create colorbar and adjust its properties
    #             cbar = plt.colorbar(contour)
    #             cbar.set_label('Risk Value', fontsize=14)  # Adjust colorbar label font size
    #             cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick label font size

    #             plt.xlabel('Longitudinal Position (m)', fontsize=14)
    #             plt.ylabel('Lateral Deviation (m)', fontsize=14)
    #             plt.xticks(fontsize=12)
    #             plt.yticks(fontsize=12)
    #             # plt.ylim(-1.875, 1.875)
    #             # plt.xlim(-15, 15)
    #             plt.axhline(0, color='white', linestyle='--', linewidth=2, label='Lane Center')
    #             plt.legend(prop={'size': 12})
    #             plt.savefig('risk_map.png', format='png', bbox_inches='tight', dpi=500)
    #             plt.show()
            
    #         elif risk_type == 'vehicle':
    #             self.r_grid_vehicles = np.vectorize(self.total_risk)(s_grid, d_grid, self.s_s, self.s_e)
    #             plt.figure(figsize=(10, 6))
    #             contour = plt.contourf(s_grid, d_grid, self.r_grid_vehicles, levels=50, cmap='jet')
                
    #             # Create colorbar and adjust its properties
    #             cbar = plt.colorbar(contour)
    #             cbar.set_label('Risk Level', fontsize=14)  # Adjust colorbar label font size
    #             cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick label font size

    #             plt.xlabel('Longitudinal Position (m)', fontsize=14)
    #             plt.ylabel('Lateral Deviation (m)', fontsize=14)
    #             plt.axhline(0, color='white', linestyle='--', linewidth=2, label='Lane Center')
    #             plt.legend(prop={'size': 12})
    #             plt.show()