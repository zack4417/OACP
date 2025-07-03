import os
import os.path as osp
import sys 
# Add the current directory to sys.path
cur_d = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_d) 
from std_msgs.msg import ColorRGBA
from math import radians
from risk_model import VehicleRiskModel
import random
import tf
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import yaml
import math
from shapely.geometry import Point as Point_safe
from shapely.affinity import scale, rotate 
import rospy
from visualization_msgs.msg import Marker, MarkerArray  
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3, TransformStamped
from oacp.msg import States, Controls  
from tf.transformations import quaternion_from_euler
 
COLOR_RGB_MAP = {
  'black':      np.array([0, 0, 0]) / 255.,
  'white':      np.array([255, 255, 255]) / 255.,
  'grey':       np.array([50, 50, 50]) / 255.,
  'red':        np.array([255, 0, 0]) / 255.,
  'green':      np.array([0, 255, 0]) / 255.,
  'yellow':     np.array([255, 255, 0]) / 255.,
  'dark_green': np.array([0, 100, 0]) / 255.,
  'blue':       np.array([0, 0, 255]) / 255.,
  'light_blue': np.array([0, 255, 255]) / 255.,
  'orange':     np.array([255,165,0]) / 255.,
}

TWO_COLOR_SCHEME1 = [
  np.array([255., 59., 59.]) / 255., # red
  np.array([7., 7., 7.]) / 255., # black
]

TWO_COLOR_SCHEME2 = [
  np.array([254., 129., 125.]) / 255., # red
  np.array([129., 184., 223.]) / 255., # blue
]

THREE_COLOR_SCHEME1 = [
  np.array([210., 32., 39.]) / 255., # red
  np.array([56., 89., 137.]) / 255., # blue1
  np.array([127., 165., 183.]) / 255., # blue2
]

FOUR_COLOR_SCHEME1 = [
  np.array([43., 85., 125]) / 255., # Cblue
  np.array([69., 189., 155.]) / 255., # Cgreen
  np.array([240., 207., 110.]) / 255., # Cred: red
  np.array([253., 207., 110.]) / 255., # Cyel: yellow
]

FIVE_COLOR_SCHEME1 = [
  np.array([89., 89., 100.]) / 255., # purble black
  np.array([95., 198., 201.]) / 255., # light blue
  np.array([1., 86., 153.]) / 255., # blue
  np.array([250., 192., 15.]) / 255., # yellow
  np.array([243., 118., 74.]) / 255., # orange
]

FIVE_SEQ_COLOR_MAPS = [
  'Blues',
  'Greens',
  'Reds',
  'Greys',
  'Purples',
]

FIVE_MARKER_TYPES = [
  'o', '^', 's', 'X', ''
]

color_map_five = {
    "purple_black": (89/255, 89/255, 100/255, 1.0),
    "lightblue": (95/255, 198/255, 201/255, 1.0),
    "vscodeblue": (0.0, 0.0, 1.0, 1.0),
    "blue": (0.0, 0.7, 0.8, 1.0), 
    "yellow": (250/255, 192/255, 15/255, 1.0), 
    "orange": (243/255, 118/255, 74/255, 0.4), 
    "red": (1.0, 0.3, 0.1, 0.5),
    "orangeRed": (1.0, 0.275, 0.0, 1.0),
    "redarrow": (240/255, 207/255, 110/255, 1)
} 
color_map = {
    "kblack": (0.0, 0.0, 0.0, 1.0),
    "kWhite": (1.0, 1.0, 1.0, 1.0),
    "kRed": (1.0, 0.0, 0.0, 1.0),
    "kPink": (1.0, 0.44, 0.70, 1.0),
    "kGreen": (0.0, 1.0, 0.0, 1.0),
    "kBlue": (0.0, 0.0, 1.0, 1.0),
    "kMarine": (0.5, 1.0, 0.83, 1.0),
    "kYellow": (1.0, 1.0, 0.0, 1.0),
    "kCyan": (0.0, 1.0, 1.0, 1.0),
    "kMagenta": (1.0, 0.0, 1.0, 1.0),
    "kViolet": (0.93, 0.43, 0.93, 1.0),
    "kOrangeRed": (1.0, 0.275, 0.0, 1.0),
    "kOrange": (1.0, 0.65, 0.0, 1.0),
    "kDarkOrange": (1.0, 0.6, 0.0, 1.0),
    "kGold": (1.0, 0.84, 0.0, 1.0),
    "kGreenYellow": (0.5, 1.0, 0.0, 1.0),
    "kFroestGreen": (0.13, 0.545, 0.13, 1.0),
    "kSpringGreen": (0.0, 1.0, 0.5, 1.0),
    "kSkyBlue": (0.0, 0.749, 1.0, 1.0),
    "kMediumOrchid": (0.729, 0.333, 0.827, 1.0),
    "kGrey": (0.5, 0.5, 0.5, 1.0),
    "kPurple": (0.5, 0.0, 0.5, 1.0)
} 
class MinimalSubscriber:
    def __init__(self):    
        random.seed(0)
        self.a_ell = 5.6
        self.b_ell = 2.9
        self.a_rect_ev = 3.6
        self.b_rect_ev = 1.22
        self.a_rect = 3.4
        self.b_rect = 2
        self.perception_lat = 200 
        self.perception_lon = 5
        self.cnt = 1
        self.loop = 0
        self.index = 0
        self.intersection = [False, False, False, False, False, False]
        self.upper = 70 #130 for cruisie
        self.lower_lim = -70
        self.upper_lim = self.upper
        # risk parameters 
        self.r_total = 0
        self.v_occ_s = 0
        self.v_occ_s1 = 0
        # Define model parameters
        self.v_max = 10  # Maximum phantom vehicle velocity (in m/s)
        self.T_pred = 4  # Prediction horizon (in seconds)
        self.l_w = 3.75   # Lane width (in meters)
        self.Z = 2        # Confidence interval factor for lateral deviation
        self.s_s = -30     # Start of Phantom Vehicle Set (PVS)
        self.s_e = -10      # End of Phantom Vehicle Set (PVS)
        # Risk thresholds
        self.c_th_min = 0.2 
        self.c_th_max = 40.0
        self.c_th_max1 = 60.0
        self.v_occ_min = 1  # Ego agent: Minimum velocity allowed (m/s)
        self.v_occ_max = 10 # Ego agent: Maximum velocity allowed (m/s)
        # Initialize the risk model
        self.risk_model = VehicleRiskModel(self.v_max, self.T_pred, self.l_w, self.Z, self.s_s, self.s_e, self.c_th_min, self.c_th_max, self.v_occ_min, self.v_occ_max)
        self.risk_model1 = VehicleRiskModel(self.v_max, self.T_pred, self.l_w, self.Z, self.s_s, self.s_e, self.c_th_min, self.c_th_max1, self.v_occ_min, self.v_occ_max)
        sleep(5)
        # Initialize velocity boundaries with defaults
        self.vxc_min = 1.0
        self.vxc_max = 8
        self.pre_x = []
        self.pre_y = []
        self.Gotit = 1 
 
        self.psi_constrols = np.array([])
        self.num_goal = 10
        # visualization rviz
        self.sample_dis = 2.0
        self.dash_length = 3  # Length of the dashes
        self.gap_length = 6.0   # Length of the gaps
        # Add highway lines IDM
        self.lines = [
            # (-5.625, -5.625),
            (1.875, 1.875) 
            # (-1.875, -1.875)
            # (5.625, 5.625),
            # (-9.375, -9.375) 
            # (9.375, 9.375) 
        ]
        self.boundary = [
              (-1.875, -1.875),
            #  (-13.125, -13.125),
              (5.625, 5.625)
            # (13.125, 13.125)
        ]  
        self.lines_hv = [
            # (-5.625, -5.625),
            (1.875, 1.875)  
        ]
        self.boundary_hv = [
              (-1.875, -1.875),
            #  (-13.125, -13.125),
              (5.625, 5.625)
            # (13.125, 13.125)
        ]  
        self.marker_array_static = MarkerArray()
        
        self.init_static_markers()
        self.init_occlusion_obstacles()
            #       (0, 0),
            # (-3.75, -3.75),
            # (3.75, 3.75),
            # (-7.5, -7.5),
            # (7.5, 7.5) 
        with open(osp.join(cur_d, '../config.yaml')) as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader) # 
            setting = str(data["setting"])
 
        with open(osp.join(cur_d, './config.yaml')) as f_obstacles:
            data_obstacles = f_obstacles.read()
            data_obstacles = yaml.load(data_obstacles, Loader=yaml.FullLoader) #
             
            obstacles_setting = str(data_obstacles["obstacles"])
            self.num_obs = data_obstacles["num_obs"]
            self.dt = data_obstacles["delta_time"]
            self.obstacles_setting = 1
            if obstacles_setting  == "static":
                self.obstacles_setting = 0
            print (self.obstacles_setting)
        if setting == "cruise_IDM":
            self.setting = 0 

        elif setting == "OCC_IDM":
            self.setting = 1 
        else:
            self.setting = 3
 
        print (self.setting)
     
        self.obs = np.zeros([self.num_obs+1, 6])  
        self.obs[0,:4] = [-40, -0,  0, 0.0]
        self.other_vehicles = np.array([])
        # self.obs_std_devs = [1.0, 0.5, 0.5, 0.0, 0, 0]
        # self.sv_control_std_devs = [0.2]
        # # [0.05, 0.02, 0.02, 0.005]
        self.obs_std_devs = [0, 0, 0, 0, 0, 0]
        self.sv_control_std_devs = [0]
        # [0.05, 0.02, 0.02, 0.005]        
        if self.setting == 1:    #occ
            self.obs[0,:4]= [-50,  0, 5.0, 0.0]          
            total_obs = 10
            mid_lane_x = [3.75, 0]
            self.other_vehicles = np.zeros([total_obs, 6])       # x y vx vy dist psi

            # Assign lane positions to vehicles (alternating between lanes)
            lanes = np.tile(mid_lane_x, int(np.ceil(total_obs / len(mid_lane_x))))
            self.other_vehicles[:, 0] = lanes[:total_obs]  # Assign x positions (lanes)
                
            self.other_vehicles[:,1] = np.array([-10, 30, -20, 20, -30, 10, 
                                                    -40, 40, -50, 50 ]) 
            self.other_vehicles[:,5] = np.zeros(total_obs) 
            # Desired speeds for each vehicle in vy (primary movement)
            # self.other_vehicles_desired = [9, -8.5, 4, -7, 5.0, -5.5, 4.5, -6.5, 7.5, -8.0]  # m/s 
            self.other_vehicles_desired = [9, -9.5, 4, -7, 5.0, -5.5, 6.5, -7.5, 8.5, -9.0]  # m/s 
            # self.other_vehicles[:,3] = [5, -6, 4, -4, 4.0, -2.5, 4.0, -3.0]
            self.other_vehicles[:,3] = np.roll(self.other_vehicles_desired, 2)

            # Calculate initial distances to the ego vehicle (index 0 of obs)
            ego_x, ego_y = self.obs[0][0], self.obs[0][1]
            delta_x = self.other_vehicles[:, 0] - ego_x
            delta_y = self.other_vehicles[:, 1] - ego_y 
            self.other_vehicles[:, 4] = np.sqrt(delta_x**2 + delta_y**2)  
            self.other_vehicles[:,4] = np.where(
                        ( (self.other_vehicles[:,1] > 5.5) & (self.other_vehicles[:,0] > 2) & (self.other_vehicles[:,0] < 5) ) |
                        ( (self.other_vehicles[:,1] < -5.5) & (self.other_vehicles[:,0] > -2) & (self.other_vehicles[:,0] < 2) ),
                        np.inf,
                        self.other_vehicles[:,4]
                    )
            self.other_vehicles = self.other_vehicles[self.other_vehicles[:, 4].argsort()]
                            
            for i in range(self.num_obs):
                self.obs[i+1] = self.other_vehicles[i,:6];  # [x, y, vx, vy, dist, psi] 
            
        else: 
            self.obs[0,:4]= [-50, 0, 15.0, 0.0]   #   cruise 
            total_obs = 18
            mid_lane_y = [-11.25, -7.5, -3.75, 0, 3.75, 7.5]
            self.other_vehicles = np.zeros([total_obs, 6])       # x y vx vy dist psi
            self.other_vehicles[:,1] = (np.hstack((mid_lane_y, mid_lane_y, mid_lane_y)))
            self.other_vehicles[:,5] = np.zeros(total_obs)
            #cruise and hsrl
            if self.setting == 0:
                self.other_vehicles[:,0] = np.array([-10, 25, 60, 40, -20, 55,
                                                    70, 105, 120, 10, 80, 190,
                                                    130, 140, 180, 160,  205, 95 
                ])
            elif self.setting == 2:
                self.other_vehicles[:,0] = np.array([-10, 25, 60, 40, -20, 155,
                                                    70, 105, 120, 10, 80, 290,
                                                    130, 140, 180, 160,  105, 195 
                ])
            else:
                self.other_vehicles[:,0] = np.array([-10, 25,   60, -20, 5, -35,
                                                    90, 85,   75, 10, 25, 50,
                                                    20, 40,   10,  100, 50, 95 
                ])
            self.other_vehicles[:,4] = np.sqrt((self.other_vehicles[:,0] - self.obs[0][0])**2 + (self.other_vehicles[:,1] - self.obs[0][1])**2) + (((-self.other_vehicles[:,0] + self.obs[0][0] - self.a_ell/2)/(abs(-self.other_vehicles[:,0] + self.obs[0][0] - self.a_ell/2)+0.0001)) + 1) * 10000
            self.other_vehicles[:,4] = np.where(abs(self.other_vehicles[:,1] - self.obs[0][1])  < self.perception_lat, self.other_vehicles[:,4], np.inf)
            self.other_vehicles = self.other_vehicles[self.other_vehicles[:, 4].argsort()]

            if self.setting == 2: 
                self.other_vehicles_desired = [10, 11, 8.0, 9.5, 22, 28.5, 
                            7.5, 8.5, 7.0, 9.0, 21.5, 29.0,
                            10.0, 9.5, 7.2, 9.5, 20, 27.5]
            else:
                self.other_vehicles_desired = [10, 11, 8.0, 9.5, 18.5, 22, 
                            7.5, 8.5, 7.0, 9.0, 18.5, 19.0,
                            10.0, 9.5, 7.2, 9.5, 15, 20.5]
            self.other_vehicles[:,2] = np.roll(self.other_vehicles_desired, 0)
            for i in range(self.num_obs):
                self.obs[i+1] = self.other_vehicles[i,:] 
      
        self.ours_x = []
        self.ours_y = [] 
        self.former_traj_x = []
        self.former_traj_y = []
        self.former_other_vehicles = []
        self.former_acc = np.array([])   
        self.former_theta= np.array([])
        self.former_pre_x = []
        self.former_pre_y = [] 
        self.former_pre_x_other1 = []
        self.former_pre_y_other1 = []        
        self.former_pre_x_other2 = []
        self.former_pre_y_other2 = []        
        self.former_pre_x_other3 = []
        self.former_pre_y_other3 = []        
        self.former_pre_x_other4 = []
        self.former_pre_y_other4 = []        
        self.former_pre_x_other5 = []
        self.former_pre_y_other5 = []
        self.former_obs = []
        
        self.collision = []
        self.v = self.obs[0][2]
        self.v_lon = self.obs[0][2]
        self.x = self.obs[0][0]
        self.y = self.obs[0][1]
        self.v_lat = 0
        self.w = 0.0 
        self.theta = 0
        self.steering_angle = 0
        self.a = 0
        self.jx = 0
        self.jy = 0 

        self.prev_psi = 0.0
        self.psi = 0.0

        self.sim_time = np.array([]) 
        self.flag = 1    

        # Subscriptions
        self.subscription = rospy.Subscriber('ego_vehicle_cmds', Controls, self.listener_callback)
   
        # Publishers
        self.publisher_markers = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.publisher_ev_markers = rospy.Publisher('visualization_marker_ev_array', MarkerArray, queue_size=10)
        # self.publisher_ = rospy.Publisher('ego_vehicle_obs', States, queue_size=10)
        self.publisher_ = rospy.Publisher('ego_vehicle_obs', States, queue_size=10, latch=True)
        # self.publisher_SRQ = rospy.Publisher('ego_vehicle_vel_bound', States, queue_size=10, latch=True)

        # TF Broadcaster
        self.br = tf.TransformBroadcaster() 
        # Timer
        rospy.Timer(rospy.Duration(0.05), self.timer_callback)


    def adjust_std_devs(self, distance, base_std_devs):
        adjusted_std_devs = []
        for i, base_std_dev in enumerate(base_std_devs):
            # Now, ensuring noise decreases as the distance decreases
            # The +0.1 prevents division by zero and ensures the factor doesn't become excessively large
            distance_factor = 10 / (math.sqrt(distance[i] + 0.1))  
            # Apply the distance factor in a manner that decreases noise as the factor increases
            adjusted_std = base_std_dev / max(distance_factor, 1)  # Ensure noise doesn't increase for very close distances
            if (distance[i] < 15):
                adjusted_std = adjusted_std * 0
            adjusted_std_devs.append(adjusted_std)
        return adjusted_std_devs
    def broadcast_transform(self, px, py, robot_orientation):
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (px, py, 0),
            (robot_orientation.x, robot_orientation.y, robot_orientation.z, robot_orientation.w),
            rospy.Time.now(),
            "base_link",
            "map"
        )
    
    def create_ellipse(self, center, axes, inclination):
        #mute for rviz 
        p = Point_safe(center)
        c = p.buffer(1)
        ellipse = scale(c)
        # ellipse = scale(c, *axes)
        # ellipse = rotate(ellipse, inclination)
        return ellipse

    def checkCollision(self):
        
        obs_ellipse = [self.create_ellipse((self.other_vehicles[i][0], self.other_vehicles[i][1]), (self.a_ell/2, self.b_ell/2), 0) for i in range(self.num_obs)]
        ego_ellipse = self.create_ellipse((self.obs[0][0], self.obs[0][1]), (self.a_ell/2, self.b_ell/2), 0*self.psi*180.0/np.pi)

        self.intersection = [ego_ellipse.intersects(obs_ellipse[i]) for i in range(self.num_obs)]
        for i in range(self.num_obs):
            if self.intersection[i]:
                ptsx, ptsy = ego_ellipse.intersection(obs_ellipse[i]).exterior.coords.xy
                if len(ptsx) < 10:
                    self.intersection[i] = False
        self.intersection = any([inter == True for inter in self.intersection])
     
    def add_marker(self, marker_array, marker_type, marker_id, scale, color, points=None, position=None, orientation=None):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.id = marker_id

        if points:
            marker.points.extend(points)
        if position:
            marker.pose.position = position
        if orientation:
            marker.pose.orientation = orientation

        marker_array.markers.append(marker)
     

    def add_ev_marker(self, marker_array, marker_type, marker_id, scale, color, points=None, position=None, orientation=None):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.id = marker_id
        
        if points:
            marker.points.extend(points)
        if position:
            marker.pose.position = position
        if orientation:
            marker.pose.orientation = orientation

        marker_array.markers.append(marker)
 
    def create_text_marker(self, marker_id, position, text, color="kblack", scale=1.2):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now() 
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = scale  # Scale it appropriately
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_map[color]
        marker.id = marker_id
        marker.pose.position = position
        marker.text = text
        return marker
        
    def create_line_points(self, start, end):
        points = []
        num_of_points = math.ceil((50000 - 0) / self.sample_dis)
        sample_interval = (50000 - 0) / num_of_points

        for i in range(num_of_points):
            x = -50 + i * sample_interval
            p1 = Point(x=float(x), y=float(start), z=0.0)
            p2 = Point(x=float(x + sample_interval), y=float(end), z=0.0)
            points.append(p1)
            points.append(p2)
        return points
    
    def create_dashed_line_points(self, start, end):
            points = []
            num_of_dashes = math.ceil((5000 - 0) / (self.dash_length + self.gap_length))

            for i in range(num_of_dashes):
                dash_start = -50 + i * (self.dash_length + self.gap_length)
                dash_end = dash_start + self.dash_length
                if dash_end > 50000:
                    dash_end = 50000

                p1 = Point(x=float(dash_start), y=float(start), z=0.0)
                p2 = Point(x=float(dash_end), y=float(end), z=0.0)
                points.extend([p1, p2])

                # Skip the gap
                if dash_end == 50000:
                    break

            return points 
 
    def add_house_obstacle1(self, marker_array, marker_id, position,  scale_roof=(1, 1, 1), color=(1.0, 1.0, 1.0, 1)):
        """
        Adds a house-shaped occlusion obstacle to the RViz visualization. 
        :param marker_array: The MarkerArray to add markers to.
        :param marker_id: Unique ID for the marker.
        :param position: Position of the obstacle (Point object). 
        :param scale_roof: Scale of the house (x, y, z). 
        :param color: RGBA color tuple for the obstacle.
        """
        color_msg = ColorRGBA()
        color_msg.r, color_msg.g, color_msg.b, color_msg.a = color  # Set RGBA color
 
        
        # 2. Add the Roof (MESH_RESOURCE for a custom 3D model)
        roof_marker = Marker()
        roof_marker.header.frame_id = "map"
        roof_marker.header.stamp = rospy.Time.now()
        roof_marker.type = Marker.MESH_RESOURCE
        roof_marker.action = Marker.ADD
        roof_marker.id = marker_id + 1
        roof_marker.color = color_msg

        # Path to your 3D mesh file
        roof_marker.mesh_resource = "file:///home/zack/Documents/ros_ws/OACP_ws/src/oacp/python_env/mesh/t.stl"

        # Position and scale the roof mesh
        roof_marker.pose.position = position
        roof_marker.pose.position.z += 4.0  # Add 2 meters to the Z position (altitude)

        roof_marker.scale.x = scale_roof[0]  # Scale the mesh if needed
        roof_marker.scale.y = scale_roof[1]
        roof_marker.scale.z = scale_roof[2]
        
        marker_array.markers.append(roof_marker)
       

    def add_house_obstacle2(self, marker_array, marker_id, position,  scale_roof=(1, 1, 1), color=(1.0, 1.0, 1.0, 1)):
        """
        Adds a house-shaped occlusion obstacle to the RViz visualization. 
        :param marker_array: The MarkerArray to add markers to.
        :param marker_id: Unique ID for the marker.
        :param position: Position of the obstacle (Point object). 
        :param scale_roof: Scale of the house (x, y, z). 
        :param color: RGBA color tuple for the obstacle.
        """
        color_msg = ColorRGBA()
        color_msg.r, color_msg.g, color_msg.b, color_msg.a = color  # Set RGBA color
 
        
        # 2. Add the Roof (MESH_RESOURCE for a custom 3D model)
        roof_marker = Marker()
        roof_marker.header.frame_id = "map"
        roof_marker.header.stamp = rospy.Time.now()
        roof_marker.type = Marker.MESH_RESOURCE
        roof_marker.action = Marker.ADD
        roof_marker.id = marker_id + 1
        roof_marker.color = color_msg

        # Path to your 3D mesh file
        roof_marker.mesh_resource = "file:///home/zack/Documents/ros_ws/OACP_ws/src/oacp/python_env/mesh/t.stl"

        # Position and scale the roof mesh
        roof_marker.pose.position = position
        roof_marker.pose.position.z += 4.0  # Add 2 meters to the Z position (altitude)

        roof_marker.scale.x = scale_roof[0]  # Scale the mesh if needed
        roof_marker.scale.y = scale_roof[1]
        roof_marker.scale.z = scale_roof[2]
         # # Set the orientation of the roof mesh
        # # Use a 180-degree rotation around the Z-axis (reverse the orientation)
        quat = tf.transformations.quaternion_from_euler(0, 0, 3.14159)  # 3.14159 radians = 180 degrees
        roof_marker.pose.orientation = Quaternion(*quat)       
        marker_array.markers.append(roof_marker)
       
    def add_house_obstacle3(self, marker_array, marker_id, position,  scale_roof=(1, 1, 1), color=(1.0, 1.0, 1.0, 1)):
        """
        Adds a house-shaped occlusion obstacle to the RViz visualization.

        :param marker_array: The MarkerArray to add markers to.
        :param marker_id: Unique ID for the marker.
        :param position: Position of the obstacle (Point object). 
        :param scale_roof: Scale of the house (x, y, z). 
        :param color: RGBA color tuple for the obstacle.
        """
        color_msg = ColorRGBA()
        color_msg.r, color_msg.g, color_msg.b, color_msg.a = color  # Set RGBA color
 
        
        # 2. Add the Roof (MESH_RESOURCE for a custom 3D model)
        roof_marker = Marker()
        roof_marker.header.frame_id = "map"
        roof_marker.header.stamp = rospy.Time.now()
        roof_marker.type = Marker.MESH_RESOURCE
        roof_marker.action = Marker.ADD
        roof_marker.id = marker_id + 1
        roof_marker.color = color_msg

        # Path to your 3D mesh file
        roof_marker.mesh_resource = "file:///home/zack/Documents/ros_ws/OACP_ws/src/oacp/python_env/mesh/t.stl"

        # Position and scale the roof mesh
        roof_marker.pose.position = position
        roof_marker.pose.position.z += 4.0  # Add 2 meters to the Z position (altitude)

        roof_marker.scale.x = scale_roof[0]  # Scale the mesh if needed
        roof_marker.scale.y = scale_roof[1]
        roof_marker.scale.z = scale_roof[2] 
        # # # Set the orientation of the roof mesh
        # # # Use a 180-degree rotation around the Z-axis (reverse the orientation)
        # quat = tf.transformations.quaternion_from_euler(0, 0, 3.14159)  # 3.14159 radians = 180 degrees
        # roof_marker.pose.orientation = Quaternion(*quat)
        
        marker_array.markers.append(roof_marker)

    def add_house_obstacle4(self, marker_array, marker_id, position,  scale_roof=(8, 8, 8), color=(1.0, 1.0, 1.0, 1)):
        """
        Adds a house-shaped occlusion obstacle to the RViz visualization.

        :param marker_array: The MarkerArray to add markers to.
        :param marker_id: Unique ID for the marker.
        :param position: Position of the obstacle (Point object). 
        :param scale_roof: Scale of the house (x, y, z). 
        :param color: RGBA color tuple for the obstacle.
        """
        color_msg = ColorRGBA()
        color_msg.r, color_msg.g, color_msg.b, color_msg.a = color  # Set RGBA color
 
        
        # 2. Add the Roof (MESH_RESOURCE for a custom 3D model)
        roof_marker = Marker()
        roof_marker.header.frame_id = "map"
        roof_marker.header.stamp = rospy.Time.now()
        roof_marker.type = Marker.MESH_RESOURCE
        roof_marker.action = Marker.ADD
        roof_marker.id = marker_id + 1
        roof_marker.color = color_msg

        # Path to your 3D mesh file
        roof_marker.mesh_resource = "file:///home/zack/Documents/ros_ws/OACP_ws/src/oacp/python_env/mesh/tt.stl"
        # Position and scale the roof mesh
        roof_marker.pose.position = position
        roof_marker.pose.position.z += 4.0  # Add 2 meters to the Z position (altitude)

        roof_marker.scale.x = scale_roof[0]  # Scale the mesh if needed
        roof_marker.scale.y = scale_roof[1]
        roof_marker.scale.z = scale_roof[2] 
        # # Set the orientation of the roof mesh
        # # Use a 180-degree rotation around the Z-axis (reverse the orientation)
        quat = tf.transformations.quaternion_from_euler(0, 0, 3.14159)  # 3.14159 radians = 180 degrees
        roof_marker.pose.orientation = Quaternion(*quat)
        
        marker_array.markers.append(roof_marker)
       
    def init_occlusion_obstacles(self):
        """
        Initializes and adds house-shaped occlusion obstacles at the intersection corners.
        :param marker_array: MarkerArray to add markers to.
        """
        marker_id = 3000  # Starting ID for occlusion markers
        
        # Define positions of obstacles (for example, at the corners of the intersection)
  
        self.add_house_obstacle1(self.marker_array_static, marker_id, Point(x=-20, y=-14.5, z=0))
        marker_id += 1  
        self.add_house_obstacle2(self.marker_array_static, marker_id,  Point(x=-20, y= 18, z=0))
        marker_id += 1   
        self.add_house_obstacle3(self.marker_array_static, marker_id,  Point(x=24, y=-14.5, z=0))
        marker_id += 1  
        self.add_house_obstacle3(self.marker_array_static, marker_id,  Point(x=55, y=-14.5, z=0))
        marker_id += 1  
        self.add_house_obstacle4(self.marker_array_static, marker_id,  Point(x=24, y= 15, z=0))
        marker_id += 1   
        self.add_house_obstacle4(self.marker_array_static, marker_id,  Point(x=48.5, y= 15, z=0))
        marker_id += 1   
    def create_line_points_hv(self, start, end):
        points = []
        num_of_points = math.ceil((50000 - 0) / self.sample_dis)
        sample_interval = (50000 - 0) / num_of_points

        for i in range(num_of_points):
            y = -200 + i * sample_interval
            p1 = Point(x=float(start), y=float(y), z=0.0)
            p2 = Point(x=float(end), y=float(y + sample_interval), z=0.0)
            points.append(p1)
            points.append(p2)
        return points
        
    def create_dashed_line_points_hv(self, start, end):
            points = []
            num_of_dashes = math.ceil((5000 - 0) / (self.dash_length + self.gap_length))

            for i in range(num_of_dashes):
                dash_start = -200 + i * (self.dash_length + self.gap_length)
                dash_end = dash_start + self.dash_length
                if dash_end > 50000:
                    dash_end = 50000

                p1 = Point(x=float(start) , y=float(dash_start), z=0.0)
                p2 = Point(x=float(end), y=float(dash_end), z=0.0)
                points.extend([p1, p2]) 
                # Skip the gap
                if dash_end == 50000:
                    break 
            return points
    def init_static_markers(self):
            marker_id = 1000
            # Add the boundary markers
            for i, (start_y, end_y) in enumerate(self.boundary):
                line_points = self.create_line_points(start_y, end_y)
                self.add_marker(self.marker_array_static, Marker.LINE_STRIP, marker_id, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), points=line_points)
                marker_id += 1 
            # Add the dashed line markers
            for i, (start_y, end_y) in enumerate(self.lines):
                line_points = self.create_dashed_line_points(start_y, end_y)
                self.add_marker(self.marker_array_static, Marker.LINE_LIST, marker_id, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), points=line_points)
                marker_id += 1
            marker_id = 2000
            # Add the boundary markers
            for i, (start_x, end_x) in enumerate(self.boundary_hv):
                line_points = self.create_line_points_hv(start_x, end_x)
                self.add_marker(self.marker_array_static, Marker.LINE_STRIP, marker_id, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), points=line_points)
                marker_id += 1
            # Add the dashed line markers
            for i, (start_x, end_x) in enumerate(self.lines_hv):
                line_points = self.create_dashed_line_points_hv(start_x, end_x)
                self.add_marker(self.marker_array_static, Marker.LINE_LIST, marker_id, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), points=line_points)
                marker_id += 1
    def publish_markers(self):
        marker_array = MarkerArray()
        marker_array_ev = MarkerArray()
        marker_id = 0
        marker_id_ev = 100
        if self.flag == 0: 
            prev = 0 
            for i in range(self.num_goal):   
                if i >= len(self.steps):
                    rospy.logwarn(f"Index {i} is out of range for steps with length {len(self.steps)}.")
                    continue 
                points = []
                for j in range(prev + 1, prev + self.steps[i]):
                    if j >= len(self.pre_x):
                        rospy.logwarn(f"Index {j} is out of range for pre_x with length {len(self.pre_x)}.")
                        continue 
                    point = Point()
                    point.x = self.pre_x[j]
                    # point.y = self.pre_y[j]  # to plot line
                    point.y = 0  # to plot line
                    point.z = 0.0
                    points.append(point) 
                if i == self.index:
                    self.add_marker(marker_array, Marker.LINE_STRIP, marker_id, (0.12, 0.12, 0.12), color_map_five["orangeRed"], points)
                    marker_id += 1
                    self.add_marker(marker_array, Marker.SPHERE_LIST, marker_id, (0.25, 0.25, 0.25), color_map_five["orangeRed"], points)
                    marker_id += 1

                    endpoint_pose = Point()
                    endpoint_pose.x = self.pre_x[prev + self.steps[i] - 1]
                    endpoint_pose.y = self.pre_y[prev + self.steps[i] - 1]
                    endpoint_pose.z = 0.0
                    self.add_marker(marker_array, Marker.SPHERE, marker_id, (0.5, 0.5, 0.5),  color_map_five["orangeRed"], position=endpoint_pose)
                    marker_id += 1

                    # arrow_start = Point()
                    # arrow_start.x = self.pre_x[prev + 0]
                    # arrow_start.y = self.pre_y[prev + 0]
                    # arrow_start.z = 0.0

                    # arrow_end = Point()
                    # arrow_end.x = self.pre_x[prev + 0] + self.obs[0][2]
                    # arrow_end.y = self.pre_y[prev + 0] + self.obs[0][3]
                    # arrow_end.z = 0.0
                    # #  # Define the scale for the arrowhead
                    # # arrowhead_scale = (0.2, 1.0, 1.0)  # (shaft diameter, head diameter, head length)
                    # self.add_marker(marker_array, Marker.ARROW, marker_id, (0.2, 0.5, 1.618), color_map_five["vscodeblue"], [arrow_start, arrow_end])
                    # marker_id += 1
                else:
                    self.add_marker(marker_array, Marker.LINE_STRIP, marker_id, (0.12, 0.12, 0.12), color_map_five["purple_black"], points)
                    marker_id += 1   
                    self.add_marker(marker_array, Marker.SPHERE_LIST, marker_id, (0.25, 0.25, 0.25), color_map_five["purple_black"], points)
                    marker_id += 1
                prev += self.steps[i]

            diag = np.sqrt(self.a_rect ** 2 + self.b_rect ** 2)
            for i, vehicle in enumerate(self.obs[1:]):
                position = Point()
                position.x = vehicle[0]
                position.y = vehicle[1]
                position.z = 0.5

                orientation = Quaternion()
                orientation.z = np.sin(vehicle[5] / 2)
                orientation.w = np.cos(vehicle[5] / 2) 
                # # color = (color_map_five["blue"] if i >= self.num_obs else color_map_five["orange"]) 
                # color = color_map_five["orange"]  
                # self.add_marker(marker_array, Marker.CUBE, marker_id, (4.0, 1.4, 1.2), color, position=position, orientation=orientation)
              
                # Adjust the color to include transparency (RGBA)
                color = color_map_five["orange"]
                transparency = 0.9  # Set transparency level (0: fully transparent, 1: fully opaque)

                if (np.sqrt((self.obs[0][0]-vehicle[0])**2 + (self.obs[0][1]-vehicle[1])**2) > 20 and i > 1): # not fully observable
                    transparency = 0.5  # Set transparency level (0: fully transparent, 1: fully opaque)
                color_with_transparency = (color[0], color[1], color[2], transparency)
                # Add the marker with the adjusted color
                # self.add_marker(marker_array, Marker.CUBE, marker_id, (4.0, 1.4, 1.2), color_with_transparency, position=position, orientation=orientation)
                self.add_marker(marker_array, Marker.CUBE, marker_id, (1.4, 4.0, 1.2), color_with_transparency, position=position, orientation=orientation)
                marker_id += 1
                
                # Create a text marker for the velocity
                velocity =  np.sqrt(vehicle[2]**2 + vehicle[3]**2 )   # Assuming the velocity is at index 2
                velocity_text = "{:.2f} m/s".format(velocity)
                text_position = Point()
                text_position.x = position.x
                text_position.y = position.y
                text_position.z = position.z + 1.0  # Position it above the vehicle
                velocity_marker = self.create_text_marker(marker_id, text_position, velocity_text)
                marker_array.markers.append(velocity_marker)
                marker_id += 1
            for i, vehicle in enumerate(self.other_vehicles[len(self.obs)-1:,:6]):
                position = Point()
                position.x = vehicle[0]
                position.y = vehicle[1]
                position.z = 0.5
                orientation = Quaternion()
                orientation.z = np.sin(vehicle[5] / 2)
                orientation.w = np.cos(vehicle[5] / 2)
                color = color_map_five["blue"]
                # self.add_marker(marker_array, Marker.CUBE, marker_id, (4.0, 1.4, 1.2), color, position=position, orientation=orientation)
                self.add_marker(marker_array, Marker.CUBE, marker_id, (1.4, 4.0, 1.2), color, position=position, orientation=orientation)
                marker_id += 1
                # Create a text marker for the velocity
                velocity = np.sqrt(vehicle[2]**2 + vehicle[3]**2 )  #  
                # print("sv:vel_text: ", velocity)
                velocity_text = "{:.2f} m/s".format(velocity)
                text_position = Point()
                text_position.x = position.x
                text_position.y = position.y
                text_position.z = position.z + 1.0  # Position it above the vehicle
                velocity_marker = self.create_text_marker(marker_id, text_position, velocity_text)
                marker_array.markers.append(velocity_marker)
                marker_id += 1
            robot_position = Point()
            robot_position.x = self.obs[0][0]
            robot_position.y = self.obs[0][1]
            robot_position.y = 0
            robot_position.z = 0.5

            robot_orientation = Quaternion()
            robot_orientation.z = np.sin(self.psi / 2)
            robot_orientation.w = np.cos(self.psi / 2)
            robot_orientation.z = np.sin(0)
            robot_orientation.w = np.cos(0)

            self.add_ev_marker(marker_array_ev, Marker.CYLINDER, marker_id_ev, (self.a_ell, self.b_ell, 1.0), (1.0, 102/255, 102/255, 0.2), position=robot_position, orientation=robot_orientation)
            marker_id_ev += 1

            position = Point() 
            position.x = self.obs[0][0]
            position.y = self.obs[0][1]
            position.y = 0
            position.z = 0.5

            self.add_ev_marker(marker_array_ev, Marker.CUBE, marker_id_ev, (4.0, 1.4, 1.2), color_map_five[ "red"], position=position, orientation=robot_orientation)
            marker_id_ev += 1

            # Create a text marker for the  EV  velocity 
            velocity_text = "{:.2f} m/s".format( self.obs[0][2])
            # print("ev:vel_text: ", self.obs[0][2])
            text_position = Point() 
            text_position.x = position.x
            text_position.y = position.y
            text_position.z = position.z + 1.0  # Position it above the vehicle
            velocity_marker = self.create_text_marker(marker_id, text_position, velocity_text)
            marker_array.markers.append(velocity_marker)
            marker_id += 1 
             
            self.broadcast_transform(position.x, position.y, robot_orientation)
            self.publisher_markers.publish(marker_array)
            self.publisher_ev_markers.publish(marker_array_ev) 
            self.publisher_markers.publish(self.marker_array_static) 
    def IDM(self): 
        so = 3
        T = 1 
        l = self.a_ell 
        a_dcc = 4
        a_acc = 3
        b = 3 
        mean_noise = 0.
       
        for i in range(len(self.other_vehicles)):
            nearest = -1
            x = self.other_vehicles[i][0]
            y = self.other_vehicles[i][1]
            inLane = self.other_vehicles[:,1] - y * np.ones(len(self.other_vehicles))
            index = np.where( inLane == 0)

            min = 10000
            for k in range(len(index[0])):
                if index[0][k] != i:
                    if x < self.other_vehicles[index[0][k]][0]:
                        dist = self.other_vehicles[index[0][k]][0] - x
                        if dist < min:
                            min = dist
                            nearest = index[0][k]
            v_0 = self.other_vehicles[nearest][2]
            x_0 = self.other_vehicles[nearest][0]
            if (y - self.obs[0][1]) <= self.b_ell and x < self.obs[0][0]:
                if nearest != -1:
                    if self.obs[0][0] < self.other_vehicles[nearest][0]: 
                        nearest = 1
                        v_0 = self.obs[0][2]
                        x_0 = self.obs[0][0]
                else:
                    v_0 = self.obs[0][2]
                    x_0 = self.obs[0][0]
            v_1 = self.other_vehicles[i][2]
            x_1 = x

            v_r = self.other_vehicles_desired[i]
            delta_v = v_1 - v_0
            s_alpha = x_0 - x_1 - l

            # Calculate control noise
            noise = np.random.normal(mean_noise, self.sv_control_std_devs) 

            if nearest != -1:
                s_star = so + v_1 * T + (v_1 * delta_v)/(2 * np.sqrt(a_dcc*b))
                decc = a_dcc * (1 - (v_1/v_r)**4 - (s_star/s_alpha)**2) + noise
                if decc < -a_dcc:
                    decc = -a_dcc 
                self.other_vehicles[i][2] += decc * self.dt
            else:
                decc = a_acc * (1 - (v_1/v_r)**4) + noise
                if decc > a_acc:
                    decc = a_acc
                self.other_vehicles[i][2] += decc * self.dt
 
    def calculate_acceleration(self, v_1, v_0, v_r, s_alpha, so, T, a_dcc, b, a_acc, noise):
        delta_v = v_1 - v_0
        s_star = so + v_1 * T + (v_1 * delta_v) / (2 * np.sqrt(a_dcc * b))
        if s_alpha > 0:
            acceleration = a_dcc * (1 - (v_1 / v_r) ** 4 - (s_star / s_alpha) ** 2) + noise
        else:
            acceleration = a_acc * (1 - (v_1 / v_r) ** 4) + noise
        return np.clip(acceleration, -a_dcc, a_acc)

    def find_nearest_vehicle(self, index, i, y, forward=True):
        min_dist = float('inf')
        nearest = -1
        for k in index:
            if k != i:
                if (forward and y < self.other_vehicles[k][1]) or (not forward and y > self.other_vehicles[k][1]):
                    dist = abs(self.other_vehicles[k][1] - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = k
        return nearest

    def IDM_OCC(self): # change parameter to set the reaction of SVs to the EV in interaction 
        # IDM Parameters
        so = 3           # Minimum spacing (meters)
        T = 1.0            # Safe time headway (seconds)
        l = self.a_ell   # Vehicle length or desired spacing
        a_dcc = 4        # Maximum deceleration (m/s²)
        a_acc = 4        # Maximum acceleration (m/s²)
        b = 4            # Comfortable braking deceleration (m/s²)
        mean_noise = 0.  # Mean of control noise

        lane_tolerance = 2  # meters, adjust based on actual lane width

        for i in range(len(self.other_vehicles)):
            x = self.other_vehicles[i][0]  # Lane position
            y = self.other_vehicles[i][1]  # Longitudinal position

            # Identify vehicles in the same lane (within a tolerance)
            inLane = np.abs(self.other_vehicles[:, 0] - x) <= lane_tolerance
            index = np.where(inLane)[0]

            if x == 3.75:
                # Handle forward lane vehicle behavior
                nearest = self.find_nearest_vehicle(index, i, y, forward=True)

                v_0 = self.other_vehicles[nearest][3] if nearest != -1 else 0
                y_0 = self.other_vehicles[nearest][1] if nearest != -1 else y + 100  # large value for no vehicle ahead

                if abs(x - self.obs[0][0]) <= 3.75  and y < self.obs[0][1]:
                    # Adjust nearest vehicle based on observation
                    if nearest != -1 and self.obs[0][1] < self.other_vehicles[nearest][1]:
                        v_0 = self.obs[0][3]
                        y_0 = self.obs[0][1]
                    else:
                        v_0 = self.obs[0][3]
                        y_0 = self.obs[0][1]

                v_1 = self.other_vehicles[i][3]  # Current vy
                v_r = self.other_vehicles_desired[i]
                s_alpha = y_0 - y - l

                # Control noise
                noise = np.random.normal(mean_noise, self.sv_control_std_devs)

                # Calculate acceleration
                acceleration = self.calculate_acceleration(v_1, v_0, v_r, s_alpha, so, T, a_dcc, b, a_acc, noise)
                self.other_vehicles[i][3] += acceleration * self.dt

            else:
                # Handle backward lane vehicle behavior
                nearest = self.find_nearest_vehicle(index, i, y, forward=False) 
                v_0 = self.other_vehicles[nearest][3] if nearest != -1 else 0
                y_0 = self.other_vehicles[nearest][1] if nearest != -1 else y - 100  # large value for no vehicle behind
                # rospy.loginfo(f"delta to ev : { abs(x - self.obs[0][0]) } y : {y } self.obs[0][1]: {self.obs[0][1]}") 

                if abs(x - self.obs[0][0]) <= 3.75 and y > self.obs[0][1]: 
                    # rospy.logwarn(f"delta to ev : { abs(x - self.obs[0][0]) } y : {y } self.obs[0][1]: {self.obs[0][1]}") 
                    # Adjust nearest vehicle based on observation
                    if nearest != -1 and self.obs[0][1] > self.other_vehicles[nearest][1]:
                        v_0 = self.obs[0][3]
                        y_0 = self.obs[0][1]
                    else:
                        v_0 = self.obs[0][3]
                        y_0 = self.obs[0][1]

                v_1 = self.other_vehicles[i][3]  # Current vy
                v_r = self.other_vehicles_desired[i]
                s_alpha = abs(y_0 - y) - l

                # Control noise
                noise = np.random.normal(mean_noise, self.sv_control_std_devs)

                # Calculate acceleration
                acceleration = self.calculate_acceleration(abs(v_1), abs(v_0), v_r, s_alpha, so, T, a_dcc, b, a_acc, noise)
                self.other_vehicles[i][3] -= acceleration * self.dt
    def listener_callback(self, msg): 
        self.Gotit = 1 
        self.v = msg.v
        self.w = msg.w     
        self.jx = msg.jx
        self.jy = msg.jy   
        cnt = 0
        self.steps = []
        self.pre_x = [] 
        self.pre_y = [] 
        for i in msg.batch.poses:
            self.pre_x.append(i.position.x)
            self.pre_y.append(i.position.y)
            if abs(i.position.x - self.pre_x[0]) < 0.001 and len(self.pre_x) > 1:
                self.steps.append(cnt)
                cnt = 0
            cnt+=1
        self.steps.append(cnt)
        # print("received msg")  
      
        self.num_goal = msg.goals
        self.index = msg.index
    
    def timer_callback(self, event):
        # rospy.loginfo("Timer callback triggered")
        # rospy.loginfo(f"Current obs value: {self.obs}")
 
        if self.Gotit or self.flag:
                t1 = time()
                dt = self.dt
                self.former_traj_x.append(self.obs[0][0])
                self.former_traj_y.append(self.obs[0][1]- 0.2)
                self.former_other_vehicles.append([self.other_vehicles])
                # self.collision.append([])
                self.ours_x.append(self.obs[0][0])
                self.ours_y.append(self.obs[0][1])
                self.loop += 1
                self.sim_time = np.append(self.sim_time, self.loop * dt)
                if self.Gotit: 
                    self.psi += self.w * dt
                    self.obs[0][2] = self.v * np.cos(self.psi)    #vx
                    self.obs[0][3] = self.v * np.sin(self.psi)    #vy
                    self.obs[0][0] += self.obs[0][2] * dt    #x
                    self.obs[0][1] += self.obs[0][3] * dt    #y
                    self.former_obs.append([self.obs[0][0], self.obs[0][1], self.obs[0][2], self.obs[0][3]])

            
                    # self.IDM()
                    self.IDM_OCC()
                    if self.obstacles_setting == 0:
                        self.other_vehicles[:,2] = 0    #vx
                        self.other_vehicles[:,3] = 0    #vy
                    self.other_vehicles[:,0] += self.other_vehicles[:,2] * dt    #x
                    self.other_vehicles[:,1] += self.other_vehicles[:,3] * dt    #y
            
                    self.other_vehicles[:,4] = np.sqrt((self.other_vehicles[:,0] - self.obs[0][0])**2 + (self.other_vehicles[:,1] - self.obs[0][1])**2) + (((-self.other_vehicles[:,0] + self.obs[0][0] - self.perception_lon)/(abs(-self.other_vehicles[:,0] + self.obs[0][0]-self.perception_lon)+0.0001)) + 1) * 10000
                    self.other_vehicles[:,4] = np.where(abs(self.other_vehicles[:,1] - self.obs[0][1]) < self.perception_lat, self.other_vehicles[:,4], np.inf)
                    self.other_vehicles[:,4] = np.where(
                        ( (self.other_vehicles[:,1] > 5.5) & (self.other_vehicles[:,0] > 2) & (self.other_vehicles[:,0] < 5) ) |
                        ( (self.other_vehicles[:,1] < -5.5) & (self.other_vehicles[:,0] > -2) & (self.other_vehicles[:,0] < 2) ),
                        np.inf,
                        self.other_vehicles[:,4]
                    )
                    self.other_vehicles = self.other_vehicles[self.other_vehicles[:, 4].argsort()]
                    # Determine the number of vehicles to add noise to
                    num_vehicles = min(len(self.obs) - 1, len(self.other_vehicles))
                    distance = np.sqrt((self.other_vehicles[:num_vehicles,0] - self.obs[0][0])**2 + (self.other_vehicles[:num_vehicles,1] - self.obs[0][1])**2)  
                    # Add Gaussian noise to each dimension of the observations, and dynamic adjust noise level
                    noise_matrix = np.random.normal(0, self.obs_std_devs, (num_vehicles, 6)) 
                    noise_matrix = self.adjust_std_devs(distance, noise_matrix)   
                    # print("Added noise:\n", noise_matrix)
                    self.obs[1:] = self.other_vehicles[:len(self.obs)-1,:] + noise_matrix  
                    # self.other_vehicles[:,0] += self.other_vehicles[:,2] * dt    #x
                    # self.other_vehicles[:,1] += self.other_vehicles[:,3] * dt    #y
                    # self.other_vehicles[:,4] = np.sqrt((self.other_vehicles[:,0] - self.obs[<80][0])**2 + (self.other_vehicles[:,1] - self.obs[0][1])**2) + (((-self.other_vehicles[:,0] + self.obs[0][0]-self.a_ell/2)/(abs(-self.other_vehicles[:,0] + self.obs[0][0]-self.a_ell/2)+0.0001)) + 1) * 10000
                    # self.other_vehicles = self.other_vehicles[self.other_vehicles[:, 4].argsort()]
                    # self.obs[1:] = self.other_vehicles[:len(self.obs)-1,:]
                    
                self.checkCollision()
                msg = States()
                msg.x = self.obs[:,0].T.tolist()
                msg.y = self.obs[:,1].T.tolist()
                msg.vx = self.obs[:,2].T.tolist()
                msg.vy = self.obs[:,3].T.tolist()
                msg.psi = (self.psi * np.ones(5)).tolist()
                msg.psidot = ((msg.psi[0] - self.prev_psi)/dt)  
                self.prev_psi = msg.psi[0]
                """
                Publish the current velocity boundaries at a regular interval.
                """  
                # Calculate total risk for a sample position and lateral deviation

                s_ego = self.obs[0,1]  # position of the ego vehicle
                d_ego = 0  # Lateral deviation of the ego vehicle
                if self.obs[0,0] < 5 and (self.obs[0,0] + self.obs[0,2] * 4 > 0):
                    # RISK in Point 1
                    if  self.obs[0,0] > 0:
                        risk1_buildings = 0                        
                        risk2_buildings = 0
                    else: 
                        # OCC buildings1
                        self.s_s, self.s_e = self.risk_model.calculate_occlusion_area(self.obs[0, 0], self.obs[0, 1], -4.5, -32, -4.5, -12, 0)  # Left buildings
                        # print(f"Building 1: start{self.s_s},  end{self.s_e}, {self.obs[0, 0]}, {self.obs[0, 1]}")

                        risk1_buildings = self.risk_model.total_risk(s_ego, d_ego, self.s_s, self.s_e)
                       # OCC buildings2
                        self.s_s, self.s_e = self.risk_model.calculate_occlusion_area(self.obs[0,0], self.obs[0,1],  -4.5, -38.5,  -4.5, -8.5, 3.75) #right buildings
                        # print(f"Building 2: start{self.s_s},  end{self.s_e}")

                        # OCC vehicles
                        risk2_buildings = self.risk_model.total_risk(s_ego, d_ego, self.s_s, self.s_e)
                        # self.risk_model.cal_risk_grid('buildings')
                    # OCC vehicles
                    risk1_vehicles = 0  
                    # Calculate total risk
                    risk1 = risk1_buildings + risk1_vehicles  
                    # RISK in Point 2
                    # Filter by y < 0 and -1 < x < 1 
                    filtered_vehicles = self.other_vehicles[
                        (self.other_vehicles[:, 1] < 0) & 
                        (-1 < self.other_vehicles[:, 0]) & (self.other_vehicles[:, 0] < 1)
                    ] 
                    # Check if there are any vehicles matching the criteria
                    if filtered_vehicles.shape[0] > 0:
                        # Sort by distance (5th column) and get the nearest vehicle
                        nearest_vehicle0 = filtered_vehicles[filtered_vehicles[:, 4].argsort()][0]
                        # print(f"nearest_vehicle: x{nearest_vehicle0[0]},  y{nearest_vehicle0[1]}")
                        risk2_vehicle1 = 0
                        if filtered_vehicles.shape[0] > 1:
                            nearest_vehicle1 = filtered_vehicles[filtered_vehicles[:, 4].argsort()][1]
                            # print(f"nearest_vehicle: x{nearest_vehicle1[0]},  y{nearest_vehicle1[1]}")
                            self.s_s, self.s_e = self.risk_model.calculate_occlusion_area(
                            self.obs[0, 0], self.obs[0, 1], 0.5, nearest_vehicle1[1] - 3, 0.5, nearest_vehicle1[1] + 2, 3.75)  # Left buildings
                            risk2_vehicle1 = self.risk_model.total_risk(s_ego, d_ego, self.s_s, self.s_e)

                        # Calculate occlusion area using the nearest vehicle's position
                        self.s_s, self.s_e = self.risk_model.calculate_occlusion_area(
                            self.obs[0, 0], self.obs[0, 1], 0.5, nearest_vehicle0[1] - 3, 0.5, nearest_vehicle0[1] + 2, 3.75)  # Left buildings
                        risk2_vehicle0 = self.risk_model.total_risk(s_ego, d_ego, self.s_s, self.s_e)
                        # self.risk_model.cal_risk_grid('building') 
                        risk2_vehicles = risk2_vehicle0 + risk2_vehicle1
                        # print(f"risk2_vehicles: {risk2_vehicles*2}, OCC START Vehicle2: {self.s_s}, OCC END Vehicle2: {self.s_e}, risk2_buildings: {risk2_buildings}")
                    else:
                        # print("No vehicles meet the criteria -OCC")
                        risk2_vehicles = 0
                    # if self.obs[0,0] < -9 and self.obs[0,0] > -10:
                    #     self.risk_model.plot_combined_risk_map()
 
                    risk2 = risk2_buildings + risk2_vehicles * 2.0
                    self.r_total = max(risk1, risk2) 
                    self.v_occ_s = self.risk_model.dynamic_velocity_boundary(self.r_total)
                   
                    risk2 = risk2_buildings + risk2_vehicles * 2.0
                    self.r_total = max(risk1, risk2) 
                    self.v_occ_s1 = self.risk_model1.dynamic_velocity_boundary(self.r_total)
                    # print(f"Total Risk: {self.r_total}, Risk 1: {risk1}, Risk 2: {risk2}")
                    # print(f"Safe Velocity Boundary: {self.v_occ_s} m/s, Pos_EV: {self.obs[0, 0]}m")              
    
                    msg.vxc_min = 0
                    msg.vxc_max = self.v_occ_s                
                    msg.vxc1_max = self.v_occ_s1                
                    # msg.vxc_min = 0
                    # msg.vxc_max = 8
                else:
                    msg.vxc_min = 0
                    msg.vxc_max = 10
                    msg.vxc1_max = 10
                    # print(f"Total Risk: {0}")
                    # print(f"Safe Velocity Boundary: {10} m/s")              
    
            
                # print("222 self.publish_markers()  ") 
                self.publish_markers()  
                self.publisher_.publish(msg)
            
                self.Gotit = 0
                self.flag = 0 # flag=1 -> no visualization
                # print("delta: time:", time() - t1)
     
                if self.setting == 0:  #IDM
                    for i in range(len(self.other_vehicles)):
                        if self.other_vehicles[i][0] < self.lower_lim-6:                    
                            self.other_vehicles[i][0] = self.upper_lim + 5 + self.other_vehicles[i][2]
                            self.other_vehicles[i][2] -= 2 * (i%2)
                            if (self.other_vehicles[i][1] == -10 and self.other_vehicles[i][1] == -6) or self.setting == 1 or self.setting == 0: # for hsrl
                                self.other_vehicles[i][0] += 15 * (i%3)
                else: #Intersection
                    for i in range(len(self.other_vehicles)):
                        original_y = self.other_vehicles[i][1]  # Store the original value
                        # First condition
                        if original_y < self.lower_lim - 6:                    
                            self.other_vehicles[i][1] = self.upper_lim + 4
                            self.other_vehicles[i][3] -= 2 * (i % 2) 
                        
                        # Second condition checks the original value
                        if original_y > self.upper_lim + 5:                    
                            self.other_vehicles[i][1] = self.lower_lim - 5 
                            self.other_vehicles[i][3] -= 2 * (i % 2)
                        else: 
                            pass
            
          
def main():
    rospy.init_node('env_sub', anonymous=True)
    try:
        # Instantiate your subscriber class
        minimal_subscriber = MinimalSubscriber()
        # ROS 1 uses spin() to keep the script for subscribers alive
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
if __name__ == '__main__':
    main()
