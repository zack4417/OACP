# Control-Tree Baseline Implamentation for Occlusion-Aware Contingency Planning for Autonomous Vehicles

[![arXiv](https://img.shields.io/badge/arXiv-2502.06359-b31b1b.svg)](https://arxiv.org/abs/2502.06359)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Repository of baseline experiment for the paper:
**"Safe and Real-Time Consistent Planning for Autonomous Vehicles in Partially Observed Environments via Parallel Consensus Optimization"**

## Control-Tree Optimization 
For details of Control-Tree Optimization, please refer to the paper: 
```bash
C. Phiquepal and M. Toussaint, “Control-Tree Optimization: an approach to MPC under discrete partial observability,” in IEEE International Conference on Robotics and Automation. IEEE, 2021, pp. 9666–9672.
```
**Original version of Control-Tree Optimization can be found in [Control Tree](https://github.com/ControlTrees/icra2021?tab=readme-ov-file)**

## Repository Structure

```
.
├── src/
│   ├── oacp/              # Implementation of oacp approach
│   │   ├── python_env/               # Highway driving simulator
│   │   │   ├── mesh/                 # 3D building models for visualization
│   │   │   └── highway_car2.py       # Main simulator script
│   │   └── ...                       # Other package files
|   |—— control_tree_car   # Baseline approach Control-Tree files
├── config.yaml                       # Configuration file for scenarios
└── ...
```

## Dependencies

- **Core:**

  - [ROS Noetic](http://wiki.ros.org/noetic/Installation)
  - [Eigen QuadProg](https://github.com/jrl-umi3218/eigen-quadprog)
  - Python 3.8+
- **Datasets:**

  - [NGSIM I-80 Dataset](https://drive.google.com/drive/folders/1cgsOWnc4JTeyNdBN6Fjef2-J5HqjnWyX?usp=sharing)

  ```bash
  # Place downloaded files in:
  ros_ws/src/oacp/python_env/highway/
  ```
- **OSQP**
  See [OSQP Installation](https://osqp.org/docs/release-0.6.3/)
- **Other dependencies**

```bash
libann-dev, gnuplot, libjsoncpp-dev, libx11-dev, liblapack-dev, libf2c2-dev, libeigen3-dev, libglew-dev, freeglut3-dev. 
```

  They can be installed by calling ``bash sudo apt install PACKAGE_NAME``

## Installation

```bash
# Create workspace
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src

# Clone repository
git clone -b Control-Tree https://github.com/yourusername/oacp.git

# Build external sources
cd ~/ros_ws/src/control_tree_car/externals/rai
make

# Build package
cd ~/ros_ws
catkin_make
source devel/setup.bash
```

## Configuration

Modify `config.yaml` to set your scenario:

```yaml
setting: "OCC_IDM"  # Available options:
                    #   "OCC_IDM" - Occlusion-aware intersection driving
                    #   "Racing_IDM" - High-speed racing scenario
                    #   "NGSIM" - Real-world trajectory replay
obstacles: "dynamic" # "static" or "dynamic"
num_obs: 5          # Number of obstacles
delta_time: 0.05    # Simulation timestep
```

## Usage

Terminal 1 (Planning Node):

```bash
source devel/setup.bash
roslaunch control_tree_car obstacle_avoidance_OACP.launch
```

Terminal 2 (Simulator):

```bash
source devel/setup.bash
rosrun oacp highway_car2.py
```

Terminal 3 (Visualization):

```bash
source devel/setup.bash
rosrun rviz rviz -d src/config.rviz 
```

## Visualizing Results

The simulator provides RVIZ visualization with:

- Vehicle trajectories
- Dynamic obstacles
- Occlusion-aware risk fields
- 3D environment models

![Simulation Demo](docs/simulation_demo.gif)

## Mesh Path Configuration

In `highway_car2.py`, use absolute paths for meshes. Replace `[YOUR_USERNAME]` with your actual username:

```python
# For house obstacles 1-3
roof_marker.mesh_resource = "file:///home/[YOUR_USERNAME]/ros_ws/src/oacp/python_env/mesh/t.stl"

# For house obstacle 4
roof_marker.mesh_resource = "file:///home/[YOUR_USERNAME]/ros_ws/src/oacp/python_env/mesh/tt.stl"
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for full text.

```
 
```
