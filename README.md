# Occlusion-Aware Contingency Planning for Autonomous Vehicles

[![arXiv](https://img.shields.io/badge/arXiv-2502.06359-b31b1b.svg)](https://arxiv.org/abs/2502.06359)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Repository for the paper:  
**"Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Vehicles"**


## Features
https://zack4417.github.io/oacp-website/

## Features
- Real-time contingency planning for autonomous vehicles in occluded environments
- Parallel consensus optimization framework
- Highway driving simulator with configurable scenarios
- Support for synthetic IDM-based obstacles 
- ROS-based implementation for easy integration

## Citation
If you use this code in your research, please cite:
```bibtex
@article{zheng2025safe,
  title={Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Vehicles},
  author={Zheng, Lei and Yang, Rui and Zheng, Minzhe and Peng, Zengqi and Wang, Michael Yu and Ma, Jun},
  journal={arXiv preprint arXiv:2502.06359},
  year={2025}
}
```

## Repository Structure
```
.
├── src/
│   ├── oacp/              # Implementation of oacp approach
│   │   ├── python_env/               # Highway driving simulator
│   │   │   ├── mesh/                 # 3D building models for visualization
│   │   │   └── highway_car2.py       # Main simulator script
│   │   └── ...                       # Other package files
│   └── stats/                        # Simulation data output
├── config.yaml                       # Configuration file for scenarios
└── ...
```

## Dependencies
- **Core:**
  - [ROS Noetic](http://wiki.ros.org/noetic/Installation)
  - [Eigen QuadProg](https://github.com/jrl-umi3218/eigen-quadprog)
  - Python 3.8+
  - [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
   
## Installation
```bash
# Create workspace
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src

# Clone repository
git clone https://github.com/yourusername/oacp.git

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
 
obstacles: "dynamic" # "static" or "dynamic"
num_obs: 4         # Number of obstacles
delta_time: 0.1    # Simulation timestep
```

## Usage
Terminal 1 (Planning Node):
```bash
source devel/setup.bash
roscore
```
Terminal 2 (Planning Node):
```bash
source devel/setup.bash
rosrun oacp oacp_node
```

Terminal 3 (Simulator):
```bash
source devel/setup.bash
rosrun oacp highway_car2.py
```

Terminal 4 (Visualization):
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

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a pull request

## License
BSD 3-Clause License. See [LICENSE](LICENSE) for full text.
```
 