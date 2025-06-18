cmake_minimum_required(VERSION 3.5)
project(oacp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Ofast -mtune=corei7 -fopenmp -g ${CMAKE_CXX_FLAGS} -lstdc++fs")
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

message(STATUS ${YAML_CPP_INCLUDE_DIR})
message(STATUS ${YAML_CPP_LIBRARIES})

# message(STATUS ${YAML_CPP_INCLUDE_DIR})
# message(STATUS ${YAML_CPP_LIBRARIES})
# ${YAML_CPP_LIBRARIES}



find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  message_generation
)

# catkin_package(
#   INCLUDE_DIRS include
#   LIBRARIES oacp_node_lib
#   CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs
#   DEPENDS eigen-quadprog yaml-cpp
# )

# Declare ROS messages and services
add_message_files(
  FILES
  States.msg
  Controls.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)
 

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES oacp_node_lib
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs
)

 
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(oacp_node_lib src/optim_consensus.cpp)
target_link_libraries(oacp_node_lib ${catkin_LIBRARIES} eigen-quadprog)

add_executable(oacp_node src/main_consensus.cpp)
target_link_libraries(oacp_node oacp_node_lib ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})

 

# install(TARGETS oacp_node oacp_node_lib
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# )

# install(DIRECTORY launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS 
  python_env/highway_car2.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 

catkin_install_python(PROGRAMS 
  python_env/vel_adjust_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 
# catkin_python_setup()