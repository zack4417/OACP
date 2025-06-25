cmake_minimum_required (VERSION 2.8)
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" project_name ${project_name})
project(${project_name})

add_definitions(-std=c++14)

add_subdirectory("libs/GeometricUtilityTasks")
add_subdirectory("libs/MathUtility")
add_subdirectory("libs/Optimization")
add_subdirectory("libs/MotionPlanning")
add_subdirectory("libs/POMTP_interfaces")
