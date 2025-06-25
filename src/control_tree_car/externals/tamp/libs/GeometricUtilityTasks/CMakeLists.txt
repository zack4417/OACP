cmake_minimum_required (VERSION 3.5)

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

#################################
# Define executable
#################################
add_definitions(-std=c++14)

file(GLOB SOURCES *.cpp *.h)

add_library(${project_name} ${SOURCES})
#set_target_properties(${project_name} PROPERTIES PUBLIC_HEADER "car_kinematic.h")
target_include_directories(${project_name} PUBLIC ".")
target_link_libraries(
 ${project_name}
 Core
 Geo
 Kin
 KOMO
 MathUtility
 )

add_subdirectory(test)
