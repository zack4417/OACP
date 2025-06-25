cmake_minimum_required (VERSION 2.8.12)

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

#################################
# Locate depedencies
#################################
find_package(Boost COMPONENTS system filesystem serialization REQUIRED)

#################################
# Define executable
#################################
add_definitions(-std=c++14)

file(GLOB SOURCES *.cpp *.h)

add_library(${project_name} ${SOURCES})
target_include_directories(${project_name} PUBLIC ".")
target_link_libraries(
 ${project_name}
 ${Boost_LIBRARIES}
 Core
 )

add_subdirectory(test)
