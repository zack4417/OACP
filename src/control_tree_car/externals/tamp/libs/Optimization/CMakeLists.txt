cmake_minimum_required (VERSION 2.8.12)

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

#################################
# Locate depedencies
#################################

#find Boost
find_package(Boost COMPONENTS system filesystem serialization REQUIRED)

#################################
# Define executable
#################################
add_definitions(-O0 -Wall -DRAI_LAPACK -DRAI_EIGEN -DRAI_PTHREAD -Wno-terminate -fPIC -std=c++14)

#add_definitions(-std=c++14 -fPIC)

file(GLOB_RECURSE SOURCES src/*.cpp include/*.h *.tpp)

add_library(${project_name} STATIC ${SOURCES})
target_include_directories(${project_name} PUBLIC "include")
target_link_libraries(
 ${project_name}
 ${Boost_LIBRARIES}
 Core
 Optim
 pthread
)

add_subdirectory(test)
