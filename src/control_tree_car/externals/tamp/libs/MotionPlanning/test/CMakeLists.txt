cmake_minimum_required (VERSION 2.8)

# list all tests
file(GLOB_RECURSE tests test_*.cpp)

#find GTest
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

# specify target for each test
foreach(test ${tests})
    get_filename_component(test_name ${test} NAME_WE)
    message(STATUS "TEST NAME: ${test_name} SOURCE: ${test}")
    add_executable(${test_name} ${test})
    target_link_libraries(${test_name}
                          MotionPlanning
                          gtest
                          pthread
                          )
endforeach()
