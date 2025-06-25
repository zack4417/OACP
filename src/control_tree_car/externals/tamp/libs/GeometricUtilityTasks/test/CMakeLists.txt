cmake_minimum_required (VERSION 2.8)

# list all tests
file(GLOB_RECURSE tests *.cpp)

#find GTest
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

# specify target for each test
foreach(test ${tests})
    get_filename_component(test_name ${test} NAME_WE)
    add_executable(${test_name} ${test})
    target_link_libraries(${test_name}
                          GeometricUtilityTasks
                          gtest
                          pthread
                          )
endforeach()

# copy directory data
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
