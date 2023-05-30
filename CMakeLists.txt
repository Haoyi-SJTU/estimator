cmake_minimum_required(VERSION 3.0.2)
project(estimator)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

add_message_files(
FILES
AprilTagDetection.msg
AprilTagDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include

  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs message_generation message_runtime

  # DEPENDS system_lib
)

include_directories(

  # include
  ${catkin_INCLUDE_DIRS}
  ./src/libs
  ./include
  ${PROJECT_SOURCE_DIR}/include
)

ADD_EXECUTABLE(estimator src/estimator.cpp)
TARGET_LINK_LIBRARIES(estimator ${catkin_LIBRARIES})

# ADD_EXECUTABLE(translate_xsens_msg src/translate_xsens_msg.cpp)
# TARGET_LINK_LIBRARIES(translate_xsens_msg ${catkin_LIBRARIES})

# TARGET_LINK_LIBRARIES(imu_listener ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/src/libs/libjakaAPI.so)
ADD_DEPENDENCIES(estimator ${${PROJECT_NAME}_EXPORTED_TARGETS})
