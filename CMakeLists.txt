cmake_minimum_required(VERSION 2.8.3)
project(ug_stereomatcher)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  geometry_msgs
  image_transport
  sensor_msgs
  stereo_msgs
  message_generation
  pcl_ros
)

find_package(PCL REQUIRED)

add_message_files(
	FILES
	foveatedstack.msg
  CamerasSync.msg
)

add_service_files(
  FILES
  GetDisparitiesGPU.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
   stereo_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs message_runtime pcl_ros stereo_msgs
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

### testing non fovea (working)
add_executable(UGdisp_image src/disparitymaps_viz/UG_dispImgTest.cpp)
add_dependencies(UGdisp_image ug_stereomatcher_generate_messages_cpp)
target_link_libraries(UGdisp_image boost_signals)
target_link_libraries(UGdisp_image ${catkin_LIBRARIES})
# testing fovea (working)
add_executable(UGFdisp_image src/disparitymaps_viz/UG_FdispImgTest.cpp)
add_dependencies(UGFdisp_image ug_stereomatcher_generate_messages_cpp)
target_link_libraries(UGFdisp_image boost_signals)
target_link_libraries(UGFdisp_image ${catkin_LIBRARIES})
###

add_executable(UGpoint_cloud src/pointcloud/getPointCloud.cpp)
add_dependencies(UGpoint_cloud ug_stereomatcher_generate_messages_cpp)
target_link_libraries(UGpoint_cloud boost_signals)
target_link_libraries(UGpoint_cloud ${catkin_LIBRARIES})      

add_executable(UGpublish_images src/utils/publish_images.cpp)
add_dependencies(UGpublish_images ug_stereomatcher_generate_messages_cpp)
target_link_libraries(UGpublish_images ${catkin_LIBRARIES})

# ######################### GPUmatcher
find_package(CUDA REQUIRED) 

# ######################### WARNING #########################
# Different gpus use different gencode flags. Uncomment accordingly
# ######################### WARNING #########################

# Setiings below are for Geforce GTX 970 and most likely for 900 series
# Compatible with CUDA 8.0
SET(CUDA_NVCC_FLAGS " -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50")

# ###########################################################
# ###########################################################
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

ADD_DEFINITIONS(-DCUDA_BUILD)
SET(CMAKE_CXX_FLAGS "-O3")

include_directories(/usr/local/cuda/samples/common/inc)

CUDA_ADD_LIBRARY(MatchLib src/gpu_matcher/MatchLib.cu)
add_library(MatchGPULib src/gpu_matcher/MatchGPULib.cpp)
add_library(convolutionSeparable_gold src/gpu_matcher/convolutionSeparable_gold.cpp)

target_link_libraries(MatchGPULib ${OpenCV_LIBRARIES} )
target_link_libraries(convolutionSeparable_gold ${OpenCV_LIBRARIES})

target_link_libraries(MatchGPULib MatchLib convolutionSeparable_gold)

target_link_libraries(MatchLib ${OpenCV_LIBRARIES})


add_executable(UG_matcher_gpu src/gpu_matcher/UG_GPU_matcher.cpp)
add_dependencies(UG_matcher_gpu ug_stereomatcher_generate_messages_cpp)
target_link_libraries(UG_matcher_gpu MatchGPULib ${catkin_LIBRARIES})


