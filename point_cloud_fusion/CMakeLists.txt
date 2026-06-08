cmake_minimum_required(VERSION 3.8)
project(point_cloud_fusion LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find MPI before PCL/VTK to avoid CMake import errors
find_package(MPI REQUIRED COMPONENTS C CXX)

# Option to require CUDA (fail build if not found)
option(REQUIRE_CUDA "Fail build if CUDA is not found" OFF)

# Check if CUDA was explicitly disabled during Docker build (e.g., ARM64)
set(CUDA_UNAVAILABLE_MARKER "/opt/ros-cmake-cache/cuda-unavailable")
if(EXISTS "${CUDA_UNAVAILABLE_MARKER}")
  message(STATUS "CUDA marked as unavailable (ARM64 or similar). Skipping CUDA detection.")
  set(CUDA_EXPLICITLY_UNAVAILABLE TRUE)
else()
  set(CUDA_EXPLICITLY_UNAVAILABLE FALSE)
endif()

# Auto-detect CUDA support
include(CheckLanguage)
if(NOT CUDA_EXPLICITLY_UNAVAILABLE)
  check_language(CUDA)
endif()

# Handle case where CMAKE_CUDA_COMPILER is explicitly set but doesn't exist
if(DEFINED CMAKE_CUDA_COMPILER AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  if(CUDA_EXPLICITLY_UNAVAILABLE)
    message(STATUS "Ignoring CMAKE_CUDA_COMPILER on platform without CUDA support")
  elseif(REQUIRE_CUDA)
    message(FATAL_ERROR "REQUIRE_CUDA is ON and CMAKE_CUDA_COMPILER was set to "
                        "'${CMAKE_CUDA_COMPILER}' but the compiler does not exist. "
                        "Ensure CUDA Toolkit is installed.")
  else()
    message(STATUS "Specified CUDA compiler not found: ${CMAKE_CUDA_COMPILER}")
  endif()
  unset(CMAKE_CUDA_COMPILER)
  unset(CMAKE_CUDA_COMPILER CACHE)
endif()

if(CMAKE_CUDA_COMPILER)
  option(ENABLE_CUDA "Enable CUDA acceleration" ON)
else()
  option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
endif()

if(ENABLE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA requested but no CUDA compiler found. Disabling CUDA support.")
    set(ENABLE_CUDA OFF)
  else()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
  endif()
else()
  message(STATUS "CUDA support disabled (no CUDA compiler detected or ENABLE_CUDA=OFF)")
endif()

find_package(ament_cmake REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(point_cloud_transport)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tracetools REQUIRED)

set(TARGET_NAME point_cloud_fusion_component)

# Source files
set(SOURCES src/point_cloud_fusion.cpp)

# Add CUDA sources if enabled
if(ENABLE_CUDA)
  list(APPEND SOURCES src/point_cloud_fusion_cuda.cu)
  set_source_files_properties(src/point_cloud_fusion_cuda.cu PROPERTIES LANGUAGE CUDA)
endif()

add_library(${TARGET_NAME} SHARED ${SOURCES})

# Set CUDA architectures if CUDA is enabled
if(ENABLE_CUDA)
  set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Native code for common architectures + PTX for forward compatibility
    # 70 = Volta (V100), 75 = Turing (RTX 20xx), 86 = Ampere (RTX 30xx, A series)
    # 89 = Ada Lovelace (RTX 40xx), 90 = Hopper (H100)
    # The "-virtual" suffix generates PTX code for JIT compilation on future GPUs
    CUDA_ARCHITECTURES "70;75;86;89;90;90-virtual"
  )
endif()

rclcpp_components_register_node(${TARGET_NAME}
  PLUGIN "point_cloud_fusion::PointCloudFusion"
  EXECUTABLE point_cloud_fusion
  EXECUTOR "MultiThreadedExecutor"
)

target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(${TARGET_NAME} PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(${TARGET_NAME}
  pcl_conversions
  pcl_ros
  point_cloud_transport
  rclcpp
  rclcpp_components
  sensor_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  tracetools
)

# Link CUDA libraries if enabled
if(ENABLE_CUDA)
  target_link_libraries(${TARGET_NAME}
    CUDA::cudart
    CUDA::cuda_driver
  )
endif()

install(TARGETS ${TARGET_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_clang_format_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/format/.clang-format)
  set(ament_cmake_clang_tidy_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/lint/.clang-tidy)
  set(ament_cmake_flake8_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.vscode/lint/ament_flake8.ini)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
