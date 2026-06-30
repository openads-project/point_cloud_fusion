#!/bin/bash
set -euo pipefail

# Detect architecture
ARCH=$(dpkg --print-architecture)

# Create a CMake cache file for architecture-specific settings
CMAKE_CACHE_DIR="/opt/ros-cmake-cache"
mkdir -p "$CMAKE_CACHE_DIR"

apt-get update
apt-get install -y --no-install-recommends libomp-18-dev

if [ "$ARCH" = "amd64" ]; then
    echo "Installing CUDA 12.8 build/runtime packages for amd64..."

    # Add NVIDIA CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    # Install only the CUDA packages required by this repo:
    apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 \
        cuda-cudart-dev-12-8 \
        cuda-cudart-12-8 \
        cuda-driver-dev-12-8

    CUDA_ROOT="/usr/local/cuda-12.8"
    CUDA_LIB_DIR="${CUDA_ROOT}/targets/x86_64-linux/lib"
    if [ ! -d "$CUDA_LIB_DIR" ]; then
        echo "ERROR: Expected CUDA library directory not found: $CUDA_LIB_DIR"
        exit 1
    fi

    # Expose CUDA tools for interactive shells
    echo "export PATH=${CUDA_ROOT}/bin:\$PATH" >> /etc/bash.bashrc

    # Register CUDA runtime path with the dynamic linker for non-interactive runtime
    echo "$CUDA_LIB_DIR" > /etc/ld.so.conf.d/cuda.conf
    ldconfig

    # Create symlink so CMake can find CUDA without full path
    ln -sf "$CUDA_ROOT" /usr/local/cuda

    # Create marker file to indicate CUDA is available
    rm -f "$CMAKE_CACHE_DIR/cuda-unavailable"
    touch "$CMAKE_CACHE_DIR/cuda-available"

    echo "CUDA 12.8 package installation complete!"
else
    echo "Skipping CUDA installation on $ARCH (not supported)"
    echo "Building CPU-only version"
    
    # Create marker file to indicate CUDA is NOT available
    rm -f "$CMAKE_CACHE_DIR/cuda-available"
    touch "$CMAKE_CACHE_DIR/cuda-unavailable"
fi

# TODO: remove once [https://github.com/ros2/ros2_tracing/issues/211] is solved in released version
# overwrite released ros2_tracing packages with fork to support
# 'message-link instrumentation' and 'dual-session mode' in jazzy
cd /docker-ros/ws
git clone --branch jazzy-ika https://github.com/RaphvK/ros2_tracing.git src/ros2_tracing
rosdep update && rosdep install -y -i --from-paths src/ros2_tracing/tracetools src/ros2_tracing/tracetools_launch
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
colcon build --packages-up-to tracetools tracetools_launch --allow-overriding tracetools --allow-overriding tracetools_launch
rm -r src/ros2_tracing log/ build/ 
