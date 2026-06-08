# point_cloud_fusion

<p align="center">
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
</p>

This repository provides a ROS 2 point-cloud fusion node for automated driving perception stacks. The node subscribes to multiple `sensor_msgs/msg/PointCloud2` streams, transforms them into a common target frame, and publishes one fused point cloud through `point_cloud_transport`.

The package supports CPU processing and optional CUDA acceleration for high-throughput multi-LiDAR fusion.

<p align="center">
  <strong><a href="#-quick-start">Quick Start</a></strong> - <strong><a href="#-development">Development</a></strong> - <strong><a href="#-documentation">Documentation</a></strong>
</p>

> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).

## Quick Start

Run the node with the packaged launch file and parameter file.

```bash
ros2 launch point_cloud_fusion point_cloud_fusion.launch.py
```

Configure input topics, transport hints, synchronization, and output behavior in [`point_cloud_fusion/config/params.yml`](point_cloud_fusion/config/params.yml).

## Development

### Set up Development Environment

1. Clone the repository.
    ```bash
    git clone https://gitlab.ika.rwth-aachen.de/fb-fi/its-modules/perception/point_cloud_fusion.git
    ```
1. Initialize the [`.openads-dev-environment`](https://github.com/openads-project/openads-dev-environment) submodule containing development environment configuration.
    ```bash
    cd point_cloud_fusion
    git submodule update --init --recursive
    ```
1. Open the repository in [Visual Studio Code](https://code.visualstudio.com).
    ```bash
    code .
    ```
1. Install the recommended VS Code extensions.
    > *Ctrl+Shift+P / Extensions: Show Recommended Extensions / Install Workspace Recommended Extensions (Cloud Download Icon)*
1. Reopen the repository in a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers).
    > *Ctrl+Shift+P / Dev Containers: Rebuild and Reopen in Container*

### Build

> *Ctrl+Shift+B*

```bash
colcon build
```

### Run Tests

> *Ctrl+Shift+P / Tasks: Run Test Task*

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=1
colcon test
colcon test-result --verbose
```

## Documentation

Package and node interfaces are documented in the respective package READMEs listed below.

| Package | Description |
| --- | --- |
| [point_cloud_fusion](point_cloud_fusion/README.md) | Fuses multiple point clouds into one common frame |

## Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## Acknowledgements

Development and maintenance of this repository are supported by the Institute for Automotive Engineering (ika) at RWTH Aachen University.
