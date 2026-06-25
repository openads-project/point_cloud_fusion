# point_cloud_fusion

<p align="center">
  <a href="https://github.com/openads-project"><img src="https://img.shields.io/badge/OpenADS-f5ff01"/></a>
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/openads-project/point_cloud_fusion/releases/latest"><img src="https://img.shields.io/github/v/release/openads-project/point_cloud_fusion"/></a>
  <a href="https://github.com/openads-project/point_cloud_fusion/blob/main/LICENSE"><img src="https://img.shields.io/github/license/openads-project/point_cloud_fusion"/></a>
  <br>
  <a href="https://github.com/openads-project/point_cloud_fusion/actions/workflows/docker-ros.yml"><img src="https://github.com/openads-project/point_cloud_fusion/actions/workflows/docker-ros.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_fusion/actions/workflows/compose-oci.yml"><img src="https://github.com/openads-project/point_cloud_fusion/actions/workflows/compose-oci.yml/badge.svg"/></a>
  <a href="https://openads-project.github.io/point_cloud_fusion"><img src="https://github.com/openads-project/point_cloud_fusion/actions/workflows/docs.yml/badge.svg"/></a>
  <a href="https://github.com/openads-project/point_cloud_fusion/actions/workflows/consistency.yml"><img src="https://github.com/openads-project/point_cloud_fusion/actions/workflows/consistency.yml/badge.svg"/></a>
</p>

This repository provides a ROS 2 point cloud fusion node. The node subscribes to multiple `sensor_msgs/msg/PointCloud2` streams, transforms them into a common target frame, and publishes one fused point cloud through `point_cloud_transport`.

The package supports CPU processing and optional CUDA accelerated fusion.

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>


> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).


## 🚀 Quick Start

  <video src="https://github.com/user-attachments/assets/e19c19f9-0e47-4ca5-a893-c3d78b458296" width="720" style="max-width: 100%;">
  </video>

1. Launch the [`demo/docker-compose.yml`](demo/docker-compose.yml) setup. This
   starts point cloud playback, point cloud fusion, RViz, and the runtime parameter GUI:

    ```bash
    cd demo
    xhost +local: # allow GUI forwarding from containers
    docker compose up
    ```

3. The `ros-parameter-gui` service starts `rqt` with the `rqt_reconfigure` plugin. You may use it to inspect and adjust the parameters of the `point_cloud_fusion` node while the demo is active.

4. Stop the demo with `Ctrl+C`, remove its containers, and revoke GUI access:

    ```bash
    docker compose down
    xhost -local:
    ```

## 💻 Development

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


## 📝 Documentation

Package and node interfaces are documented in the respective package READMEs listed below. Implementation details are found in the [Source Code Documentation](https://fb-fi.pages.ika.rwth-aachen.de/its-modules/perception/point_cloud_fusion).

| Package | Description |
| --- | --- |
| [point_cloud_fusion](point_cloud_fusion/README.md) | Fuses multiple point clouds into a single point cloud with common target frame. |

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

Development and maintenance of this repository are supported by the following projects. We acknowledge the funding of the respective institutions.

| Project | Funding Institution | Grant Number |
| --- | --- | --- |
| [AIGGREGATE](https://aiggregate.eu/) | 🇪🇺 European Union | 101202457 |

<p>
  <img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_funded_en.jpg" height=70>
</p>

<sup><sub>Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor CINEA can be held responsible for them.</sup></sup>
