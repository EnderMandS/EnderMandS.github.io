---
layout: post
title: "Jetson Orin Nano conda torch ros env setup"
date: 2025-10-10 17:37:00 +0800
categories: [Deploy, Nvidia]
---

# Jetson Orin envrionment setup

在 Jetson Orin Nano上配置conda的torch、torchvision、Open3D、ROS环境

## Conda

安装Miniconda
``` shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
```

安装mamba
``` shell
conda install -n base -c conda-forge mamba
```

## Torch

安装jtop
``` shell
sudo pip3 install -U jetson-stats
```

使用jtop检查jetpack、L4T等版本

以下以 jetpack 5.1.1、L4T 35.3.1为例，检查

``` shell
mamba create -n torch python=3.8
pip install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp311-cp311-linux_aarch64.whl

```

## Torchvision

``` shell
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev libpng-dev
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision 
cd torchvision
export BUILD_VERSION=0.15.1
pip install pillow==9.5.0
python3 setup.py install --user
```

## ROS

安装公共依赖
``` shell
pip install rospkg catkin_pkg empy
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
```

编辑`$CONDA_PREFIX/etc/conda/activate.d`，添加以下代码到激活脚本
``` shell
#!/usr/bin/env zsh
export ROS_DISTRO=noetic
source /opt/ros/$ROS_DISTRO/setup.zsh
export PYTHONPATH=/opt/ros/$ROS_DISTRO/lib/python3.8/site-packages:$PYTHONPATH
```

编辑`$CONDA_PREFIX/etc/conda/deactivate.d`，添加以下代码到卸载脚本
``` shell
#!/usr/bin/env zsh
unset ROS_DISTRO
```

## Open3D


检查`cmake --version`不低于3.18，若低于则更新
``` shell
wget https://cmake.org/files/v3.24/cmake-3.24.0-linux-aarch64.sh
sudo bash cmake-3.24.0-linux-aarch64.sh --skip-license --prefix=/opt/cmake/3.24
rm cmake-3.24.0-linux-aarch64.sh
/opt/cmake/3.24/bin/cmake --version
```

安装
``` shell
sudo apt install -y \
  git build-essential ninja-build cmake ccache \
  libeigen3-dev libglew-dev libtiff-dev libjpeg-dev libpng-dev \
  libopenexr-dev libfreetype6-dev libxcb1-dev libx11-dev \
  libglu1-mesa-dev libxi-dev libxrandr-dev libpthread-stubs0-dev \
  python3-dev python3-pip protobuf-compiler libprotobuf-dev \
  libglfw3-dev libssl-dev libcurl4-openssl-dev
mkdir Open3d && cd Open3D
git clone --branch v0.18.0 --recursive https://github.com/isl-org/Open3D.git
cd Open3D
mkdir build && cd build
/opt/cmake/3.24/bin/cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CUDA_MODULE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DBUILD_PYTHON_MODULE=ON \
  -DBUILD_PYTORCH_OPS=ON \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_GUI=OFF \
  -DBUILD_RENDERING=OFF \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_INSTALL_PREFIX=$HOME/pkg/open3d_install
make -j
```

## Test

``` python
import rospy
import torch
import torchvision
import open3d as o3d
```

# 参考资料
- [https://blog.csdn.net/cau_weiyuhu/article/details/131056649](https://blog.csdn.net/cau_weiyuhu/article/details/131056649)

- [https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)


