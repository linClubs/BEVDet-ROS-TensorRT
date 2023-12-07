
# BEVDet by TensorRT, C++, ROS

This repository contains source code and models for BEVDet online real-time inference using CUDA, TensorRT, ROS1 & C++.


# 1 ENV

- **ubuntu-20.0、CUDA 11.3、cuDNN 8.6.0、TensorRT-8.5**
- **yaml-cpp、Eigen3、libjpeg**

# 3 Build

~~~python
mkdir -p bev_ws/src
cd bev_ws/src
git clone https://github.com/linClubs/BEVDet-ROS-TensorRT.git
cd ..
catkin_make
source devel/setup.bash
~~~

# 4 Run

Generate the TensorRT engine reference to [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp)

The rosbag folder can be downloaded from [Google drive]() or [Baidu Netdisk]()


~~~python

# 1. start bevdet_node
roslaunch bevdet bevdet_node.launch

# 2  play data
rosbag play nus.bag
~~~





## References
- [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [nuScenes](https://www.nuscenes.org/)
