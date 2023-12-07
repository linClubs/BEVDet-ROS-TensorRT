
# 1 onnx2engine

1. 拉取源码
~~~python
mkdir -p bev_ws/src
cd bev_ws/src
git clone https://github.com/linClubs/BEVDet-ROS-TensorRT.git
~~~

2. onnx2engine
~~~python
pip install onnx ruamel.yaml==0.17.32

#onnx2engine
cd BEVDet-ROS-TensorRT
python tools/export_engine.py ./ckpts/lt_d.yaml ./ckpts/img_stage_lt_d.onnx ./ckpts/bev_stage_lt_d.onnx --postfix=_lt_d --fp16 true
~~~

# 2 ros编译运行
~~~python
# 1. 编译
cd bev_ws 
catkin make

# 2 工作空间生效
source devel/setup.bash

# 3 运行
ros
~~~






# 2 普通编译

~~~python
# 1 创建build目录并进入该目录
mkdir build && cd build

# 2 编译 # 第1次make报错如下，就再次make就能通过编译
cmake .. && make 

Error copying file (if different) from "/home/lin/code/bevdet-tensorrt-cpp/build/CMakeFiles/bevdemo.dir/src/bevdemo_generated_postprocess.cu.o.depend.tmp" to "/home/lin/code/bevdet-tensorrt-cpp/build/CMakeFiles/bevdemo.dir/src/bevdemo_generated_postprocess.cu.o.depend".
CMake Error at bevdemo_generated_postprocess.cu.o.RELEASE.cmake:246 (message):
  Error generating
  /home/lin/code/bevdet-tensorrt-cpp/build/CMakeFiles/bevdemo.dir/src/./bevdemo_generated_postprocess.cu.o

# 错误2：
AttributeError: 
"load()" has been removed, use

  yaml = YAML(typ='rt')
  yaml.load(...)
改正：
pip install ruamel.yaml==0.17.32

# 错误3
报libffi.so.7的错，禁用掉conda
~~~



~~~python
# 1 运行  生成sample0/sample0_lidarbox.txt检测结果
cd build
./bevdemo ../configure.yaml

# 我的运行
./01test ../configure.yaml

# 可视化
cd ../tools
python viewer.py --config ../configure.yaml 
~~~