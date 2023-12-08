
# BEVDet-ROS-TensorRT

+ 本工程是基于[bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp)魔改
+ 为了快速实现ros版本，只修改了`/test/demo_bevdet.cpp`中的`TestSample`函数
+ 仅将`TestSample`改成ros回调函数，导致代码很乱,可读性比较差, 比如读取参数实现方式不统一, 类只封装了推理函数等问题
+ 有时间再整理更新代码风格, 代码和人一个能跑就行`^_^`。
+ 再次声明，本工程所有代码都是参考大佬们的开源项目，如发现代码有雷同，都是本人引用大佬们的。
+ **感谢大佬们的开源工作**。

# 1 onnx2engine

1. 拉取源码
~~~python
mkdir -p bev_ws/src
cd bev_ws/src
git clone https://github.com/linClubs/BEVDet-ROS-TensorRT.git
~~~

2. onnx2engine

The onnx folder can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1NGUd6PphfcceHYQ1iKVxog?pwd=f937)


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
roslaunch bevdet bevdet_node.launch

# 4 播放数据集
rosbag play nus.bag
~~~

+ 测试数据建议直接下载[Baidu Netdisk](https://pan.baidu.com/s/1f3nUnHa_4cd6FsRTV8YhkA?pwd=rjim)rosbag包, 经本人生成的包，稳的1p。

---

# 3 传感器实时数据

+ **一定要标定**（相机内参， 雷达到相机的外参）, 不然置信度极低
+ **一定要训练模型** 自己场景数据需要重新训练模型
+ 传感器数据直接通过ros发布, 就能实时接本工程并上车

# 4 待提升
1. 图像传输可以压缩
2. 因为读了6张图,可以多线程读取
3. 图像预处理可以用cuda加速，比如去畸, 仿射变换等操作
4. 可以删掉可视化, 去掉雷达数据
5. 后处理采用cuda加速
6. ...待续
7. ...待续
...

---

+ bev感知交流Q群-472648720, 欢迎各位小伙伴进群一起学习讨论bev相关知识！！！^_^

<p align="center">
  <img src="1.jpg" width="200" height="200" />
</p>



---

# 5 普通编译

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