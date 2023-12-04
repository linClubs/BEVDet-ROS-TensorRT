#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>
#include <map>

#include <yaml-cpp/yaml.h>
#include "bevdet.h"
#include "cpu_jpegdecoder.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>


#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>  // msg2pcl

#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>


using std::chrono::duration;
using std::chrono::high_resolution_clock;

typedef pcl::PointXYZI PointT;

// 打印网络信息
void Getinfo(void);
// # box转txt文件
void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel);
// box的坐标从ego系变到雷达系
void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        std::vector<Box> &lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans);
// ros函数,对TestSample做了封装,改成回调传参
class RosNode
{
private:

    std::string pkg_path_;
    YAML::Node config_;
    size_t img_N_;
    int img_w_; 
    int img_h_;
    
    // 模型配置文件路径 
    std::string model_config_;
    
    // 权重文件路径 图像部分 bev部分
    std::string imgstage_file_;
    std::string bevstage_file_;
   
    // 相机的内外配置参数
    YAML::Node camconfig_; 
    
    // 结果保存文件
    std::string output_lidarbox_;

    YAML::Node sample_;

    std::vector<std::string> imgs_file_;
    std::vector<std::string> imgs_name_;

    camsData sampleData_;
    std::shared_ptr<BEVDet> bevdet;

    uchar* imgs_dev_ = nullptr; 

    ros::NodeHandle n_;
    ros::Subscriber sub_img_;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_boxes_;

public:
    RosNode();
    ~RosNode();
    void callback(const sensor_msgs::PointCloud2ConstPtr& msg);
};