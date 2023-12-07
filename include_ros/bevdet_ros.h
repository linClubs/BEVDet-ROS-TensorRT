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

// 消息同步
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h> // 时间接近


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

void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        jsk_recognition_msgs::BoundingBoxArrayPtr lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans,  float score_thre);


// 添加从opencv的Mat转换到std::vector<char>的函数 读取图像 cv2data 
int cvToArr(cv::Mat img, std::vector<char> &raw_data)
{
    if (img.empty())
    {
        std::cerr << "image is empty. " << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<u_char> raw_data_;
    cv::imencode(".jpg", img, raw_data_);
    raw_data = std::vector<char>(raw_data_.begin(), raw_data_.end());
    return EXIT_SUCCESS;
}

int cvImgToArr(std::vector<cv::Mat> &imgs, std::vector<std::vector<char>> &imgs_data)
{
    imgs_data.resize(imgs.size());

    for(size_t i = 0; i < imgs_data.size(); i++)
    {   
        if(cvToArr(imgs[i], imgs_data[i]))
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
 
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
    std::shared_ptr<BEVDet> bevdet_;

    uchar* imgs_dev_ = nullptr; 

    ros::NodeHandle n_;
    // 发布点云和盒子
    ros::Publisher pub_cloud_;
    ros::Publisher pub_boxes_;
    
    // 订阅点云和图像
    float score_thre_;
    std::string topic_cloud_;
    std::string topic_img_f_, topic_img_fl_, topic_img_fr_;
    std::string topic_img_b_, topic_img_bl_, topic_img_br_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_f_img_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_fl_img_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_fr_img_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_b_img_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_bl_img_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_br_img_; 

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::PointCloud2, 
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
        std::shared_ptr<Sync> sync_;

public:
    RosNode();
    ~RosNode();
    // void callback(const sensor_msgs::ImageConstPtr& msg);
    void getRosParam();
    void callback(const sensor_msgs::PointCloud2ConstPtr& msg_cloud, 
    const sensor_msgs::ImageConstPtr& msg_fl_img,
    const sensor_msgs::ImageConstPtr& msg_f_img,
    const sensor_msgs::ImageConstPtr& msg_fr_img,
    const sensor_msgs::ImageConstPtr& msg_bl_img,
    const sensor_msgs::ImageConstPtr& msg_b_img,
    const sensor_msgs::ImageConstPtr& msg_br_img);
};