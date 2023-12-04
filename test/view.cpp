#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cassert>

#include <yaml-cpp/yaml.h>
#include "bevdet.h"
#include "cpu_jpegdecoder.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

#include <map>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

typedef pcl::PointXYZI PointT;

// [255, 215, 0],   // 金黄色
//            [160, 32, 240],  // 紫色
//            [3, 168, 158],   // 锰蓝
//            [255, 0, 0],     // 红色
//            [255, 97, 0],    # 橙色
//            [0, 201, 87],    # 翠绿色
//            [255, 153, 153], # 粉色
//            [255, 255, 0],    # 黄色
//            [0, 0, 0],       # 黑色

std::map< int, std::vector<int>> colormap { 
            {0, {0, 0, 255}},  // dodger blue 
            {1, {0, 201, 87}},   // 青色
            {2, {0, 201, 87}},
            {3, {160, 32, 240}},
            {4, {3, 168, 158}},
            {5, {255, 0, 0}},
            {6, {255, 97, 0}},
            {7, {30,  0, 255}},
            {8, {255, 0, 0}},
            {9, {0, 0, 255}},
            {10, {0, 0, 0}}
};



void Getinfo(void) 
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);
    }
    printf("\n");
}


void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel=false) 
{
  std::ofstream out_file;
  out_file.open(file_name, std::ios::out);
  if (out_file.is_open()) {
    for (const auto &box : boxes) {
      out_file << box.x << " ";
      out_file << box.y << " ";
      out_file << box.z << " ";
      out_file << box.l << " ";
      out_file << box.w << " ";
      out_file << box.h << " ";
      out_file << box.r << " ";
      if(with_vel)
      {
        out_file << box.vx << " ";
        out_file << box.vy << " ";
      }
      out_file << box.score << " ";
      out_file << box.label << "\n";
    }
  }
  out_file.close();
  return;
};


void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        std::vector<Box> &lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans){
    
    for(size_t i = 0; i < ego_boxes.size(); i++){
        Box b = ego_boxes[i];
        Eigen::Vector3f center(b.x, b.y, b.z);
        center -= lidar2ego_trans.translation();
        center = lidar2ego_rot.inverse().matrix() * center;
        b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        b.x = center.x();
        b.y = center.y();
        b.z = center.z();
        lidar_boxes.push_back(b);
    }
}


void TestNuscenes(YAML::Node &config)
{
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
    // 数据集路径
    std::string data_info_path = config["dataset_info"].as<std::string>();
    std::string model_config = config["ModelConfig"].as<std::string>();
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    std::string output_dir = config["OutputDir"].as<std::string>();
    std::vector<std::string> cams_name = config["cams"].as<std::vector<std::string>>();

    DataLoader nuscenes(img_N, img_h, img_w, data_info_path, cams_name);
    
    BEVDet bevdet(model_config, img_N, nuscenes.get_cams_intrin(), 
            nuscenes.get_cams2ego_rot(), nuscenes.get_cams2ego_trans(), imgstage_file,
            bevstage_file);

    std::vector<Box> ego_boxes;
    double sum_time = 0;
    int  cnt = 0;
    for(int i = 0; i < nuscenes.size(); i++)
    {
        ego_boxes.clear();
        float time = 0.f;
        bevdet.DoInfer(nuscenes.data(i), ego_boxes, time, i);
        if(i != 0)
        {
            sum_time += time;
            cnt++;
        }
        Boxes2Txt(ego_boxes, output_dir + "/bevdet_egoboxes_" + std::to_string(i) + ".txt", true);
    }
    printf("Infer mean cost time : %.5lf ms\n", sum_time / cnt);
}

// 测试一帧
void TestSample(YAML::Node &config)
{
    size_t img_N = config["N"].as<size_t>();  // 图片数量 6
    int img_w = config["W"].as<int>();        // H: 900
    int img_h = config["H"].as<int>();        // W: 1600
    
    // 模型配置文件 
    std::string model_config = config["ModelConfig"].as<std::string>();
    
    // 权重文件路径 图像部分 bev部分
    std::string imgstage_file = config["ImgStageEngine"].as<std::string>();
    std::string bevstage_file = config["BEVStageEngine"].as<std::string>();
    
    // 所有的内参配置参数
    YAML::Node camconfig = YAML::LoadFile(config["CamConfig"].as<std::string>()); 
    
    // 结果保存文件
    std::string output_lidarbox = config["OutputLidarBox"].as<std::string>();
    
    YAML::Node sample = config["sample"];

    std::vector<std::string> imgs_file;
    std::vector<std::string> imgs_name;
    
    for(auto file : sample)
    {
        imgs_file.push_back(file.second.as<std::string>());
        imgs_name.push_back(file.first.as<std::string>()); 
    }

    camsData sampleData;
    // 读取图像参数
    sampleData.param = camParams(camconfig, img_N, imgs_name);
    
    // 模型配置文件，图像数量，cam内参，cam2ego的旋转和平移，模型权重文件
    BEVDet bevdet(model_config, img_N, sampleData.param.cams_intrin, 
                sampleData.param.cams2ego_rot, sampleData.param.cams2ego_trans, 
                                                    imgstage_file, bevstage_file);

    // 图像数据
    std::vector<std::vector<char>> imgs_data;
    
    // // file读取到cpu
    // read_sample(imgs_file, imgs_data);
    
    // opencv读取BG  RGBHWC_to_BGRCHW
    read_sample_cv(imgs_file, imgs_data);

    uchar* imgs_dev = nullptr; 
    // gpu分配内参， cuda上分配6张图的大小 每个变量sizeof(uchar)个字节，并用imgs_dev指向该gpu上内存, sizeof(uchar) =1
    CHECK_CUDA(cudaMalloc((void**)&imgs_dev, img_N * 3 * img_w * img_h * sizeof(uchar)));
    
    // 拷贝从cpu上imgs_data拷贝到gpu上imgs_dev
    // std::vector<std::vector<char>> imgs_data 并进行通道转换
    decode_cpu(imgs_data, imgs_dev, img_w, img_h);

    

    // uchar *imgs_dev 已经到gpu上了数据
    sampleData.imgs_dev = imgs_dev;

    std::vector<Box> ego_boxes;
    ego_boxes.clear();
    float time = 0.f;

    // 测试推理  图像数据, boxes，时间
    bevdet.DoInfer(sampleData, ego_boxes, time);
    
    std::vector<Box> lidar_boxes;
    
    // box从ego坐标变化到雷达坐标
    Egobox2Lidarbox(ego_boxes, lidar_boxes, sampleData.param.lidar2ego_rot, 
                                            sampleData.param.lidar2ego_trans);

    
    // std::string file_name
    Boxes2Txt(lidar_boxes, output_lidarbox, false);
   
    ego_boxes.clear();
    // bevdet.DoInfer(sampleData, ego_boxes, time); // only for inference time

    // 加载点云可视化
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>); 
    std::string pcd_path = "/home/lin/code/bevdet-tensorrt-cpp/sample0/0.pcd";
    pcl::io::loadPCDFile(pcd_path, *cloud);

    pcl::visualization::PCLVisualizer viewer("points");
    viewer.setBackgroundColor(0, 0, 0); // 设置背景颜色setBackgroundColor

	// 根据点云的某个字段进行上色,常用的字段有：x,y,z,normal_x(X方向上的法线),normal_y,normal_z,rgb,curvature曲率
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> rgb(cloud, "z");
    viewer.addPointCloud(cloud, rgb, "sample cloud");


	// 设置点云大小
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	

	viewer.addCoordinateSystem(1); // 添加坐标系，并设置比例
	viewer.initCameraParameters(); // 通过设置照相机参数使得从默认的角度和方向观察点云
    
 
    int count = 0;
    for(auto box : lidar_boxes)
    {   
        if(box.score < 0.21)
            continue;
        float x = box.x;
        float y = box.y;
        float z = box.z;
        float w = box.w ;
        float h = box.h ;
        float l = box.l ;
        float yaw = box.r;
        int id = box.label;
        // 添加一个检测框
       

        pcl::PointCloud<pcl::PointXYZ> ps;
        
        float a[4] = {0.5, 0.5,-0.5,-0.5};
        float aa[2] = {0, 1};

        for (int i = 0; i < 2; i++)
        {
            float pz = aa[i] * h;
            for (int j = 0; j < 4; j++)
            {   
                // std::cout << a[j] << "," << a[(j+1) % 4] << std::endl << std::endl;
                float px = a[j] * l;
                float py = a[(j+1) % 4] * w;
                pcl::PointXYZ p(px, py, pz);
                ps.points.emplace_back(p);
            }
        }

        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        tf.translation() << x, y, z;
        tf.rotate (Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ()));
        
        pcl::transformPointCloud (ps, ps, tf);

        int b = colormap[id][0]; 
        int g = colormap[id][1];  
        int r = colormap[id][2];   
        // std::cout << r << g << b << std::endl;  

        viewer.addLine<pcl::PointXYZ>(ps.points[0], ps.points[1], r, g, b, std::to_string(count) + std::to_string(0));
        viewer.addLine<pcl::PointXYZ>(ps.points[1], ps.points[2], r, g, b, std::to_string(count) + std::to_string(1));
        viewer.addLine<pcl::PointXYZ>(ps.points[2], ps.points[3], r, g, b, std::to_string(count) + std::to_string(2));
        viewer.addLine<pcl::PointXYZ>(ps.points[3], ps.points[0], r, g, b, std::to_string(count) + std::to_string(3));
        viewer.addLine<pcl::PointXYZ>(ps.points[4], ps.points[5], r, g, b, std::to_string(count) + std::to_string(4));
        viewer.addLine<pcl::PointXYZ>(ps.points[5], ps.points[6], r, g, b, std::to_string(count) + std::to_string(5));
        viewer.addLine<pcl::PointXYZ>(ps.points[6], ps.points[7], r, g, b, std::to_string(count) + std::to_string(6));
        viewer.addLine<pcl::PointXYZ>(ps.points[7], ps.points[4], r, g, b, std::to_string(count) + std::to_string(7));
        viewer.addLine<pcl::PointXYZ>(ps.points[0], ps.points[4], r, g, b, std::to_string(count) + std::to_string(8));
        viewer.addLine<pcl::PointXYZ>(ps.points[1], ps.points[5], r, g, b, std::to_string(count) + std::to_string(9));
        viewer.addLine<pcl::PointXYZ>(ps.points[2], ps.points[6], r, g, b, std::to_string(count) + std::to_string(10));
        viewer.addLine<pcl::PointXYZ>(ps.points[3], ps.points[7], r, g, b, std::to_string(count) + std::to_string(11));
        
        std::string arrow_id = "arrow-"+ std::to_string(count);
        viewer.removeShape(arrow_id);

        pcl::PointXYZ start(x, y, z/2.), 
        end((ps.points[0].x + ps.points[1].x)/2., (ps.points[0].y + ps.points[1].y)/ 2, z/2.);
    
        viewer.addArrow<pcl::PointXYZ>(end, start, r, g, b, false, arrow_id);
        

        std::string txt_name, txt_id;
        txt_id = "txt-" + std::to_string(count);
        txt_name = std::to_string(id);
        viewer.removeText3D(txt_id);

        pcl::PointXYZ location(x, y, z + h); 
        viewer.addText3D(txt_name, location, 0.8, r, g, b, txt_id);


        count++;
    }

    viewer.spin(); //循环不断显示点云 

}

int main(int argc, char **argv)
{
    Getinfo();
    // if(argc < 2){
    //     printf("Need a configure yaml! Exit!\n");
    //     return 0;
    // }
    std::string config_file;
    if(argc = 1){
        config_file = "../configure.yaml";
    }

    YAML::Node config = YAML::LoadFile(config_file);

    printf("Successful load config : %s!\n", config_file.c_str());
    bool testNuscenes = config["TestNuscenes"].as<bool>();
    
    // 是否测试nuscenes数据集
    if(testNuscenes)
    {
        TestNuscenes(config);
    }
    else
    {   
        // 测试单个样例
        TestSample(config);
    }
    return 0;
}