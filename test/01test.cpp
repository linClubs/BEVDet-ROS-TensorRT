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

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void Getinfo(void) {
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


void TestNuscenes(YAML::Node &config){
    size_t img_N = config["N"].as<size_t>();
    int img_w = config["W"].as<int>();
    int img_h = config["H"].as<int>();
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
    read_sample(imgs_file, imgs_data);
    
    // opencv读取
    // read_sample_cv(imgs_file, imgs_data);

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

    for(auto box : lidar_boxes)
    {
        std::cout << box.label << std::endl;
        std::cout << box.x << std::endl;
        std::cout << box.y << std::endl;
        std::cout << box.z << std::endl;
        std::cout << box.l << std::endl;
        std::cout << box.w << std::endl;
        std::cout << box.h << std::endl;
        std::cout << box.r << std::endl;
        std::cout << box.score << std::endl;
        break;
    }
    

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