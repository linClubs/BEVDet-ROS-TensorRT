#include "bevdet_ros.h"

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
                                        const Eigen::Translation3f &lidar2ego_trans)
{
    
    for(size_t i = 0; i < ego_boxes.size(); i++)
    {
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

void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        jsk_recognition_msgs::BoundingBoxArrayPtr lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans, float score_thre = 0.2)
{
    
    for(auto b : ego_boxes)
    {   
        if(b.score < score_thre)
            continue;
        jsk_recognition_msgs::BoundingBox box;

        Eigen::Vector3f center(b.x, b.y, b.z + b.h/2.);
        // Eigen::Vector3f center(b.x, b.y, b.z);

        center -= lidar2ego_trans.translation();         
        center = lidar2ego_rot.inverse().matrix() * center;
        
        b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        Eigen::Quaterniond q(Eigen::AngleAxisd(b.r, Eigen::Vector3d(0, 0, 1)));

        box.pose.position.x = center.x();
        box.pose.position.y = center.y();
        box.pose.position.z = center.z();
        box.pose.orientation.x = q.x();
        box.pose.orientation.y = q.y();
        box.pose.orientation.z = q.z();
        box.pose.orientation.w = q.w();

        // 长宽高不变
        box.dimensions.x = b.l;
        box.dimensions.y = b.w;
        box.dimensions.z = b.h;
        
        box.label = b.label;
        box.header.frame_id = "map";
        box.header.stamp = ros::Time::now();

        lidar_boxes->boxes.emplace_back(box);
    }
}

RosNode::RosNode()
{
    getRosParam(); // 获取ros参数
    std::string config_path = pkg_path_ + "/cfgs/config.yaml";
    config_ = YAML::LoadFile(config_path);
    printf("Successful load config : %s!\n", config_path.c_str());
    bool testNuscenes = config_["TestNuscenes"].as<bool>();

    img_N_ = config_["N"].as<size_t>();  // 图片数量 6
    img_w_ = config_["W"].as<int>();        // H: 900
    img_h_ = config_["H"].as<int>();        // W: 1600
    
    // 模型配置文件路径 
    model_config_ = pkg_path_ + "/" + config_["ModelConfig"].as<std::string>();
    
    // 权重文件路径 图像部分 bev部分
    imgstage_file_ =pkg_path_ + "/" +  config_["ImgStageEngine"].as<std::string>();
    bevstage_file_ =pkg_path_ +"/" +  config_["BEVStageEngine"].as<std::string>();
    
    // 相机的内参配置参数
    camconfig_ = YAML::LoadFile(pkg_path_ +"/" + config_["CamConfig"].as<std::string>()); 
    // 结果保存文件
    output_lidarbox_ = pkg_path_ +"/" + config_["OutputLidarBox"].as<std::string>();
    
    sample_ = config_["sample"];

    for(auto file : sample_)
    {
        imgs_file_.push_back(pkg_path_ +"/"+ file.second.as<std::string>());
        imgs_name_.push_back(file.first.as<std::string>()); 
    }

    // 读取图像参数
    sampleData_.param = camParams(camconfig_, img_N_, imgs_name_);
    
    // 模型配置文件，图像数量，cam内参，cam2ego的旋转和平移，模型权重文件
    bevdet_ = std::make_shared<BEVDet>(model_config_, img_N_, sampleData_.param.cams_intrin, 
                sampleData_.param.cams2ego_rot, sampleData_.param.cams2ego_trans, 
                                                    imgstage_file_, bevstage_file_);
    
    
    // gpu分配内参， cuda上分配6张图的大小 每个变量sizeof(uchar)个字节，并用imgs_dev指向该gpu上内存, sizeof(uchar) =1
    CHECK_CUDA(cudaMalloc((void**)&imgs_dev_, img_N_ * 3 * img_w_ * img_h_ * sizeof(uchar)));

    
    
    pub_cloud_ = n_.advertise<sensor_msgs::PointCloud2>("/points_raw", 10); 
    pub_boxes_ = n_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/boxes", 10);   
    
    sub_cloud_.subscribe(n_, topic_cloud_, 10);
    sub_f_img_.subscribe(n_, topic_img_f_, 10);
    sub_b_img_.subscribe(n_, topic_img_b_, 10);

    sub_fl_img_.subscribe(n_,topic_img_fl_, 10);
    sub_fr_img_.subscribe(n_,topic_img_fr_, 10);
    
    sub_bl_img_.subscribe(n_,topic_img_bl_, 10);
    sub_br_img_.subscribe(n_,topic_img_br_, 10);
  
    sync_ = std::make_shared<Sync>( MySyncPolicy(10), sub_cloud_, 
    sub_fl_img_, sub_f_img_, sub_fr_img_,
    sub_bl_img_ ,sub_b_img_, sub_br_img_); 
  
    sync_->registerCallback(boost::bind(&RosNode::callback,this, _1, _2, _3, _4, _5, _6,_7)); // 绑定回调函数
  
}

void RosNode::getRosParam()
{   
    pkg_path_ = ros::package::getPath("bevdet");
    
    n_.param<float>("score_thre", score_thre_ , 0.2);
  
    n_.param<std::string>("topic_cloud", topic_cloud_, "/lidar_top");
    
    n_.param<std::string>("topic_img_f", topic_img_f_, "/cam_front/raw");
    n_.param<std::string>("topic_img_b", topic_img_b_, "/cam_back/raw");
    
    n_.param<std::string>("topic_img_fl", topic_img_fl_, "/cam_front_left/raw");
    n_.param<std::string>("topic_img_fr", topic_img_fr_, "/cam_front_right/raw");
    
    n_.param<std::string>("topic_img_bl", topic_img_bl_, "/cam_back_left/raw");
    n_.param<std::string>("topic_img_br", topic_img_br_, "/cam_back_right/raw");

}

void RosNode::callback(const sensor_msgs::PointCloud2ConstPtr& msg_cloud, 
    const sensor_msgs::ImageConstPtr& msg_fl_img,
    const sensor_msgs::ImageConstPtr& msg_f_img,
    const sensor_msgs::ImageConstPtr& msg_fr_img,
    const sensor_msgs::ImageConstPtr& msg_bl_img,
    const sensor_msgs::ImageConstPtr& msg_b_img,
    const sensor_msgs::ImageConstPtr& msg_br_img)
{   

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    
    pcl::fromROSMsg(*msg_cloud, *cloud);
   
    cv::Mat img_fl, img_f, img_fr, img_bl, img_b, img_br;
    std::vector<cv::Mat> imgs;
    img_fl = cv_bridge::toCvShare(msg_fl_img , "bgr8")->image;
    img_f  = cv_bridge::toCvShare(msg_f_img, "bgr8")->image;
    img_fr = cv_bridge::toCvShare(msg_fr_img, "bgr8")->image;
    img_bl = cv_bridge::toCvShare(msg_bl_img , "bgr8")->image;
    img_b  = cv_bridge::toCvShare(msg_b_img, "bgr8")->image;
    img_br = cv_bridge::toCvShare(msg_br_img, "bgr8")->image;

    imgs.emplace_back(img_fl);
    imgs.emplace_back(img_f);
    imgs.emplace_back(img_fr);
    imgs.emplace_back(img_bl);
    imgs.emplace_back(img_b);
    imgs.emplace_back(img_br);

    std::vector<std::vector<char>> imgs_data;
    // Mat2Vetor<char>  图像数据读取到cpu    
    cvImgToArr(imgs, imgs_data);
    
    // 拷贝从cpu上imgs_data拷贝到gpu上imgs_dev
    // std::vector<std::vector<char>> imgs_data 并进行通道转换
    decode_cpu(imgs_data, imgs_dev_, img_w_, img_h_);

    // uchar *imgs_dev 已经到gpu上了数据
    sampleData_.imgs_dev = imgs_dev_;

    std::vector<Box> ego_boxes;
    ego_boxes.clear();
    float time = 0.f;
    // 测试推理  图像数据, boxes，时间
    bevdet_->DoInfer(sampleData_, ego_boxes, time);
    
    // std::vector<Box> lidar_boxes;
    jsk_recognition_msgs::BoundingBoxArrayPtr lidar_boxes(new jsk_recognition_msgs::BoundingBoxArray);
    
    lidar_boxes->boxes.clear();
    // box从ego坐标变化到雷达坐标
    Egobox2Lidarbox(ego_boxes, lidar_boxes, sampleData_.param.lidar2ego_rot, 
                                            sampleData_.param.lidar2ego_trans);

    lidar_boxes->header.frame_id = "map";
    lidar_boxes->header.stamp = ros::Time::now();
    
    pub_boxes_.publish(*lidar_boxes);
    
    sensor_msgs::PointCloud2 msg_cloud_new;
    pcl::toROSMsg(*cloud, msg_cloud_new);

    msg_cloud_new.header.frame_id = "map";
    msg_cloud_new.header.stamp = ros::Time::now();
    pub_cloud_.publish(msg_cloud_new);
}


RosNode::~RosNode()
{
    delete imgs_dev_;
}


int main(int argc, char **argv)
{   
    ros::init(argc, argv, "bevdet_node");
    // Getinfo(); # 打印信息
    
    auto bevdet_node = std::make_shared<RosNode>();
    ros::spin();
    return 0;
}