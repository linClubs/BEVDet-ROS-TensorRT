import numpy as np
from pypcd import pypcd

bin_path = "/home/lin/code/bevdet-tensorrt-cpp/sample0/0.bin"
points = np.fromfile(bin_path, dtype=np.float32, count=-1).reshape([-1, 5])
pcd_path = "/home/lin/code/bevdet-tensorrt-cpp/sample0/0.pcd"

points = points[:, :4]

meta_data = {
        'version': '0.7',
        'fields': ['x', 'y', 'z', 'intensity'],
        'size': [4, 4, 4, 4],
        'type': ['F', 'F', 'F', 'F'],
        'count': [1, 1, 1, 1],
        'width': points.shape[0],  # 使用点云数据的行数作为宽度
        'height': 1,
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],
        'points': points.shape[0],  # 使用点云数据的行数作为点数
        'data': 'ascii'
    }
pcd_data = pypcd.PointCloud(meta_data, points)
pcd_data.save_pcd(pcd_path, compression='ascii')

