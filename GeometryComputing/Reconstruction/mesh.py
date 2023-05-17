import numpy as np
import open3d as o3d

def read_xyz_file(filename):
    points = []
    normals = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z, nx, ny, nz = map(float, line.split())
            points.append((x, y, z))
            normals.append((nx, ny, nz))
    return np.array(points), np.array(normals)

DATA_ROOT = "gargoyle.xyz"
id = 7000
points, normals = read_xyz_file(DATA_ROOT)
# 假设你有一个点云数据和法线数据
point_cloud = points  # N x 3的点云坐标矩阵
normals = np.load(f"/Users/genghaoran/Code/HomeworkProject/GeometryComputing/Reconstruction/output/{id}_pred.npy")  # N x 3的法线向量矩阵

# 创建Open3D的PointCloud对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.normals = o3d.utility.Vector3dVector(normals)

# 使用Open3D中的函数进行网格重建
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

# 可选：进行网格的平滑和细化处理
# mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
# mesh = mesh.filter_sharpen()

# # 可选：显示和保存网格模型
# o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh(f"output/mesh_{id}.obj", mesh)
