import numpy as np
import trimesh

# 加载 3DMM 数据
vertices = np.load("3dmm_mean_shape.npy").reshape(-1, 3)  # 顶点坐标
faces = np.load("3dmm_faces.npy")  # 三角面索引
uv_map = np.load("3dmm_uv.npy")  # UV 贴图坐标
normals = np.load("normal_map.npy")  # 法线
texture = "base_color_texture.png"  # 纹理贴图

# 创建 trimesh 对象
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

# 生成 .obj 文件
obj_filename = "3dmm_model.obj"
mesh.export(obj_filename)

print(f"✅ 已导出 {obj_filename}")
