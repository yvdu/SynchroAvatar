# 创建 MTL 文件
mtl_filename = "3dmm_model.mtl"
with open(mtl_filename, "w") as f:
    f.write("newmtl material_0\n")  # 定义材质
    f.write("Ka 1.0 1.0 1.0\n")  # 环境光
    f.write("Kd 1.0 1.0 1.0\n")  # 漫反射
    f.write("Ks 0.0 0.0 0.0\n")  # 镜面反射
    f.write("Ns 10.0\n")  # 镜面高光
    f.write("map_Kd base_color_texture.png\n")  # 关联 Diffuse 贴图
