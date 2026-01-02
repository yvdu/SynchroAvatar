# 读取 .obj 文件
obj_filename = "3dmm_model.obj"
with open(obj_filename, "r") as f:
    obj_lines = f.readlines()

# 插入 mtllib 行
if not any("mtllib" in line for line in obj_lines):
    obj_lines.insert(0, "mtllib 3dmm_model.mtl\n")

# 设置所有顶点的材质
obj_lines.append("\nusemtl material_0\n")

# 写回 .obj 文件
with open(obj_filename, "w") as f:
    f.writelines(obj_lines)
