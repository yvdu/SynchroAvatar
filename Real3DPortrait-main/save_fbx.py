import fbx
import numpy as np

# 加载 3DMM 数据
vertices = np.load("3dmm_mean_shape.npy").reshape(-1, 3)
faces = np.load("3dmm_faces.npy")
blendshape_basis = np.load("blendshape_basis.npy")  # 形变基底

# 初始化 FBX 管理器
manager = fbx.FbxManager.Create()
scene = fbx.FbxScene.Create(manager, "MyScene")

# 创建 FBX 3D 对象
mesh = fbx.FbxMesh.Create(scene, "3DMM_Mesh")
node = fbx.FbxNode.Create(scene, "3DMM_Node")
scene.GetRootNode().AddChild(node)
node.SetNodeAttribute(mesh)

# **初始化顶点**
mesh.InitControlPoints(len(vertices))

# **设置顶点**
for i, v in enumerate(vertices):
    mesh.SetControlPointAt(fbx.FbxVector4(v[0], v[1], v[2]), i)

# **添加三角面**
for f in faces:
    mesh.BeginPolygon()
    for i in range(3):
        mesh.AddPolygon(int(f[i]))
    mesh.EndPolygon()

# **修正错误：将 BlendShape 绑定到 mesh**
blendshape = fbx.FbxBlendShape.Create(scene, "BlendShape")
mesh.AddDeformer(blendshape)  # ✅ 绑定到 mesh，而不是 node

for i in range(blendshape_basis.shape[1]):  # 遍历所有 BlendShape
    blendshape_channel = fbx.FbxBlendShapeChannel.Create(scene, f"BlendShape_{i}")
    blendshape.AddBlendShapeChannel(blendshape_channel)
    blendshape_shape = fbx.FbxShape.Create(scene, f"BlendShape_{i}_Shape")

    for j in range(len(vertices)):  # 遍历所有顶点
        delta = blendshape_basis[j, i] * np.array([1, 1, 1]) * 0.01  # 确保是 3D 向量
        v = vertices[j]  # 获取原始顶点坐标

        blendshape_shape.SetControlPointAt(fbx.FbxVector4(v[0] + delta[0], v[1] + delta[1], v[2] + delta[2]), j)

    blendshape_channel.AddTargetShape(blendshape_shape)


# **导出 FBX**
exporter = fbx.FbxExporter.Create(manager, "")
exporter.Initialize("3dmm_model.fbx", -1, manager.GetIOSettings())
exporter.Export(scene)
exporter.Destroy()

print("✅ 已导出 3dmm_model.fbx")
