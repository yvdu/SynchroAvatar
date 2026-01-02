import os
import argparse


def find_clip_imports(root_dir, search_text="import clip"):
    """
    递归查找目录中所有包含指定导入语句的Python文件
    :param root_dir: 要搜索的根目录
    :param search_text: 要查找的文本（默认查找import clip）
    :return: 包含匹配文件路径的列表
    """
    matched_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查是否是Python文件
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        # 检查是否包含目标文本
                        if search_text in f.read():
                            matched_files.append(filepath)
                except Exception as e:
                    print(f"无法读取文件 {filepath}: {str(e)}")

    return matched_files


if __name__ == "__main__":


    results = find_clip_imports('.', "caotiezheng")

    if results:
        print("找到以下匹配文件：")
        for idx, filepath in enumerate(results, 1):
            print(f"{idx}. {filepath}")
    else:
        print("未找到匹配文件")