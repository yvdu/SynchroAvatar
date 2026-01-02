import os
import re


def replace_imports_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    for line in lines:
        # 如果已经是 "from utils2.xxx"，则保持不变
        if re.match(r'^\s*from\s+utils\b', line):
            new_lines.append(line)
        # 替换 "from utils.xxx" -> "from utils2.xxx"
        elif re.match(r'^\s*from\s+utils2\b', line):
            new_line = re.sub(r'from\s+utils2\b', 'from utils', line)
            new_lines.append(new_line)
            modified = True
        else:
            new_lines.append(line)

    # 如果有修改，则覆盖写入
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"✅ 已修改: {file_path}")


def scan_and_replace():
    current_dir = os.getcwd()

    for root, _, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replace_imports_in_file(file_path)


if __name__ == "__main__":
    scan_and_replace()
