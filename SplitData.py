#神经网络实验2
import os
import shutil
import random

# 设置源文件夹和目标文件夹
source_folder = 'train'
train_folder = 'trainset'
test_folder = 'testset'


# 确保目标文件夹存在，如果存在则清空其内容（防止重复运行代码导致数据集混淆）
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Unable to delete {file_path}. Reason: {e}')

                # 创建目录（如果它之前被删除了）
    if not os.path.exists(directory):
        os.makedirs(directory)
clear_directory(train_folder)
clear_directory(test_folder)
    # 获取源文件夹中所有的png文件
png_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]

# 随机打乱文件列表
random.shuffle(png_files)

# 计算训练和测试集的数量
split_index = int(len(png_files) * 0.8)
train_files = png_files[:split_index]
test_files = png_files[split_index:]

# 将文件复制到相应的目标文件夹
for file in train_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))

for file in test_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

print(f"训练集包含 {len(train_files)} 个文件")
print(f"测试集包含 {len(test_files)} 个文件")