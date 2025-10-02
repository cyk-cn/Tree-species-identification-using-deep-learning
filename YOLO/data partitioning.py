import os
import shutil
import random
from collections import defaultdict

def split_dataset(dataset_dir, output_dir, train_ratio=0.8):
    # 创建训练集和验证集文件夹
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 遍历每个类别文件夹
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 获取所有文件并按照树的标识符分类
        tree_files = defaultdict(list)
        for file in os.listdir(class_dir):
            if '_' in file:
                tree_id = file.split('_')[0] + '_' + file.split('_')[1]
                tree_files[tree_id].append(file)

        # 打乱树的顺序
        tree_ids = list(tree_files.keys())
        random.shuffle(tree_ids)

        # 计算训练集和验证集的分割点
        split_point = int(len(tree_ids) * train_ratio)
        train_tree_ids = tree_ids[:split_point]
        val_tree_ids = tree_ids[split_point:]

        # 创建类别文件夹
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        # 移动文件到训练集和验证集
        for tree_id in train_tree_ids:
            for file in tree_files[tree_id]:
                shutil.move(os.path.join(class_dir, file), os.path.join(train_class_dir, file))

        for tree_id in val_tree_ids:
            for file in tree_files[tree_id]:
                shutil.move(os.path.join(class_dir, file), os.path.join(val_class_dir, file))



dataset_dir =
output_dir =
split_dataset(dataset_dir, output_dir)
