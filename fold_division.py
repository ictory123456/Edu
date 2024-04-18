import os
import numpy as np
from sklearn.model_selection import KFold
# 读取完整的数据集文件
with open('C:\\Users\\A\\Desktop\\EDU\\ATKT\\dataset\\assist2012\\assist2012.txt', 'r') as file:
    data_lines = file.read().splitlines()

# 将数据划分成每三行为一组的列表
data_groups = [data_lines[i:i+3] for i in range(0, len(data_lines), 3)]

# 创建文件夹用于保存每个折的数据
os.makedirs('cross_validation_data', exist_ok=True)

# 创建KFold对象，指定折数并不打乱数据集
kf = KFold(n_splits=5, shuffle=False)

# 执行五折交叉验证
for fold, (train_valid_index, test_index) in enumerate(kf.split(data_groups)):
    fold_dir = os.path.join('cross_validation_data', f'fold_{fold+1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 划分测试集（占一份）
    test_data = [data_groups[i] for i in test_index]
    
    # 划分验证集（占一份，大小与测试集相同）
    valid_data = [data_groups[i] for i in train_valid_index[:5804]]
    
    # 划分剩余数据为训练集（占三份）
    train_data = [data_groups[i] for i in train_valid_index[5804:]]
    
    # 将划分后的数据保存到文件中
    with open(os.path.join(fold_dir, 'train.txt'), 'w') as train_file:
        for group in train_data:
            train_file.write('\n'.join(group) + '\n')
    
    with open(os.path.join(fold_dir, 'valid.txt'), 'w') as valid_file:
        for group in valid_data:
            valid_file.write('\n'.join(group) + '\n')
    
    with open(os.path.join(fold_dir, 'test.txt'), 'w') as test_file:
        for group in test_data:
            test_file.write('\n'.join(group) + '\n')