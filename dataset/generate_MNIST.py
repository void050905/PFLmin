# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from model import MyLeNet5
from PIL import Image
from collections import Counter

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "MNIST/"
num_classes = 10

model = MyLeNet5()
model.load_state_dict(torch.load('/home/wangzishan/LeNet/save_model/best_model.pth'))
model.eval()  # 设置模型为评估模式

# 定义适用于MNIST的预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),        # MNIST原始尺寸就是28x28，这里显式声明
    transforms.ToTensor(),               # 转换为张量并自动归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的标准归一化参数
])

# 预测函数（保持不变）
def predict_label(image):
    image = transform(image).unsqueeze(0)  # 增加一个批次维度
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 计算每个客户端标签分布与总体标签分布之间的KL散度
def compute_kl_divergence(client_data, overall_distribution, epsilon=1e-10):
    """
    计算客户端标签分布与总体标签分布之间的KL散度
    使用绝对值的KL散度
    `epsilon` 用于避免零概率项对数的计算错误
    """
    # 计算客户端标签分布
    unique_labels, counts = np.unique(client_data['y'], return_counts=True)
    client_distribution = np.zeros_like(overall_distribution, dtype=np.float64)
    
    for label, count in zip(unique_labels, counts):
        if 0 <= label < num_classes:
            client_distribution[label] = count / len(client_data['y'])
    
    # 为了避免 log(0) 错误，将零概率项用 epsilon 代替
    client_distribution = np.maximum(client_distribution, epsilon)
    overall_distribution = np.maximum(overall_distribution, epsilon)
    
    # 计算KL散度（使用绝对值求和）
    kl_divergence = np.sum(np.abs(client_distribution * np.log2(client_distribution / overall_distribution)))
    
    return kl_divergence

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])

    # 加载 MNIST 数据集
    trainset = torchvision.datasets.MNIST(
        root=os.path.join(dir_path, "rawdata"), train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=os.path.join(dir_path, "rawdata"), train=False, download=True, transform=transform)
    
    # 加载完整的训练和测试数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # 合并训练和测试数据
    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().numpy())
    dataset_image.extend(testset.data.cpu().numpy())
    dataset_label.extend(trainset.targets.cpu().numpy())
    dataset_label.extend(testset.targets.cpu().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

     # 定义额外数据文件夹路径（确保路径正确且包含8个文件夹）
    extra_image_folders = [
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data1',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data2',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data3',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data4',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data5',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data6',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data7',
        '/home/wangzishan/PFL-min/PFLlib/dataset/extra_minist/extra-data8'

    ]

    extra_images = []

   # 读取并处理每个额外文件夹中的 .png 图像
    for folder in extra_image_folders:
        images_in_folder = []
        labels_in_folder = []
        if not os.path.exists(folder):
            print(f"Warning: Extra image folder '{folder}' does not exist. Skipping.")
            extra_images.append((np.array(images_in_folder), np.array(labels_in_folder)))
            continue

        for filename in sorted(os.listdir(folder)):
          if filename.endswith('.jpeg'):
              image_path = os.path.join(folder, filename)
              
              try:
                  # 打开图像并转换为灰度图
                  image = Image.open(image_path).convert('L')
              except Exception as e:
                  print(f"Error opening image {image_path}: {e}")
                  continue
              
              # 预测标签
              predicted_label = predict_label(image)
              
              # 转换为numpy数组并归一化
              image_array = np.array(image) / 255.0
              image_array = np.expand_dims(image_array, axis=0)  # 转换为 (1, 28, 28)
              
              # 保存结果
              images_in_folder.append(image_array)
              labels_in_folder.append(predicted_label)

        extra_images.append((np.array(images_in_folder), np.array(labels_in_folder)))
    
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)

    # 计算每个客户端的标签分布与总体标签分布之间的KL散度
    overall_distribution = np.ones(num_classes) * (1 / num_classes)  # 假设总体标签分布是均匀的
    client_kl_divergence = [(i, compute_kl_divergence(train_data[i], overall_distribution)) for i in range(num_clients)]

    # 按KL散度排序，选择最不均匀的客户端
    client_kl_divergence.sort(key=lambda x: x[1], reverse=True)
    num_extra = 8  # 确定最多可以分配的客户端数量
    most_imbalanced_clients = [client[0] for client in client_kl_divergence[:num_extra]]  # 选择最不均匀的客户端

    # # 给这些客户端添加额外数据
    # for idx, client_id in enumerate(most_imbalanced_clients):
    #     if idx >= len(extra_images):
    #         print(f"Warning: Not enough extra_images to assign to client {client_id}. Skipping.")
    #         continue

    #     # 获取第 client_id 个客户端的训练数据和标签
    #     client_train_images = train_data[client_id]['x']
    #     client_train_labels = train_data[client_id]['y']
        
    #     # 获取对应的额外图像和标签
    #     folder_images, folder_labels = extra_images[idx]

    #     if folder_images.size == 0 or folder_labels.size == 0:
    #         print(f"Warning: Extra data for client {client_id} is empty. Skipping.")
    #         continue
    
    # 计算总标签分布以确定每个标签的平均样本数
    total_label_counts = np.zeros(num_classes, dtype=np.int32)
    for client_id in range(num_clients):
        unique_labels, counts = np.unique(train_data[client_id]['y'], return_counts=True)
        for label, count in zip(unique_labels, counts):
            if 0 <= label < num_classes:
                total_label_counts[label] += count
    average_label_counts = total_label_counts / num_clients  # 每个标签的平均样本数
    print(average_label_counts)
  

    # 限制最多分配的客户端数量为额外数据文件夹的数量
    num_extra = 8
    selected_clients = most_imbalanced_clients[:num_extra]
    
    num_extra = 8 # 确保最多分配 num_clients 个客户端

    print(f"Selected Clients: {selected_clients}")
    print(f"Number of Selected Clients: {len(selected_clients)}")
    # 给这些客户端分配额外数据
    for idx, client_id in enumerate(selected_clients):
        # 获取第 client_id 个客户端的训练数据和标签
        client_train_images = train_data[client_id]['x']
        client_train_labels = train_data[client_id]['y']
        
        # 获取第 idx 个额外文件夹中的图像和标签
        folder_images, folder_labels = extra_images[idx]
        
        if folder_images.size == 0 or folder_labels.size == 0:
            print(f"Warning: Extra data for client {client_id} is empty. Skipping.")
            continue
        
        # 计算当前客户端每个标签的样本数量
        unique_labels, counts = np.unique(client_train_labels, return_counts=True)
        label_samples = dict(zip(unique_labels, counts))
        
        # 针对当前客户端数据中的标签缺口，选择要补充的额外数据
        for label in range(num_classes):
            current_count = label_samples.get(label, 0)
            avg_count = average_label_counts[label]
            print(current_count)
            if current_count >= avg_count:
                # 当前标签样本数已达到或超过平均值，跳过
                continue
            missing_count = int(np.ceil(avg_count - current_count))
            if missing_count <= 0:
                continue  # 已经满足或超过平均值
            
            # 获取额外数据中该标签的样本
            label_mask = folder_labels == label
            extra_label_images = folder_images[label_mask]
            extra_label_labels = folder_labels[label_mask]
            
            # 按缺口数量选择添加的额外样本
            num_available = len(extra_label_images)
            num_to_add = min(missing_count, num_available)
            if num_to_add > 0:
                client_train_images = np.concatenate([client_train_images, extra_label_images[:num_to_add]], axis=0)
                client_train_labels = np.concatenate([client_train_labels, extra_label_labels[:num_to_add]], axis=0)
                print(f"Client {client_id}: Added {num_to_add} samples for label {label}")

        
        # 更新训练集数据
        train_data[client_id] = {'x': client_train_images, 'y': client_train_labels}
        client_data = np.concatenate((train_data[client_id]['x'], test_data[client_id]['x']), axis=0)
        client_labels = np.concatenate((train_data[client_id]['y'], test_data[client_id]['y']), axis=0)
        label_counts = Counter(client_labels)
    
    # 更新 statistic，确保每个标签从0到9都有对应的计数
        statistic[client_id] = [label_counts.get(label, 0) for label in range(10)]
 # 输出该客户端的样本数量和标签分布
        print(f"Client {client_id}\t Size of data: {len(client_data)}\t Labels: ", np.unique(client_labels))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client_id]])
        print("-" * 50)

    # 保存处理后的数据集
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)



if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)