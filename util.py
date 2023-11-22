import torch
import numpy as np
from torch.utils.data import random_split, Subset
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from sklearn.model_selection import StratifiedShuffleSplit

def split_dataset(dataset, test_samples_per_class):
    labels = np.array(dataset.label)
    unique_classes = np.unique(labels)

    test_indices = []
    train_val_indices = []

    for cls in unique_classes:
        # 找到当前类别的所有样本的索引
        indices = np.where(labels == cls)[0]
        np.random.shuffle(indices)

        # 从每个类别中抽取固定数量的样本作为测试集
        test_indices.extend(indices[:test_samples_per_class])

        # 剩余的样本用于训练和验证
        train_val_indices.extend(indices[test_samples_per_class:])

    # 随机打乱训练和验证的样本
    np.random.shuffle(train_val_indices)

    # 分割训练和验证集
    val_size = int(0.15 * len(train_val_indices))  # 假设验证集占剩余样本的 15%
    val_indices = train_val_indices[:val_size]
    train_indices = train_val_indices[val_size:]

    # 创建 PyTorch 子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 计算每个子集的类别数量
    train_class_counts = count_classes_in_subset(dataset, train_indices)
    val_class_counts = count_classes_in_subset(dataset, val_indices)
    test_class_counts = count_classes_in_subset(dataset, test_indices)

    print("Train set class distribution:", train_class_counts)
    print("Validation set class distribution:", val_class_counts)
    print("Test set class distribution:", test_class_counts)

    return train_dataset, val_dataset, test_dataset

def count_classes_in_subset(dataset, indices):
    labels = [dataset.label[i] for i in indices]
    return Counter(labels)
# def split_dataset(dataset, train_ratio, test_ratio, val_ratio):
#     # 假设 dataset.targets 包含所有样本的标签
#     labels = dataset.label

#     # 计算训练集和剩余部分（测试集和验证集）的比例
#     remaining_ratio = test_ratio + val_ratio
#     stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=remaining_ratio)

#     for train_index, remaining_index in stratified_split.split(labels, labels):
#         train_dataset = Subset(dataset, train_index)
#         remaining_dataset = Subset(dataset, remaining_index)

#     # 再次使用分层抽样来分割测试集和验证集
#     test_size_relative = test_ratio / remaining_ratio
#     stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size_relative)

#     labels_remaining = [labels[i] for i in remaining_index]

#     for val_index, test_index in stratified_split.split(remaining_dataset, labels_remaining):
#         val_dataset = Subset(remaining_dataset, val_index)
#         test_dataset = Subset(remaining_dataset, test_index)

#     # 计算每个子集的类别数量
#     train_labels = [labels[i] for i in train_index]
#     val_labels = [labels_remaining[i] for i in val_index]
#     test_labels = [labels_remaining[i] for i in test_index]

#     print("Train set class distribution:", Counter(train_labels))
#     print("Validation set class distribution:", Counter(val_labels))
#     print("Test set class distribution:", Counter(test_labels))

#     return train_dataset, test_dataset, val_dataset


# def split_dataset(dataset,train_ratio,test_ratio,val_ratio):
#     dataset_size = len(dataset)
#     train_size = int(train_ratio * dataset_size)
#     val_size = int(val_ratio * dataset_size)
#     test_size = dataset_size - train_size - val_size

#     train_dataset, remaining_dataset = random_split(dataset, [train_size, dataset_size - train_size])
#     val_dataset, test_dataset = random_split(remaining_dataset, [val_size, test_size])

#     return train_dataset,test_dataset,val_dataset



def validate(model, dataloader, num_classes):
    model.eval()
    correct = [0] * num_classes
    total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(num_classes):
                class_indices = labels == i
                class_total = torch.sum(class_indices)
                total[i] += class_total.item()
                correct[i] += torch.sum(predicted[class_indices] == labels[class_indices]).item()

    accuracies = [correct[i] / total[i] if total[i] != 0 else 0 for i in range(num_classes)]
    return accuracies