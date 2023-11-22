from datetime import datetime
import torch
import torch.optim as optim

from dataset import ElevatorDataset,augDataset,SCLDataset,TwoCropTransform
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from losses import FocalLoss,SupConLoss
from models.resnet1 import ResNet
from models.resnet2 import resnet18,resnet34

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from util import split_dataset, validate
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ElevatorDataset('processed_data/data_imbal.csv')

batch_size=4096

train_dataset,test_dataset,train_dataset=split_dataset(dataset,0.7,0.15,0.15)

train_dataset=augDataset(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# %%
model = torch.load('model/Resnet_2023-07-04-13-59-36-tag1.pth')
model.eval()
predictions = []
true_labels = []



# 选择每个类别的一定数量的样本
num_samples_per_class = 50
selected_inputs = []
selected_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        # 将样本和标签转换为NumPy数组
        inputs_np = inputs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for label in [0,1,4]:
            label_indices = np.where(labels_np == label)[0]
            selected_indices = label_indices[:num_samples_per_class]
            selected_inputs.append(inputs_np[selected_indices])
            selected_labels.append(labels_np[selected_indices])

# 将选定的样本和标签合并为单个NumPy数组
selected_inputs = np.concatenate(selected_inputs, axis=0)
selected_labels = np.concatenate(selected_labels, axis=0)

# 将选定的样本输入模型并获取预测结果
selected_inputs = torch.from_numpy(selected_inputs).to(device)

x = selected_inputs.unsqueeze(1)
# from IPython import embed; embed()
x = model.conv1(x)
x = model.bn1(x)
x = model.relu(x)
x = model.maxpool(x)

x = model.layer1(x)
x = model.layer2(x)
x = model.layer3(x)
x = model.layer4(x)

x = model.avgpool(x)
x = torch.flatten(x, 1)

# with torch.no_grad():
#     outputs = model(x)
#     _, selected_predictions = torch.max(outputs, 1)

# 使用TSNE对选定的样本进行降维
tsne = TSNE(n_components=2, random_state=42)
embedded_samples = tsne.fit_transform(x.data.cpu().numpy())

d={0:'normal',1:'F17',4:'F05'}

# 绘制TSNE可视化图
plt.figure(figsize=(10, 8))
for label in np.unique(selected_labels):
    label_indices = np.where(selected_labels == label)[0]
    plt.scatter(
        embedded_samples[label_indices, 0],
        embedded_samples[label_indices, 1],
        label=f'Class {d[label]}'
    )
plt.legend()
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.title('TSNE Visualization of Selected Samples')
plt.show()
plt.savefig('imgs/tsne3.png')

# with torch.no_grad():
#     for inputs, labels in test_dataloader:
#         inputs = inputs.to(torch.float32).to(device)
#         outputs = model(inputs).to(device)
#         _, predicted_labels = torch.max(outputs, 1)

#         predictions.extend(predicted_labels.tolist())
#         true_labels.extend(labels.tolist())

# # 将预测结果和真实标签转换为NumPy数组
# predictions = np.array(predictions)
# true_labels = np.array(true_labels)

# # 选择每个类别的一定数量的样本
# num_samples_per_class = 50
# selected_indices = []
# for label in [0,1,4]:
#     label_indices = np.where(true_labels == label)[0]
#     selected_indices.extend(label_indices[:num_samples_per_class])

# selected_predictions = predictions[selected_indices]
# selected_true_labels = true_labels[selected_indices]


