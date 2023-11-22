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
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ElevatorDataset('processed_data/data_imbal_23_11_15.csv')

batch_size=1024

train_dataset,val_dataset,test_dataset=split_dataset(dataset,40)

train_dataset=ConcatDataset([train_dataset,val_dataset])
train_dataset=augDataset(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('Train dataset size:',train_dataset.__len__())
print('Test dataset size:',test_dataset.__len__())
print('Val dataset size:',val_dataset.__len__())


num_classes = 6

# model = ResNet(num_classes=6)
model = resnet18(out_channel=num_classes)

model.to(device)

# 定义损失函数和优化器
criterion = FocalLoss()
# criterion = SupConLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# 训练模型
num_epochs = 300

train_bar=tqdm(range(num_epochs))

for epoch in train_bar:
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    accuracies = validate(model, val_loader, num_classes)

    train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Val acc:{}'.format(epoch, num_epochs, optimizer.param_groups[0]['lr'], loss.item(), str(accuracies)))


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

filename = f"model/Resnet_{timestamp}.pth"

torch.save(model, filename)

# %%
# model=torch.load('model/Resnet_2023-07-04-13-59-36-tag1.pth')
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)

        predictions.extend(predicted_labels.tolist())
        true_labels.extend(labels.tolist())

# 计算评估指标
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



cm = confusion_matrix(true_labels, predictions)

# 计算每个类别的准确率
num_classes = len(cm)
class_accuracy = {}

for i in range(num_classes):
    class_accuracy[i] = cm[i, i] / cm[i, :].sum()

# 打印每个类别的准确率
for i in range(num_classes):
    print(f"Class {i} accuracy: {class_accuracy[i]:.4f}")

# 计算总体准确率
overall_accuracy = accuracy_score(true_labels, predictions)
print("Overall accuracy:", overall_accuracy)


# 计算混淆矩阵
cm = confusion_matrix(true_labels, predictions)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.savefig(f'imgs/cm_{timestamp}.png')
plt.show()


