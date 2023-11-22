import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from models.resnet1 import ResNet
from models.resnet2 import resnet18,resnet34
from models.MLP import MLP
from losses import FocalLoss,SupConLoss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class FD_SCL(nn.Module):
    def __init__(self,train_loader,val_loader,test_loader):
        super(FD_SCL, self).__init__()
        self.train_loader = train_loader
        self.val_loader=val_loader
        self.test_loader=test_loader
        self.num_classes=6
        # self.net=resnet18(out_channel=self.num_classes)
        self.net=torch.load('model/Resnet_2023-07-04-13-59-36-tag1.pth')
        self.net.fc=nn.Identity()
        # self.net= MLP().encoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


    def train_extractor(self):
        self.net.to(self.device)
        criterion = SupConLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

        # 训练模型
        num_epochs = 1000

        train_bar=tqdm(range(num_epochs))

        for epoch in train_bar:
            self.net.train()
            for inputs, labels in self.train_loader:

                inputs = torch.cat([inputs[0], inputs[1]], dim=0)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                bsz = labels.shape[0]
                # features = self.net(inputs.unsqueeze(1))
                # features = F.normalize(features.squeeze(), dim=1)

                features = self.net(inputs)
                features = F.normalize(features, dim=1)

                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                loss = criterion(features, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # accuracies = validate(net, val_loader, num_classes)

            train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}'.format(epoch, num_epochs, optimizer.param_groups[0]['lr'], loss.item()))

        # filename = f"model/Resnet_{self.timestamp}.pth"
        # torch.save(self.net, filename)


      # train procedure common for KNN and SVC classifier (save a lot of training time)
    def train_classifier(self, method, train_loader, K_nn = None):
        torch.no_grad()
        torch.cuda.empty_cache()
        print(f'Training {method} classifier')

        feature_extractor = self.net.to(self.device)
        feature_extractor.train(False)

        # -- train a SVC classifier
        X_train, y_train = [], []

        for input, label in train_loader:
            input = input.to(self.device)
            feature = feature_extractor(input)
            feature = F.normalize(feature, dim=1)

            # feature = feature_extractor(input.unsqueeze(1))
            # feature = F.normalize(feature.squeeze(), dim=1)

            X_train.extend(feature.cpu().detach().numpy())
            y_train.extend(label)

        if method == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors = K_nn)
        elif method == 'SVC':
            classifier = LinearSVC()

        self.classifier = classifier.fit(X_train, y_train)

    # common classify function
    def KNN_SVC_classify(self, inputs):
        torch.no_grad()
        torch.cuda.empty_cache()

        inputs = inputs.to(self.device)
        feature_extractor = self.net.to(self.device)
        feature_extractor.train(False)
        features = feature_extractor(inputs)
        features = features.cpu().detach().numpy()
        # features = feature_extractor(inputs.unsqueeze(1))
        # features = features.squeeze().cpu().detach().numpy()
        preds = self.classifier.predict(features)
            # --- end prediction
        return torch.tensor(preds)


    def eval(self):
        self.net.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                preds = self.KNN_SVC_classify(inputs)

                # _, predicted_labels = torch.max(preds, 1)

                predictions.extend(preds.tolist())
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
        plt.savefig(f'imgs/cm_{self.timestamp}.png')
        plt.show()


    def save():
        pass