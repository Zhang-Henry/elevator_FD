from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import torch
from imblearn.over_sampling import SMOTE

class ElevatorDataset(Dataset):
    def __init__(self, path):
        self.df = df = pd.read_csv(path)
        # df=df_[['data1_dec','data2_dec','data3_dec','Flag']]
        df['class'] = df['Flag'].apply(self.classify_flag)

        df['data1']=df['data1_dec'].str.split().apply(lambda x: [int(val) for val in x])
        df['data2']=df['data2_dec'].str.split().apply(lambda x: [int(val) for val in x])
        df['data3']=df['data3_dec'].str.split().apply(lambda x: [int(val) for val in x])


        scaler = StandardScaler()
        # scaler = MinMaxScaler()
        for column in ['data1', 'data2', 'data3']:
            df[column] = df[column].apply(lambda x: scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten())

        df['data'] = df[['data1','data2','data3']].apply(lambda row: list(np.concatenate(row.values)), axis=1)

        df = df[df['data'].apply(lambda x: len(x) == 900)]
        df.reset_index(drop=True,inplace=True)

        print(df['class'].value_counts())

        self.data=df.data.to_numpy()
        self.label=df['class']



    def __len__(self):
        return len(self.data)

    # def norm(self,d):
    #     return (d-d.min())/(d.max()-d.min())

    def __getitem__(self, index):
        # d1=torch.tensor(self.df['data1'][index], dtype=torch.float32)
        # d1=self.norm(d1)
        # d2=torch.tensor(self.df['data2'][index], dtype=torch.float32)
        # d2=self.norm(d2)
        # d3=torch.tensor(self.df['data3'][index], dtype=torch.float32)
        # d3=self.norm(d3)
        # data = torch.cat((d1,d2,d3),0)

        data = self.data[index]
        label = self.label[index]

        # # 数据预处理和转换
        data = torch.tensor(data, dtype=torch.float32)

        # 返回特征数据和标签
        return data, label


    def classify_flag(self, flag):
        if flag == 'F14,':
            return 1
        elif flag == 'F17,':
            return 2
        elif flag == 'F09,':
            return 3
        elif flag == 'F05,':
            return 4
        elif flag == 'F10,':
            return 5
        else:
            return 0


class augDataset(Dataset):
    def __init__(self,dataset):
        data,label=[],[]
        for d, l in dataset:
            data.append(list(d))
            label.append(l)
        self.data=np.array(data)
        self.label=np.array(label)
        aug_num = {0:9500,1:4000,2:3000,3:2000,4:1000,5:1000}
        print('Augment number:',aug_num)
        sm = SMOTE(sampling_strategy=aug_num, random_state=42)
        self.data, self.label = sm.fit_resample(self.data, self.label)
        print('Augmented by SMOTE.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        # # 数据预处理和转换
        data = torch.tensor(data, dtype=torch.float32)

        # 返回特征数据和标签
        return data, label


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [x, self.transform(x)]


class SCLDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.X = x
        # self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = y
        # self.y = torch.tensor(self.y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            x = self.transform(x)
        return x,label