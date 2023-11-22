from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import ElevatorDataset,augDataset,SCLDataset,TwoCropTransform

from util import split_dataset, validate
from FD_scl_model import FD_SCL

import argparse

import math
import random

def __item_reorder(item_seq):
    item_seq_len =  item_seq.shape[0]
    beta = 0.6
    num_reorder = math.floor(item_seq_len * beta)
    reorder_begin = random.randint(0, item_seq_len - num_reorder)
    reordered_item_seq = item_seq.clone()
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
    return reordered_item_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--runs', type=int, default=5)

    # parser.add_argument('--use_bert', type=str2bool, default=True)
    # parser.add_argument('--use_cmd_sim', type=str2bool, default=True)


    dataset = ElevatorDataset('processed_data/data_imbal.csv')

    batch_size=4096

    train_dataset,test_dataset,val_dataset=split_dataset(dataset,0.7,0.15,0.15)

    train_dataset = augDataset(train_dataset)

    train_transform = T.Lambda(lambda img:__item_reorder(img))
    train_SCL =  SCLDataset(train_dataset.data, train_dataset.label, transform=TwoCropTransform(train_transform))

    train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_scl = DataLoader(train_SCL, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    model=FD_SCL(train_loader_scl,val_loader,test_loader)
    model.train_extractor()
    model.train_classifier('SVC',train_loader)
    model.eval()




