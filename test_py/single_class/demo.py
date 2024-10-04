# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  : https://tianchi.aliyun.com/notebook/231732
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
import os
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as ptl
from torchvision import transforms
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.utils.data as Data
from test_py.single_class import GRU, TCN


def abs_sum(y_pre, y_tru):
    y_pre = np.array(y_pre)
    y_tru = np.array(y_tru)
    loss = sum(abs(y_pre - y_tru))
    return loss


device = torch.device('cuda:0')
batch_size = 256

train_data = pd.read_csv('/home/PKing/nasdata/tmp/tmp/challenge/心跳信号分类预测/train.csv')
test_data = pd.read_csv('/home/PKing/nasdata/tmp/tmp/challenge/心跳信号分类预测/testA.csv')

train_feature = train_data['heartbeat_signals'].str.split(',', expand=True).astype(float)
test_feature = test_data['heartbeat_signals'].str.split(',', expand=True).astype(float)

train_label = train_data['label'].values
train_feature = train_feature.values
test_feature = test_feature.values

ros = RandomOverSampler(random_state=2021)
X_resampled, y_resampled = ros.fit_resample(train_feature, train_label)

X_test = torch.FloatTensor(test_feature)
test_torch_dataset = Data.TensorDataset(X_test.reshape([X_test.shape[0], X_test.shape[1], 1]))
test_loader = Data.DataLoader(
    dataset=test_torch_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.3, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class MyDataset(Dataset):
    def __init__(self, X, Y, is_train=False):
        # 定义好 image 的路径
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
        self.is_train = is_train

    def __getitem__(self, index):
        x = self.X[index]
        if self.is_train:
            # add random noise
            x += torch.rand_like(x) * 0.03

            # shift
            offset = int(np.random.uniform(-10, 10))
            if offset >= 0:
                x = torch.cat((x[offset:], torch.rand(offset) * 0.001))
            else:
                x = torch.cat((torch.rand(-offset) * 0.001, x[:offset]))

        x = x.reshape(*x.shape, 1)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


epochs = 40
time_step = 205
input_size = 1
lr = 0.001

plot_loss = []
output_test = np.zeros((test_feature.shape[0], 4))

kf = KFold(n_splits=5, shuffle=True, random_state=2021)
for k, (train, val) in enumerate(kf.split(X_resampled, y_resampled)):
    best_score = float('inf')
    model = GRU.ModelGRU()
    model = model.to(device)
    best_model = GRU.ModelGRU()
    best_model = best_model.to(device)
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    X_train = X_resampled[train]
    X_val = X_resampled[val]
    y_train = y_resampled[train]
    y_val = y_resampled[val]

    train_torch_dataset = MyDataset(X_train, y_train, is_train=True)
    val_torch_dataset = MyDataset(X_val, y_val, is_train=False)

    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = Data.DataLoader(
        dataset=val_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_acc_meter = AverageMeter()
    test_acc_meter = AverageMeter()

    for epoch in range(epochs):
        model.train()
        train_acc_meter.reset()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            loss = loss_function(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\r batch: {}/{}'.format(i, len(train_loader)), end='')

        score = 0
        model.eval()
        test_acc_meter.reset()
        for i, (batch_x, batch_y) in enumerate(val_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            pred = out.argmax(dim=1)
            score += abs_sum(pred.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
            acc = (out.max(1)[1] == batch_y).float().mean()
            test_acc_meter.update(acc.item(), 1)
        if score < best_score:
            best_score = score
            best_model.load_state_dict(model.state_dict())
        print('test epoch: {}, loss: {}, score: {}, acc:{}'.format(epoch, loss.item(), score, test_acc_meter.avg))
    result = None
    best_model.eval()
    for x in test_loader:
        x = x[0]
        x = x.to(device)
        out = best_model(x)
        out = torch.softmax(out, dim=1)
        pred = out.detach().cpu().numpy()
        if result is None:
            result = pred
        else:
            result = np.concatenate((result, pred), axis=0)
    output_test += result

# 保存
# torch.save(best_model.state_dict(), 'GRU-KFold-lm-rand-shift-03-{}.pkl'.format(k))
# plt.plot(plot_loss)
