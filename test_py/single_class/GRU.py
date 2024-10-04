# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail :
    @Date   : 2024-02-05 18:19:18
    @Brief  : https://tianchi.aliyun.com/notebook/231732
"""
import torch
from torch import nn


class ModelGRU(nn.Module):
    def __init__(self, input_size=1, output_size=4, hidden_size=300):
        """
        :param input_size: The number of expected features in the input `x`
        :param output_size: num_class
        :param hidden_size: The number of features in the hidden state `h`
        """
        super(ModelGRU, self).__init__()
        self.hidden_layer_size = hidden_size
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=0.8,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        """
        :param inputs: shape is (batch,dim-size,1)
        :return:
        """
        r_out, h_c = self.lstm(inputs, None)
        x = self.linear(r_out[:, -1, :])
        return x
