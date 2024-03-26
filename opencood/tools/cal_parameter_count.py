# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/26 13:31
@Auth ： YongTong Gu
@File ：cal_parameter_count.py
@IDE ：PyCharm
@Motto：悟已往之不谏,知来者之可追

"""
import torch
import torch.nn as nn


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


if __name__ == '__main__':
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.fc1 = nn.Linear(20 * 20 * 20, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(-1, 20 * 20 * 20)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    m = SimpleModel()
    model_structure(m)
