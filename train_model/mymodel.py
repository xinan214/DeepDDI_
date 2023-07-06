import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.dense_198 = nn.Linear(200, 1024)
        self.activation_198 = nn.ReLU()
        self.dropout_164 = nn.Dropout(0.3)
        self.batch_normalization_164 = nn.BatchNorm1d(1024)
        self.dense_199 = nn.Linear(1024, 1024)
        self.activation_199 = nn.ReLU()
        self.dropout_165 = nn.Dropout(0.3)
        self.batch_normalization_165 = nn.BatchNorm1d(1024)
        self.dense_200 = nn.Linear(1024, 1024)
        self.activation_200 = nn.ReLU()
        self.dropout_166 = nn.Dropout(0.3)
        self.batch_normalization_166 = nn.BatchNorm1d(1024)
        self.dense_201 = nn.Linear(1024, 1024)
        self.activation_201 = nn.ReLU()
        self.dropout_167 = nn.Dropout(0.3)
        self.batch_normalization_167 = nn.BatchNorm1d(1024)
        self.dense_202 = nn.Linear(1024, 113)
        self.activation_202 = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_198(x)
        x = self.activation_198(x)
        x = self.dropout_164(x)
        x = self.batch_normalization_164(x)
        x = self.dense_199(x)
        x = self.activation_199(x)
        x = self.dropout_165(x)
        x = self.batch_normalization_165(x)
        x = self.dense_200(x)
        x = self.activation_200(x)
        x = self.dropout_166(x)
        x = self.batch_normalization_166(x)
        x = self.dense_201(x)
        x = self.activation_201(x)
        x = self.dropout_167(x)
        x = self.batch_normalization_167(x)
        x = self.dense_202(x)
        x = self.activation_202(x)

        return x


# 创建模型实例
# model = MyModel()
#
# # 打印模型结构
# print(model)
