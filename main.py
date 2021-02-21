from dataset import TextClassificationDataSet
from model import Rnn
from train import train_model
from corpus import *
import torch
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Text Classification.')
parser.add_argument('--model', help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()
# print(args.accumulate(args.integers))


# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import numpy as np
# np.random.seed(2)    #指定生成“特定”的随机数-与seed 1 相关
# a = np.random.random()
# print(a)
#
# 加载数据集
# dataset = TextClassificationDataSet("data/all_data.tsv", max_seq_length=50, dict_path="dict.txt")
#
#
# model_ft = Rnn()
#
# # Define Loss Function
# loss_fn = nn.CrossEntropyLoss()
#
# # Define optimizer
# learning_rate = 0.001
# optimizer_fn = torch.optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
#
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_fn, step_size=7, gamma=0.1)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model_ft = train_model(model_ft, loss_fn, optimizer_fn, exp_lr_scheduler,
#                        num_epochs=3)

corpus_dict = CorpusDict(corpus_path="dict.txt", stopwords_path="dict.txt")
