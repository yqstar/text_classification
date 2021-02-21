from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch


# 构建Dataset
class TextClassificationDataSet(Dataset):
    """
    文本分类数据集类，继承torch.utils.data.Dataset，需要重写以下方法
    __len__方法：len(dataset)返回数据集的大小
    __getitem__方法：支持索引，以便dataset[i]可用于获取第i个样本
    """

    def __init__(self, csv_path, max_seq_length, dict_path):
        """
        初始化Dataset
        :param csv_path: 数据集csv文件存放路径
        :param max_seq_length: 分类文本最大长度，针对大于该长度进行阶段操作，小于该长度进行pad操作。
        :param dict_path: 词典存放路径
        """
        self.data = pd.read_csv(csv_path, sep="\t")
        self.max_seq_length = max_seq_length
        self.dict_path = dict_path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        text = self.data["text"][idx]
        dict_corpus = self.get_dict()
        self.text = [dict_corpus[i] for i in text]
        if len(self.text) < self.max_seq_length:
            self.text += [dict_corpus['pad']] * (self.max_seq_length - len(self.text))
        else:
            self.text = self.text[0:self.max_seq_length]
        # dict_corpus = self.get_dict()
        # self.text = self.text2idx(dict_corpus, text)
        self.label = self.data["label"][idx]
        self.sample = {"text": torch.tensor(self.text), "label": torch.tensor(self.label)}
        return self.sample

    # @staticmethod
    def get_dict(self):
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            # eval函数将字符串形式的字典转成字典
            dict_corpus = eval(f.read())
        return dict_corpus


class BuildDataLoader(object):
    def __init__(self, dataset, ):
        self.dataset = dataset
        self.test_
        pass

    def split_dataloader(self):
        val_percent = 0.3
        n_val = int(len(self.dataset) * val_percent)
        n_train = len(self.dataset) - n_val
        train_set, val_set = random_split(self.dataset, [n_train, n_val])
        dataloader_train = DataLoader(train_set, batch_size=100,
                                      shuffle=True, num_workers=0)
        dataloader_val = DataLoader(val_set, batch_size=100,
                                    shuffle=True, num_workers=0)
        dataloaders = {'train': dataloader_train, 'val': dataloader_val}
        dataset_sizes = {'train': n_train, 'val': n_val}
        return dataloaders, dataset_sizes
