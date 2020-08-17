import torch.utils.data as torchdata
from modules.data.utils import collate_fn
from modules.data.dataset import ICDAR


class ICDARDataLoader:

    def __init__(self, config):
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.shuffle = config['data_loader']['shuffle']
        self.num_workers = config['data_loader']['workers']
        self.val_split_ratio = self.config['validation']['validation_split']
        data_root = config['data_loader']['data_dir']
        input_size = config['data_loader']['input_size']
        icdar_dataset = ICDAR(data_root, input_size)
        if self.val_split_ratio > 0:
            self.__train_set, self.__val_set = self.__train_val_split(icdar_dataset)
        else:
            self.__train_set = icdar_dataset

    def train(self):
        trainLoader = torchdata.DataLoader(self.__train_set, num_workers=self.num_workers, batch_size=self.batch_size,
                                           shuffle=self.shuffle, collate_fn=collate_fn)
        return trainLoader

    def val(self):
        assert self.val_split_ratio > 0, 'Error: call val() method'
        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__val_set, num_workers=self.num_workers, batch_size=self.batch_size,
                                         shuffle=shuffle, collate_fn=collate_fn)
        return valLoader

    def __train_val_split(self, ds):
        """

        :param ds: dataset
        :return:
        """
        split = self.val_split_ratio
        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val
