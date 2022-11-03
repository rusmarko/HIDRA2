import random
from datetime import datetime

import numpy as np
import torch

from hidra2.hidra2 import HIDRA
from utils.dataset import Data
from utils.training import Training


class Config():
    def __init__(self):
        self.seed = 0

        self.data_path = '../data'

        self.lr = .0001
        self.batch_size = 512
        self.epochs = 40

        self.num_workers = 4
        self.device = torch.device('cuda:0')

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


if __name__ == '__main__':
    print('start:', datetime.now())

    config = Config()
    training = Training(
        config,
        net=HIDRA(),
        data=Data(config)
    )
    training.train()

    print('\nfinish:', datetime.now())
