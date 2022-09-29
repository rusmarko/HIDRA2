import torch
import yaml
from torch.utils.data import Dataset, DataLoader


class Train(Dataset):
    def __init__(self, config, norm):
        super(Train, self).__init__()

        data = torch.load(f'{config.data_path}/train.pth')
        self.atmos = data['atmos']
        self.tide = data['tide']
        self.ssh = data['ssh']
        self.valid_idx = data['valid idx']

        # normalizing
        self.atmos[:, 0] = (self.atmos[:, 0] - norm['atmos']['mean'][0]) / norm['atmos']['std'][0]
        self.atmos[:, 1] = (self.atmos[:, 1] - norm['atmos']['mean'][1]) / norm['atmos']['std'][1]
        self.atmos[:, 2] = (self.atmos[:, 2] - norm['atmos']['mean'][2]) / norm['atmos']['std'][2]
        self.tide = (self.tide - norm['tide']['mean']) / norm['tide']['std']
        self.ssh = (self.ssh - norm['ssh']['mean']) / norm['ssh']['std']

        print(f'{len(self.valid_idx)} train instances.')

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        i = self.valid_idx[i]

        ssh = self.ssh[i - 72:i]
        tide = self.tide[i - 72:i + 72]

        lbl_ssh = self.ssh[i:i + 72]

        atmos = self.atmos[i - 24:i + 72]

        return atmos, ssh, tide, lbl_ssh


class Data:
    def __init__(self, config):
        with open(f'{config.data_path}/data normalization parameters.yaml') as file:
            self.norm = yaml.safe_load(file)

        self.train_lister = Train(config, self.norm)
        self.train = DataLoader(self.train_lister, batch_size=config.batch_size, num_workers=config.num_workers,
                                pin_memory=True, drop_last=False, shuffle=True)
