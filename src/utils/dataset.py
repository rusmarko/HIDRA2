from utils.root import *


class CustomDataset(Dataset):
    def __init__(self, config, subset, output_index=False):
        super(CustomDataset, self).__init__()

        self.subset = subset
        self.output_index = output_index

        print('Reading data.')

        data = torch.load(config.dataset_path)
        self.weather = data['weather']
        self.tide = data['tide']
        self.ssh = data['ssh']
        self.times = data['times']
        self.valid_i = []
        self.norm = data['norm']

        # subsample weather
        s = 3
        self.weather = F.conv2d(self.weather, torch.ones(4, 1, s, s) / (s * s), groups=4, stride=s)

        # remove T
        self.weather = self.weather[:, :3]

        if subset == 'train':
            a = pd.to_datetime(config.train_params['range'][0])
            b = pd.to_datetime(config.train_params['range'][1])
        else:
            raise
        if a is None:
            a = pd.Timestamp.min
        if b is None:
            b = pd.Timestamp.max

        for i in data['valid_i']:
            if a <= self.times[i] <= b:
                self.valid_i.append(i)

        print(f'{subset} set: {len(self.valid_i)} instances.')

    def __len__(self):
        return len(self.valid_i)

    def __getitem__(self, i):
        i = self.valid_i[i]

        # checking time
        times = self.times[i - 72:i + 72]
        for t in range(1, len(times)):
            a = times[t - 1]
            b = times[t]
            assert (b - a).seconds == 3600

        ssh = self.ssh[i - 72:i]
        tide = self.tide[i - 72:i + 72]

        lbl_ssh = self.ssh[i:i + 72]

        weather = self.weather[i - 24:i + 72]

        if self.output_index:
            return weather, ssh, tide, lbl_ssh, torch.tensor(i)
        return weather, ssh, tide, lbl_ssh


class Data:
    def __init__(self, config, train=True, output_index=False):
        if train:
            self.train_lister = CustomDataset(config, subset='train', output_index=output_index)
            self.train = DataLoader(self.train_lister, batch_size=config.batch_size, num_workers=config.num_workers,
                                    pin_memory=True, drop_last=False, shuffle=True)
            self.norm = self.train_lister.norm
