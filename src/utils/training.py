import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Training:
    def __init__(self, config, net, data):
        self.config = config
        self.net = net
        self.data = data
        self.norm = self.data.norm

        self.lr = config.lr

    def train(self):
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        optimizer.zero_grad()

        MSELoss = nn.MSELoss()

        self.net = self.net.to(self.config.device)
        self.net.train()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs,
                                                               eta_min=self.config.lr / 10)

        for epoch in tqdm(range(self.config.epochs)):
            for data in self.data.train:
                weather, ssh, tide, lbl_ssh = data
                weather = weather.to(self.config.device)
                ssh = ssh.to(self.config.device)
                tide = tide.to(self.config.device)
                lbl_ssh = lbl_ssh.to(self.config.device)

                y = self.net(weather, ssh, tide)
                loss = MSELoss(y, lbl_ssh)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        torch.save(self.net.state_dict(), f'{self.config.data_path}/HIDRA2 parameters.pth')
        print('Model weights saved to the data folder.')
