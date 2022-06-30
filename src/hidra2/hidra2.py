import torch
import torch.nn as nn


class HIDRA(nn.Module):
    def __init__(self):
        super(HIDRA, self).__init__()

        self.wind = nn.Sequential(
            nn.Conv2d(2 * 4, 64, kernel_size=3, stride=2, padding=0), nn.ReLU(), nn.Dropout(),
            nn.Conv2d(64, 512, kernel_size=(4, 5), padding=0),
        )

        self.pressure = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=0), nn.ReLU(), nn.Dropout(),
            nn.Conv2d(64, 512, kernel_size=(4, 5), padding=0),
        )

        self.atmos_temporal = nn.Conv1d(1024, 256, kernel_size=5)
        self.atmos = nn.ModuleList([
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout()),
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout()),
        ])
        self.atmos_final = nn.Sequential(
            nn.Conv1d(256, 32, kernel_size=1),
        )
        self.atmos_bn = nn.BatchNorm1d(1)

        self.tide_temporal = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, stride=2), nn.ReLU(), nn.Dropout(),
        )
        self.tide = nn.ModuleList([
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout()),
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout()),
        ])
        self.tide_final = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(256, 16, kernel_size=1),
        )
        self.tide_bn = nn.BatchNorm1d(1)

        self.ssh_temporal = nn.Sequential(
            nn.Conv1d(2, 256, kernel_size=3, stride=2), nn.ReLU(), nn.Dropout(),
        )
        self.ssh = nn.ModuleList([
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout()),
            nn.Sequential(nn.Conv1d(256, 256, kernel_size=1), nn.SELU(), nn.Dropout()),
        ])
        self.ssh_final = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(256, 16, kernel_size=1),
        )
        self.ssh_bn = nn.BatchNorm1d(1)

        self.reduce_dim = nn.Sequential(
            nn.Linear(1184, 512),
        )

        self.regression = nn.ModuleList([
            nn.Sequential(nn.Linear(512 + 72, 512 + 72), nn.SELU()),
            nn.Sequential(nn.Linear(512 + 72, 512 + 72), nn.SELU()),
        ])
        self.final = nn.Linear(512 + 72, 72)

    def forward(self, weather, ssh, tide):
        """
        :param weather: 2 96 3 9 12 (batch size, time, type, height, width)
        :param ssh: 2 72
        :param tide: 2 144
        """

        batch_size = weather.shape[0]

        # atmosphere batched
        x = weather.view(batch_size * 24, 4, 3, *weather.shape[-2:])  # 2x24 4 3 9 12
        y1 = self.wind(x[:, :, 1:].reshape(batch_size * 24, 4 * 2, *weather.shape[-2:]))  # 2x24 512 1 1
        y2 = self.pressure(x[:, :, 0])  # 2x24 512 1 1
        y = torch.cat((y1, y2), dim=1)  # 2x24 1024 1 1
        x = y.view(batch_size, 24, 1024)  # 2 24 1024

        # atmosphere temporal
        x = x.permute(0, 2, 1)  # 2 1024 24
        x = self.atmos_temporal(x)  # 2 256 20
        for layer in self.atmos:
            x = x + layer(x)
        x = self.atmos_final(x)  # 2 32 20
        x = x.view(batch_size, -1)  # 2 640
        weather_features = self.atmos_bn(x.unsqueeze(1)).squeeze(1)  # 2 640

        # tide
        x = self.tide_temporal(tide[:, -72:].unsqueeze(1))  # 2 256 35
        for layer in self.tide:
            x = x + layer(x)
        x = self.tide_final(x)  # 2 16 17
        x = x.view(batch_size, -1)  # 2 272
        tide_features = self.tide_bn(x.unsqueeze(1)).squeeze(1)  # 2 272

        # ssh
        x = torch.cat((
            tide[:, :72].unsqueeze(1),
            ssh.unsqueeze(1),
        ), dim=1)  # 2 2 72
        x = self.ssh_temporal(x)  # 2 256 35
        for layer in self.ssh:
            x = x + layer(x)
        x = self.ssh_final(x)  # 2 16 17
        x = x.view(batch_size, -1)  # 2 272
        ssh_features = self.ssh_bn(x.unsqueeze(1)).squeeze(1)  # 2 272

        # regression
        x = torch.cat((
            weather_features,
            tide_features,
            ssh_features,
        ), 1)  # 2 1184
        x = self.reduce_dim(x)  # 2 512
        x = torch.cat((
            x,
            ssh,
        ), 1)  # 2 512+72
        for layer in self.regression:
            x = x + layer(x)
        y = self.final(x)  # 2 72

        return y
