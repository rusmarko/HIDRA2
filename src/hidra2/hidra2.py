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

    def forward(self, atmos, ssh, tide):
        """
        :param atmos: b 96 3 9 12 (batch size, time, type, height, width)
        :param ssh: b 72
        :param tide: b 144
        """

        batch_size = atmos.shape[0]

        # atmosphere batched
        x = atmos.reshape(batch_size * 24, 4, 3, *atmos.shape[-2:])  # bx24 4 3 9 12
        y1 = self.wind(x[:, :, 1:].reshape(batch_size * 24, 4 * 2, *atmos.shape[-2:]))  # bx24 512 1 1
        y2 = self.pressure(x[:, :, 0])  # bx24 512 1 1
        y = torch.cat((y1, y2), dim=1)  # bx24 1024 1 1
        x = y.view(batch_size, 24, 1024)  # b 24 1024

        # atmosphere temporal
        x = x.permute(0, 2, 1)  # b 1024 24
        x = self.atmos_temporal(x)  # b 256 20
        for layer in self.atmos:
            x = x + layer(x)
        x = self.atmos_final(x)  # b 32 20
        x = x.view(batch_size, -1)  # b 640
        weather_features = self.atmos_bn(x.unsqueeze(1)).squeeze(1)  # b 640

        # tide
        x = self.tide_temporal(tide[:, -72:].unsqueeze(1))  # b 256 35
        for layer in self.tide:
            x = x + layer(x)
        x = self.tide_final(x)  # b 16 17
        x = x.view(batch_size, -1)  # b 272
        tide_features = self.tide_bn(x.unsqueeze(1)).squeeze(1)  # b 272

        # ssh
        x = torch.cat((
            tide[:, :72].unsqueeze(1),
            ssh.unsqueeze(1),
        ), dim=1)  # b 2 72
        x = self.ssh_temporal(x)  # b 256 35
        for layer in self.ssh:
            x = x + layer(x)
        x = self.ssh_final(x)  # b 16 17
        x = x.view(batch_size, -1)  # b 272
        ssh_features = self.ssh_bn(x.unsqueeze(1)).squeeze(1)  # b 272

        # regression
        x = torch.cat((
            weather_features,
            tide_features,
            ssh_features,
        ), 1)  # b 1184
        x = self.reduce_dim(x)  # b 512
        x = torch.cat((
            x,
            ssh,
        ), 1)  # b 512+72
        for layer in self.regression:
            x = x + layer(x)
        y = self.final(x)  # b 72

        return y
