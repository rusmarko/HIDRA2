from utils.root import *


class Training:
    def __init__(self, config, net, data):
        self.config = config
        self.net = net
        self.data = data
        self.norm = self.data.norm

        self.name = self.net.__class__.__name__
        self.weights_path = f'{config.weights_folder}/weights.pth'

        if config.save_model:
            assert os.path.exists(os.path.split(self.weights_path)[0]), 'Weights path folder does not exist.'
            assert not os.path.exists(self.weights_path), 'Weights already there.'

        self.lr = config.lr
        self.optimizer = None
        self.best_mae = None

    def save_params(self):
        data = {
            'net': self.net.state_dict(),
        }
        torch.save(data, self.weights_path)

    def load_params(self):
        data = torch.load(self.weights_path, map_location=self.config.device)
        self.net.load_state_dict(data['net'])

    def train(self):
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        MSELoss = nn.MSELoss()

        self.net = self.net.to(self.config.device)
        self.net.train()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs,
                                                               eta_min=self.config.lr / 10)

        if self.config.load_model:
            print('Loading params.')
            self.load_params()

        time.start('train')

        for epoch in range(self.config.epochs):
            losses = []
            maes = []
            time.start('epoch')

            bar = tqdm(self.data.train, total=self.config.no_batches)
            for i, data in enumerate(bar):
                if i >= self.config.no_batches:
                    break

                weather, ssh, tide, lbl_ssh = data
                weather = weather.to(self.config.device)
                ssh = ssh.to(self.config.device)
                tide = tide.to(self.config.device)
                lbl_ssh = lbl_ssh.to(self.config.device)

                y = self.net(weather, ssh, tide)
                loss = MSELoss(y, lbl_ssh)

                with torch.no_grad():
                    mae = torch.mean(torch.abs(y - lbl_ssh))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                maes.append(mae.item())

            if self.config.save_model:
                self.save_params()

            print(f'epoch: {epoch:3}, '
                  f'loss: {sum(losses) / len(losses):.4f}, '
                  f'mae: {sum(maes) / len(maes) * self.norm["ssh"]["std"]:.4f}, '
                  f'lr: {self.optimizer.param_groups[0]["lr"]}, '
                  f'time: {time.state("epoch"):2.1f} s')

            scheduler.step()

        if self.config.save_model:
            self.save_params()
