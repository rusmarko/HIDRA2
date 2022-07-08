from utils.root import *
from utils.dataset import Data
from utils.training import Training
from hidra2.hidra2 import HIDRA


class Config():
    def __init__(self):
        self.seed = 0

        self.dataset_path = '../data/training data example.pt'

        self.train_params = {
            'range': (None, '2018-12-31 00:00:00'),
        }
        self.flood_thr = 300

        self.lr = .0001
        self.batch_size = 512
        self.no_batches = 1000
        self.epochs = 60

        self.num_workers = 4
        self.weights_folder = 'out'
        self.save_model = True
        self.load_model = False

        self.device = torch.device('cuda:0')

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if not os.path.exists(self.weights_folder):
            os.mkdir(self.weights_folder)


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
    time.print_()
