import torch
import yaml
from tqdm import tqdm

from hidra2.hidra2 import HIDRA

data_path = '../data'

test_data = torch.load(f'{data_path}/test.pth', map_location='cpu')

with open(f'{data_path}/data normalization parameters.yaml') as file:
    norm = yaml.safe_load(file)

# loading model
hidra = HIDRA()
hidra.load_state_dict(torch.load(f'{data_path}/HIDRA2 parameters.pth', map_location='cpu'))
hidra.eval()

# making predictions
predictions = {}
for time in tqdm(test_data):
    instance = test_data[time]
    atmos = instance['atmos']
    tide = instance['tide']
    ssh = instance['ssh']

    ground_truth = ssh[72:]

    # normalizing
    atmos[:, :, 0] = (atmos[:, :, 0] - norm['atmos']['mean'][0]) / norm['atmos']['std'][0]
    atmos[:, :, 1] = (atmos[:, :, 1] - norm['atmos']['mean'][1]) / norm['atmos']['std'][1]
    atmos[:, :, 2] = (atmos[:, :, 2] - norm['atmos']['mean'][2]) / norm['atmos']['std'][2]
    tide = (tide - norm['tide']['mean']) / norm['tide']['std']
    ssh = (ssh[:72] - norm['ssh']['mean']) / norm['ssh']['std']

    # forward pass
    p = hidra(atmos, ssh.unsqueeze(0).expand(50, -1), tide.unsqueeze(0).expand(50, -1)).detach()
    p = p * norm['ssh']['std'] + norm['ssh']['mean']

    predictions[time] = {
        'predictions': p,
        'ground truth': ground_truth,
    }

torch.save(predictions, f'{data_path}/predictions.pth')
print('Predictions saved to the data folder.')
