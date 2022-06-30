import matplotlib.pyplot as plt
import torch
import yaml

from hidra2.hidra2 import HIDRA

# loading input example
atmosphere, past_ssh, tide, ground_truth = torch.load('../data/input example.pt', map_location='cpu')

# normalization parameters
with open('../data/data normalization parameters.yaml') as file:
    norm = yaml.safe_load(file)

# loading model
hidra = HIDRA()
hidra.load_state_dict(torch.load('../data/HIDRA2 parameters.pt', map_location='cpu')['net'])
hidra.eval()

# predicting
predictions = hidra(atmosphere, past_ssh, tide)

# denormalizing
predictions = predictions.squeeze() * norm['ssh']['std'] + norm['ssh']['mean']
ground_truth = ground_truth.squeeze() * norm['ssh']['std'] + norm['ssh']['mean']

# plotting
plt.plot(predictions.detach().numpy())
plt.plot(ground_truth.detach().numpy())
plt.legend(['pred.', 'GT'])
plt.show()
