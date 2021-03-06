## HIDRA2: Deep-Learning Ensemble Storm Surge Forecasting in the Presence of Seiches – the Case of Northern Adriatic

<p align="center">
    <img src="images/logo.png" alt="HIDRA logo" width="300px">
</p>

HIDRA2 is a state-of-the-art deep neural model for sea-level prediction based on past sea level observations and future tidal and atmospheric forecasts.

![Qualitative example of sea level predictions (compared with NEMO, from 2020/10/14).](./images/qualitative_example-2020-10-14.png)


### Setup

To install the required packages, run
```
pip3 install -r requirements.txt
```

### Loading the Pretrained Model

This repository contains the implementation of HIDRA2, which can be inspected in file `src/hidra2/hidra2.py`, and parameters
of the pretrained model trained on 2006-2018. To load the pretrained model, perform one forward pass and visualize the
result, run

```
cd src
python3 predict.py
```

File `data/input example.pt` contains one validation datapoint from 2019. Inputs to the HIDRA2 model are atmospheric, 
ssh and tidal data, each structured as follows:

| Field name | Shape               | Description                                                                                                                                                                                      |
|------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `weather`  | b × 96 × 3 × 9 × 12 | Hourly atmospheric forecasts from 24 h prior the prediction point to 72 h into the future. Third dimension contains pressure and two channels of wind. Last two dimensions are height and width. |
| `ssh`      | b × 72              | Past SSH.                                                                                                                                                                                        |
| `tide`     | b × 144             | Past and future tide.                                                                                                                                                                            |

Here `b` stands for batch dimension. The model outputs 72 points representing hourly sea level forecast.

All inputs and output of HIDRA2 are normalized, the statistics are stored in `data/data normalization parameters.yaml`.

### Training the Model

To train the model, firstly add the training data to `data` folder and change parameters in `src/train.py`. Then, run

```
cd src
python3 train.py
```

File `data/training data example.pt` showcases the format used for training data. It contains a dictionary with six elements,`times`, `weather`, 
`tide`, `ssh`, `valid_i` and `norm`. `norm` contains normalization coefficients, `weather`, `tide` and `ssh` are tensors of an equal size,
where a datapoint at some index belongs to the time defined in `times`. For a range to be valid, all points from 72 h before and after
the prediction point need to be defined, `valid_i` contains those valid indexes of predictions points.

The format of `weather`, `ssh` and `tide` is as folows: 

| Field name | Shape           | Description                                                                                                                                        |
|------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `weather`  | N × 4 × 28 × 36 | Hourly atmospheric forecasts. Second dimension contains pressure, two channels of wind, and temperature. Last two dimensions are height and width. |
| `ssh`      | N               | SSH.                                                                                                                                               |
| `tide`     | N               | Tide.                                                                                                                                              |

where `N` represents the number of hourly datapoints in the training dataset.
