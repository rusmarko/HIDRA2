from math import sqrt

import pandas as pd
import torch


class Metrics:
    def __init__(self, floods_thr=300, acc_thr=10):
        self.floods_thr = floods_thr
        self.acc_thr = acc_thr

        # overall
        self.maes = self.rmses = self.bias = self.n = self.ok = 0

        # floods
        self.fmaes = self.frmses = self.fbias = self.fn = self.fok = 0

    def add(self, true, pred):
        """
        :param true: Tensor containing 72 GT points.
        :param pred: Tensor containing 72 predictions.
        """
        self.maes += torch.sum(torch.abs(true - pred)).item()
        self.rmses += torch.sum(torch.square(true - pred)).item()
        self.bias += torch.sum(pred - true).item()
        self.ok += torch.sum(torch.abs(true - pred) <= self.acc_thr).item()
        self.n += true.shape[0]

        # floods
        mask = true >= self.floods_thr
        self.fmaes += torch.sum(torch.abs(true - pred) * mask).item()
        self.frmses += torch.sum(torch.square(true - pred) * mask).item()
        self.fbias += torch.sum((pred - true) * mask).item()
        self.fok += torch.sum((torch.abs(true - pred) <= self.acc_thr) * mask).item()
        self.fn += torch.sum(mask).item()

    def print(self):
        print(f'{self.n} points')
        print(f'{self.fn} / {self.n} ({self.fn / self.n * 100:.1f} %) floods')
        print()
        print(f'ALL: mae: {self.maes / self.n:.2f}, '
              f'rmse: {sqrt(self.rmses / self.n):.2f}, '
              f'bias: {self.bias / self.n:.2f}, '
              f'acc: {self.ok / self.n * 100:.2f}')
        print(f'FLOODS: mae: {self.fmaes / self.fn:.2f}, '
              f'rmse: {sqrt(self.frmses / self.fn):.2f}, '
              f'bias: {self.fbias / self.fn:.2f}, '
              f'acc: {self.fok / self.fn * 100:.2f}')


data_path = '../data'

predictions = torch.load(f'{data_path}/predictions.pth')

# standard measures
metric = Metrics()

times = sorted(list(predictions.keys()))
print(f'from {times[0]} to {times[-1]}, {len(times)} days')

for time in predictions:
    pred = predictions[time]['predictions']
    ground_truth = predictions[time]['ground truth']

    pred = torch.mean(pred, dim=0)

    metric.add(ground_truth, pred)

metric.print()


def precision_recall(ssh, predictions, window=3, tolerance=10, floods_thr=300, prediction_horizon=72):
    """
    :param ssh: Dict containing hourly SSH measurements of all predicted time points.
    :param predictions: Dict containing predictions in the format:
            {day: {'predictions': tensor of #ensemblesx72 elements}, ...}.
    :param window: Half the window size in hours which is a minimal distance between two points to be considered
            two different instances of floods.
    :param tolerance: If prediction lies within the tolerance of ground truth (in cm), it is
            considered for the flood to be detected properly, even if prediction is below `floods_thr`.
    :param floods_thr: Local maximum of the SSH signal to be considered flood if above or equal to it (in cm).
    :param prediction_horizon: Prediction horizon (in h).
    """

    times = sorted(list(ssh.keys()))
    temp = torch.zeros(len(times), dtype=torch.float64)
    for i, time in enumerate(times):
        temp[i] = ssh[time]
    ssh_list = temp

    # Retrieving peaks (=local maximums) of the SSH with respect to the window size.
    peaks = set()
    for i in range(len(times)):

        # Getting boundaries of the signal withing the time window.
        a = max(0, i - window)
        b = min(len(times), i + window + 1)
        for j in range(a, i):
            if (times[j + 1] - times[j]).total_seconds() != 3600:
                a = j + 1
        for j in range(i, b - 1):
            if (times[j + 1] - times[j]).total_seconds() != 3600:
                b = j + 1
                break

        # Checking if SSH at index i is local maximum.
        if ssh_list[i] == torch.max(ssh_list[a:b]):
            peaks.add(times[i])

    tp = fn = fp = 0

    # Aggregating scores for all predictions.
    for time in predictions:
        pred = predictions[time]['predictions']
        pred = torch.mean(pred, dim=0)

        # Finding local maximums in the predicted signal.
        pred_peaks = {}
        pred_peaks_i = {}
        for i in range(1, prediction_horizon - 1):  # Local maximum cannot be detected at the interval extremes.
            time_temp = time + pd.to_timedelta(i, 'h')
            a = max(0, i - window)
            b = min(72, i + window + 1)
            if pred[i] == torch.max(pred[a:b]):
                pred_peaks[time_temp] = pred[i]
                pred_peaks_i[time_temp] = i
        gt_peaks = {}
        gt_peaks_i = {}
        for i in range(-2 * window,
                       prediction_horizon + 2 * window):  # Looking also around the prediction horizon (important for FP).
            time_temp = time + pd.to_timedelta(i, 'h')
            if time_temp in peaks:
                gt_peaks[time_temp] = ssh[time_temp]
                gt_peaks_i[time_temp] = i

        # TP, FN
        for peak in gt_peaks:
            if not (time <= peak < time + pd.to_timedelta(prediction_horizon, 'h')) or \
                    gt_peaks[peak] < floods_thr:  # Here looking only at the peaks in the prediction horizon.
                continue

            # We now have flood in the ground truth, which is either TP or FN.
            is_in_pred = False
            thr = min(gt_peaks[peak] - tolerance, floods_thr)
            for i in range(-window, window + 1):
                time_temp = peak + pd.to_timedelta(i, 'h')
                if time_temp in pred_peaks and pred_peaks[time_temp] >= thr:
                    is_in_pred = True
                    break

            # The case when peak in the prediction signal is at the interval extremes.
            if not is_in_pred:
                i = gt_peaks_i[peak]
                if i < window or i >= prediction_horizon - window:
                    a = max(0, i - window)
                    b = min(prediction_horizon, i + window + 1)
                    if torch.max(pred[a:b]) >= thr:
                        is_in_pred = True

            # Flood is either TP or FN.
            if is_in_pred:
                tp += 1
            else:
                fn += 1

        # FP
        for peak in pred_peaks:
            if pred_peaks[peak] < floods_thr:
                continue
            is_in_gt = False
            thr = min(pred_peaks[peak] - tolerance, floods_thr)
            for i in range(-window, window + 1):
                time_temp = peak + pd.to_timedelta(i, 'h')
                if time_temp in gt_peaks and gt_peaks[time_temp] >= thr:
                    is_in_gt = True
                    break
            if not is_in_gt:
                fp += 1

    recall = tp / (tp + fn) if tp + fn > 0 else -1
    precision = tp / (tp + fp) if tp + fp > 0 else -1
    f1 = 2 / (recall ** -1 + precision ** -1) if recall != -1 and precision != -1 else -1

    return precision, recall, f1


ssh = torch.load(f'{data_path}/ssh.pth')

precision, recall, f1 = precision_recall(ssh, predictions)

print(f'recall: {recall * 100:.2f}, '
      f'precision: {precision * 100:.2f}, '
      f'F1: {f1 * 100:.2f}')
