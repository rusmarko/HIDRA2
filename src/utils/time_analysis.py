from collections import defaultdict
from time import time

starts = {}
times = defaultdict(float)


def start(key):
    starts[key] = time()


def end(key):
    if key in starts:
        times[key] += time() - starts[key]


def state(key):
    if key in starts:
        return time() - starts[key]
    return -1


def print_(*, reset=False):
    if len(times) > 0 or len(starts) > 0:
        print('time analysis:')
        for key, t in times.items():
            print(f'\t{key}: {t:.4f} s')
        for key in starts:
            if key not in times:
                print(f'\t{key}: {time() - starts[key]:.4f} s')
    if reset:
        starts.clear()
        times.clear()
