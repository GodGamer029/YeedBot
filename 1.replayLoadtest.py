import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
import numpy as np
import io
import functools

trainDatasetPath = 'file:E:/replays/scripts/datasets/packed/train.tar'


def dataDecoder(value):
    return torch.from_numpy(np.delete(np.loadtxt(io.BytesIO(value), delimiter=' '), 0, 1))


def main():
    dataset = (
        wds.WebDataset(trainDatasetPath)
        .shuffle(100)
        .decode(wds.handle_extension(".dataset.csv", dataDecoder))
    )

    '''for sample in islice(dataset, 0, 3):
        for key, value in sample.items():
            print(key, repr(value))
        print()

    '''
    batch_size = 20
    dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=1, batch_size=None)
    x, y = next(iter(dataloader))
    print(x, y)

if __name__ == '__main__':
    main() 