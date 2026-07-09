#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from torchvision import datasets, transforms


SPECS = {
    'cifar10': (datasets.CIFAR10, 3, {'train': True}, {'train': False}),
    'cifar100': (datasets.CIFAR100, 3, {'train': True}, {'train': False}),
    'svhn': (datasets.SVHN, 3, {'split': 'train'}, {'split': 'test'}),
    'fashionmnist': (datasets.FashionMNIST, 1, {'train': True}, {'train': False}),
}


def transform(channels: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * channels), tuple([0.5] * channels)),
    ])


def download_one(name: str, data_root: Path, retries: int):
    cls, channels, train_kwargs, test_kwargs = SPECS[name]
    root = data_root / name
    for attempt in range(1, retries + 1):
        try:
            print(f'[{name}] download/check train in {root} attempt {attempt}/{retries}')
            cls(root=str(root), download=True, transform=transform(channels), **train_kwargs)
            print(f'[{name}] download/check test in {root} attempt {attempt}/{retries}')
            cls(root=str(root), download=True, transform=transform(channels), **test_kwargs)
            print(f'[{name}] ready')
            return
        except Exception as exc:
            if attempt == retries:
                raise
            wait_s = 10 * attempt
            print(f'[{name}] failed: {exc}. retry in {wait_s}s')
            time.sleep(wait_s)


def main():
    parser = argparse.ArgumentParser(description='Scarica una volta sola i dataset torchvision usati dalle run.')
    parser.add_argument('--datasets', nargs='+', default=list(SPECS), choices=list(SPECS))
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--retries', type=int, default=3)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        download_one(name, data_root, args.retries)


if __name__ == '__main__':
    main()
