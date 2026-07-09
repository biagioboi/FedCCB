#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


class SmallCNN(nn.Module):
    def __init__(self, channels: int, num_classes: int, image_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class LeafImageFolder(Dataset):
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        leaf_dirs = []
        for directory in sorted(path for path in self.root.rglob('*') if path.is_dir()):
            if any(file.suffix.lower() in self.IMG_EXTENSIONS for file in directory.iterdir() if file.is_file()):
                leaf_dirs.append(directory)
        if not leaf_dirs:
            raise RuntimeError(f'No image leaf directories found in {self.root}')
        self.classes = sorted({directory.name for directory in leaf_dirs})
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []
        for directory in leaf_dirs:
            class_idx = self.class_to_idx[directory.name]
            for file in sorted(directory.iterdir()):
                if file.is_file() and file.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append((str(file), class_idx))
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class LabelSwapDataset(Dataset):
    def __init__(self, base: Dataset, label_map: dict[int, int], fraction: float, seed: int):
        self.base = base
        self.label_map = label_map
        rng = random.Random(seed)
        self.flip = [rng.random() < fraction for _ in range(len(base))]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        y = int(y)
        if self.flip[idx] and y in self.label_map:
            y = self.label_map[y]
        return x, y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_spec(name: str):
    specs = {
        'cifar10': (datasets.CIFAR10, 3, 10, 32, {'train': True}, {'train': False}),
        'cifar100': (datasets.CIFAR100, 3, 100, 32, {'train': True}, {'train': False}),
        'svhn': (datasets.SVHN, 3, 10, 32, {'split': 'train'}, {'split': 'test'}),
        'fashionmnist': (datasets.FashionMNIST, 1, 10, 28, {'train': True}, {'train': False}),
    }
    if name not in specs:
        raise ValueError(f'Dataset non supportato: {name}')
    return specs[name]


def make_transforms(channels: int, train: bool):
    ops = []
    if train and channels == 3:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(32, scale=(0.8, 1.0))])
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple([0.5] * channels), tuple([0.5] * channels)))
    return transforms.Compose(ops)


def load_imagefolder_fallback(name: str, data_root: Path, channels: int, image_size: int):
    root = data_root / name
    train_dir = root / 'train'
    test_dir = root / 'test'
    if not test_dir.exists():
        test_dir = root / 'valid'
    if not train_dir.exists() or not test_dir.exists():
        return None
    if name == 'cifar100':
        train_ds = LeafImageFolder(train_dir, transform=make_transforms(channels, True))
        test_ds = LeafImageFolder(test_dir, transform=make_transforms(channels, False))
    else:
        train_ds = ImageFolder(str(train_dir), transform=make_transforms(channels, True))
        test_ds = ImageFolder(str(test_dir), transform=make_transforms(channels, False))
    num_classes = len(train_ds.classes)
    return train_ds, test_ds, channels, num_classes, image_size


def load_datasets(name: str, data_root: Path, download: bool = True):
    cls, channels, num_classes, image_size, train_kwargs, test_kwargs = dataset_spec(name)
    fallback = load_imagefolder_fallback(name, data_root, channels, image_size)
    if fallback is not None:
        print(f'[{name}] using local ImageFolder cache from {data_root / name}')
        return fallback
    try:
        train_ds = cls(root=str(data_root / name), download=download, transform=make_transforms(channels, True), **train_kwargs)
        test_ds = cls(root=str(data_root / name), download=download, transform=make_transforms(channels, False), **test_kwargs)
        return train_ds, test_ds, channels, num_classes, image_size
    except Exception as exc:
        fallback = load_imagefolder_fallback(name, data_root, channels, image_size)
        if fallback is not None:
            print(f'[{name}] using ImageFolder fallback from {data_root / name}')
            return fallback
        raise RuntimeError(
            f"Dataset {name} non trovato in {data_root / name}. "
            f"Prova: bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh {name}"
        ) from exc


def get_label(dataset, idx: int) -> int:
    if isinstance(dataset, Subset):
        return get_label(dataset.dataset, dataset.indices[idx])
    if hasattr(dataset, 'targets'):
        return int(dataset.targets[idx])
    if hasattr(dataset, 'labels'):
        return int(dataset.labels[idx])
    if hasattr(dataset, 'samples'):
        return int(dataset.samples[idx][1])
    return int(dataset[idx][1])


def split_federated(dataset, num_clients: int, alpha: float, server_fraction: float, num_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    server_size = int(len(indices) * server_fraction)
    server_indices = indices[:server_size].tolist()
    pool_indices = indices[server_size:].tolist()

    by_class = [[] for _ in range(num_classes)]
    for idx in pool_indices:
        by_class[get_label(dataset, idx)].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for class_indices in by_class:
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(np.array(class_indices), cuts)):
            client_indices[client_id].extend(split.tolist())

    return client_indices, server_indices


def make_loaders(train_ds, test_ds, args, num_classes: int):
    client_indices, server_indices = split_federated(train_ds, args.num_clients, args.alpha, args.server_fraction, num_classes, args.seed)
    label_map = {i: (i + args.label_shift) % num_classes for i in range(num_classes)}
    malicious = set(int(x) for x in args.malicious_clients.split(',') if x.strip())
    rng = random.Random(args.seed)
    client_loaders = []
    active_malicious = []
    client_num_workers = args.num_workers
    if args.num_workers > 0 and args.client_parallelism != 1:
        client_num_workers = 0
        print(
            '[data] forcing client DataLoader num_workers=0 because threaded client training '
            f'is enabled (client_parallelism={args.client_parallelism})'
        )
    for client_id, indices in enumerate(client_indices):
        subset = Subset(train_ds, indices)
        if client_id in malicious and rng.random() < args.malicious_probability:
            subset = LabelSwapDataset(subset, label_map, args.label_swap_fraction, args.seed + client_id)
            active_malicious.append(client_id)
        client_loaders.append(DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=client_num_workers))
    server_loader = DataLoader(Subset(train_ds, server_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return client_loaders, server_loader, test_loader, active_malicious


def train_local(model, loader, epochs, lr, device):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_clients(model_factory, global_state, client_loaders, epochs, lr, device, client_parallelism):
    def train_one(loader):
        model = model_factory().to(device)
        model.load_state_dict(global_state, strict=True)
        return train_local(model, loader, epochs, lr, device)

    if client_parallelism == 1 or len(client_loaders) <= 1:
        return [train_one(loader) for loader in client_loaders]

    max_workers = len(client_loaders) if client_parallelism <= 0 else min(client_parallelism, len(client_loaders))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(train_one, client_loaders))


def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction='sum').item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def flatten_state(state):
    return torch.cat([v.float().reshape(-1) for v in state.values()])


def state_update(global_state, local_state):
    return {k: local_state[k].float() - global_state[k].float() for k in global_state}


def add_update(global_state, update, lr=1.0):
    return {k: (global_state[k].float() + lr * update[k]).clone() for k in global_state}


def average_updates(updates, weights=None):
    if not updates:
        raise ValueError('average_updates received an empty update list')
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    total = {k: torch.zeros_like(v.float()) for k, v in updates[0].items()}
    for weight, update in zip(weights, updates):
        for k in total:
            total[k] += float(weight) * update[k].float()
    return total


def vector_to_update(vec, template):
    out, start = {}, 0
    for k, v in template.items():
        n = v.numel()
        out[k] = vec[start:start+n].reshape_as(v).clone()
        start += n
    return out


def aggregate_krum(updates):
    vectors = torch.stack([flatten_state(u) for u in updates])
    n = len(updates)
    neighbor_count = max(1, n - 2)
    scores = []
    for i in range(n):
        dists = torch.sum((vectors[i] - vectors) ** 2, dim=1)
        scores.append(torch.topk(dists, k=neighbor_count + 1, largest=False).values[1:].sum().item())
    return updates[int(np.argmin(scores))]


def aggregate_trimmed_mean(updates, trim_ratio=0.2):
    vectors = torch.stack([flatten_state(u) for u in updates])
    trim = min(int(len(updates) * trim_ratio), max(0, (len(updates) - 1) // 2))
    sorted_vectors, _ = torch.sort(vectors, dim=0)
    kept = sorted_vectors[trim:len(updates)-trim] if trim > 0 else sorted_vectors
    return vector_to_update(kept.mean(dim=0), updates[0])


def aggregate_clipped_clustering(updates):
    vectors = torch.stack([flatten_state(u) for u in updates])
    norms = torch.norm(vectors, dim=1).clamp_min(1e-12)
    clip = torch.median(norms)
    clipped = vectors * torch.clamp(clip / norms, max=1.0).unsqueeze(1)
    if len(updates) < 3 or KMeans is None:
        return vector_to_update(clipped.mean(dim=0), updates[0])
    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(clipped.numpy())
    counts = np.bincount(labels)
    chosen = int(np.argmax(counts))
    return vector_to_update(clipped[labels == chosen].mean(dim=0), updates[0])


def cosine_weights(updates, server_update, floor=0.0):
    server_vec = flatten_state(server_update)
    weights = []
    for update in updates:
        score = F.cosine_similarity(flatten_state(update), server_vec, dim=0).item()
        weights.append(max(floor, score))
    total = sum(weights)
    if total <= 1e-12:
        return [1.0 / len(updates)] * len(updates)
    return [w / total for w in weights]


def aggregate_rflpa(updates, server_update):
    return average_updates(updates, cosine_weights(updates, server_update))


def aggregate_adaaggrl(updates, server_update, history_vectors):
    server_vec = flatten_state(server_update)
    weights = []
    new_history = []
    for idx, update in enumerate(updates):
        vec = flatten_state(update)
        trust = max(0.0, F.cosine_similarity(vec, server_vec, dim=0).item())
        if idx < len(history_vectors):
            drift = torch.norm(vec - history_vectors[idx]).item() / math.sqrt(max(vec.numel(), 1))
            stability = math.exp(-drift)
            hist = 0.7 * history_vectors[idx] + 0.3 * vec
        else:
            stability = 1.0
            hist = vec
        weights.append(trust * stability)
        new_history.append(hist.detach().cpu())
    total = sum(weights)
    if total <= 1e-12:
        weights = [1.0 / len(updates)] * len(updates)
    else:
        weights = [w / total for w in weights]
    return average_updates(updates, weights), new_history


def aggregate_fedlad(updates, max_condition_number=1e4, min_residual_ratio=1e-3):
    vectors = torch.stack([flatten_state(update) for update in updates]).float()
    norms = torch.norm(vectors, dim=1).clamp_min(1e-12)
    normalized = vectors / norms.unsqueeze(1)

    selected_indices = []
    basis = None
    residual_ratios = []
    leverage_scores = []

    median_norm = torch.median(norms).item()
    mad_norm = torch.median(torch.abs(norms - median_norm)).item() + 1e-12
    norm_z = torch.abs(norms - median_norm) / mad_norm
    candidate_order = torch.argsort(norm_z).cpu().tolist()

    for idx in candidate_order:
        vector = normalized[idx]
        if basis is None:
            selected_indices.append(idx)
            basis = vector.unsqueeze(0)
            residual_ratios.append(1.0)
            leverage_scores.append(float(norm_z[idx].item()))
            continue

        coefficients = torch.linalg.lstsq(basis.T, vector).solution
        projection = basis.T @ coefficients
        residual = vector - projection
        residual_ratio = (torch.norm(residual) / torch.norm(vector).clamp_min(1e-12)).item()
        trial_basis = torch.cat([basis, vector.unsqueeze(0)], dim=0)
        singular_values = torch.linalg.svdvals(trial_basis)
        condition_number = (singular_values.max() / singular_values.min().clamp_min(1e-12)).item()

        if residual_ratio >= min_residual_ratio and condition_number <= max_condition_number:
            selected_indices.append(idx)
            basis = trial_basis
            residual_ratios.append(residual_ratio)
            leverage_scores.append(float(norm_z[idx].item()))

    fallback_used = False
    if not selected_indices:
        selected_indices = list(range(len(updates)))
        fallback_used = True

    selected_updates = [updates[idx] for idx in selected_indices]
    selected_norms = np.array([norms[idx].item() for idx in selected_indices], dtype=float)
    norm_center = np.median(selected_norms)
    norm_distance = np.abs(selected_norms - norm_center)
    weights = 1.0 / np.maximum(norm_distance + 1e-6, 1e-6)
    weights = (weights / weights.sum()).tolist()

    return average_updates(selected_updates, weights), {
        'selected_clients': selected_indices,
        'selected_count': len(selected_indices),
        'residual_ratios': residual_ratios,
        'norm_z_selected': leverage_scores,
        'fallback_fedavg': fallback_used,
    }


def client_loss_profile(weights, model_factory, server_loader, device):
    model = model_factory().to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    num_classes = getattr(model.classifier[-1], 'out_features', 10)
    loss_sum = torch.zeros(num_classes, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for x, y in server_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses = F.cross_entropy(logits, y, reduction='none').detach().cpu()
            labels = y.detach().cpu()
            for cls in range(num_classes):
                mask = labels == cls
                if mask.any():
                    loss_sum[cls] += losses[mask].sum().double()
                    counts[cls] += mask.sum().double()
    profile = loss_sum / counts.clamp_min(1.0)
    observed = counts > 0
    if observed.any():
        fill = profile[observed].mean()
        profile[~observed] = fill
    return profile.float()


def aggregate_sherpa(global_state, updates, model_factory, server_loader, device):
    profiles = []
    norms = []
    for update in updates:
        candidate_weights = add_update(global_state, update)
        profiles.append(client_loss_profile(candidate_weights, model_factory, server_loader, device))
        norms.append(torch.norm(flatten_state(update)).item())

    profile_matrix = torch.stack(profiles)
    norm_column = torch.tensor(norms, dtype=torch.float32).reshape(-1, 1)
    feature_matrix = torch.cat([profile_matrix, norm_column], dim=1)

    median = feature_matrix.median(dim=0).values
    mad = (feature_matrix - median).abs().median(dim=0).values.clamp_min(1e-6)
    z = (feature_matrix - median).abs() / mad
    anomaly_scores = z.mean(dim=1).cpu().numpy().tolist()

    score_array = np.array(anomaly_scores, dtype=float)
    q1, q3 = np.percentile(score_array, [25, 75])
    threshold = q3 + 1.5 * max(q3 - q1, 1e-6)
    selected_indices = [idx for idx, score in enumerate(anomaly_scores) if score <= threshold]
    fallback_used = False
    if not selected_indices:
        selected_indices = list(range(len(updates)))
        fallback_used = True

    selected_scores = np.array([anomaly_scores[idx] for idx in selected_indices], dtype=float)
    inv_scores = 1.0 / np.maximum(selected_scores, 1e-6)
    weights = (inv_scores / inv_scores.sum()).tolist()
    selected_updates = [updates[idx] for idx in selected_indices]
    return average_updates(selected_updates, weights), {
        'anomaly_scores': anomaly_scores,
        'selected_clients': selected_indices,
        'threshold': float(threshold),
        'fallback_fedavg': fallback_used,
    }


def aggregate_fedgreed(global_state, updates, model_factory, server_loader, device):
    selected = []
    best_loss = float('inf')
    remaining = list(range(len(updates)))
    while remaining:
        trial_losses = []
        for idx in remaining:
            candidate = selected + [idx]
            update = average_updates([updates[i] for i in candidate])
            model = model_factory().to(device)
            model.load_state_dict(add_update(global_state, update), strict=True)
            loss, _ = evaluate(model, server_loader, device)
            trial_losses.append((loss, idx))
        loss, chosen = min(trial_losses)
        if loss <= best_loss or not selected:
            selected.append(chosen)
            remaining.remove(chosen)
            best_loss = loss
        else:
            break
    if not selected:
        selected = list(range(len(updates)))
    return average_updates([updates[i] for i in selected]), selected


def aggregate(method, global_state, local_states, server_state, model_factory, server_loader, device, history_vectors):
    updates = [state_update(global_state, state) for state in local_states]
    server_update = state_update(global_state, server_state) if server_state is not None else None
    info = {}
    if method in {'fedavg', 'fedsgd'}:
        return average_updates(updates), history_vectors, info
    if method == 'proposed_confidence':
        scores = []
        log_c = math.log(max(2, getattr(model_factory().classifier[-1], 'out_features', 10)))
        for update in updates:
            candidate = model_factory().to(device)
            candidate.load_state_dict(add_update(global_state, update), strict=True)
            loss, _ = evaluate(candidate, server_loader, device)
            scores.append(math.exp(-max(0.0, loss - log_c)))
        if len(scores) >= 3 and KMeans is not None:
            labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(np.array(scores).reshape(-1, 1))
            means = []
            for cluster in range(2):
                cluster_scores = [s for s, label in zip(scores, labels) if label == cluster]
                means.append(float(np.mean(cluster_scores)) if cluster_scores else float('-inf'))
            chosen = int(np.argmax(means))
            selected = [update for update, label in zip(updates, labels) if label == chosen]
        else:
            threshold = float(np.median(scores)) if scores else float('-inf')
            selected = [update for update, score in zip(updates, scores) if score >= threshold]
        fallback_used = False
        if not selected:
            selected = updates
            fallback_used = True
        return average_updates(selected), history_vectors, {'confidence_scores': scores, 'selected_clients': len(selected), 'fallback_fedavg': fallback_used}
    if method == 'krum':
        return aggregate_krum(updates), history_vectors, info
    if method == 'trimmed_mean':
        return aggregate_trimmed_mean(updates), history_vectors, info
    if method == 'fltrust':
        return aggregate_rflpa(updates, server_update), history_vectors, {'server_guided': True}
    if method == 'clipped_clustering':
        return aggregate_clipped_clustering(updates), history_vectors, info
    if method == 'rflpa':
        return aggregate_rflpa(updates, server_update), history_vectors, {'server_guided': True}
    if method == 'adaaggrl':
        update, new_history = aggregate_adaaggrl(updates, server_update, history_vectors)
        return update, new_history, {'adaptive_history': True}
    if method == 'fedgreed':
        update, selected = aggregate_fedgreed(global_state, updates, model_factory, server_loader, device)
        return update, history_vectors, {'selected_clients': selected}
    if method == 'sherpa':
        update, sherpa_info = aggregate_sherpa(global_state, updates, model_factory, server_loader, device)
        return update, history_vectors, sherpa_info
    if method == 'fedlad':
        update, fedlad_info = aggregate_fedlad(updates)
        return update, history_vectors, fedlad_info
    raise ValueError(f'Metodo non supportato: {method}')


def write_metrics(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'svhn', 'fashionmnist'])
    parser.add_argument('--method', required=True, choices=['proposed_confidence', 'fedavg', 'fedsgd', 'krum', 'trimmed_mean', 'fltrust', 'clipped_clustering', 'rflpa', 'adaaggrl', 'fedgreed', 'sherpa', 'fedlad'])
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--attack', default='medio')
    parser.add_argument('--num-rounds', type=int, default=50)
    parser.add_argument('--num-clients', type=int, default=25)
    parser.add_argument('--server-fraction', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--server-epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label-swap-fraction', type=float, default=0.5)
    parser.add_argument('--label-shift', type=int, default=1)
    parser.add_argument('--malicious-probability', type=float, default=0.5)
    parser.add_argument('--malicious-clients', default='1,3,6,7,9,10,12,13,15,19,20,23')
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--no-download', action='store_true', help='Disabilita il download automatico e usa solo dataset gia presenti in cache.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--client-parallelism', type=int, default=0, help='Numero di client da addestrare in parallelo; 0 usa tutti i client.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'args.json').write_text(json.dumps(vars(args), indent=2) + '\n', encoding='utf-8')

    train_ds, test_ds, channels, num_classes, image_size = load_datasets(args.dataset, Path(args.data_root), download=not args.no_download)
    client_loaders, server_loader, test_loader, active_malicious = make_loaders(train_ds, test_ds, args, num_classes)
    device = torch.device(args.device)

    def model_factory():
        return SmallCNN(channels, num_classes, image_size)

    global_model = model_factory().to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    history_vectors = []
    metrics = []

    for round_idx in range(1, args.num_rounds + 1):
        epochs = 1 if args.method == 'fedsgd' else args.local_epochs
        local_states = train_clients(
            model_factory,
            global_state,
            client_loaders,
            epochs,
            args.lr,
            device,
            args.client_parallelism,
        )

        server_model = model_factory().to(device)
        server_model.load_state_dict(global_state, strict=True)
        server_state = train_local(server_model, server_loader, args.server_epochs, args.lr, device)

        update, history_vectors, info = aggregate(args.method, global_state, local_states, server_state, model_factory, server_loader, device, history_vectors)
        global_state = add_update(global_state, update)

        eval_model = model_factory().to(device)
        eval_model.load_state_dict(global_state, strict=True)
        loss, acc = evaluate(eval_model, test_loader, device)
        row = {'round': round_idx, 'loss': loss, 'accuracy': acc, 'active_malicious_clients': json.dumps(active_malicious), 'info': json.dumps(info)}
        metrics.append(row)
        print(json.dumps(row))
        torch.save({'state_dict': global_state, 'round': round_idx, 'args': vars(args)}, output_dir / 'last_checkpoint.pt')
        write_metrics(output_dir / 'metrics.csv', metrics)


if __name__ == '__main__':
    main()
