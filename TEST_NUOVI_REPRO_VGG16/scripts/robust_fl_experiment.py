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
    from sklearn.cluster import KMeans, AgglomerativeClustering
except Exception:
    KMeans = None
    AgglomerativeClustering = None

try:
    import shap
except Exception:
    shap = None


VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_vgg16_features(channels: int) -> nn.Sequential:
    layers = []
    in_channels = channels
    for v in VGG16_CFG:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG16 (Simonyan & Zisserman, ICLR 2015) with BatchNorm, adapted for small
    (28-32px) federated-learning images: ceil-mode max-pools plus a trailing
    adaptive average pool avoid degenerate spatial sizes, and the classifier
    head is a single Linear layer on the pooled 512-d features (as in the
    VGG16 configuration previously used for this project's experiments)."""

    def __init__(self, channels: int, num_classes: int, image_size: int):
        super().__init__()
        self.features = make_vgg16_features(channels)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

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
            f"Prova: bash TEST_NUOVI_REPRO_VGG16/scripts/download_datasets_curl.sh {name}"
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


def aggregate_krum(updates, byzantine_f):
    """Krum (Blanchard et al., NeurIPS 2017): score(i) = sum of squared distances
    to its n-f-2 nearest neighbours; pick the update with the lowest score."""
    vectors = torch.stack([flatten_state(u) for u in updates])
    n = len(updates)
    neighbor_count = max(1, n - byzantine_f - 2)
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


def aggregate_clipped_clustering(updates, norm_history):
    """ClippedClustering (arXiv:2302.07173): clip update norms to the median of
    *historical* round norms, then bipartition clients by cosine-similarity
    agglomerative clustering (Sattler et al., ICASSP 2020) and keep the majority
    (larger) cluster as benign."""
    vectors = torch.stack([flatten_state(u) for u in updates])
    norms = torch.norm(vectors, dim=1).clamp_min(1e-12)
    clip = float(np.median(norm_history)) if norm_history else torch.median(norms).item()
    clipped = vectors * torch.clamp(clip / norms, max=1.0).unsqueeze(1)
    new_norm_history = norm_history + [torch.median(norms).item()]

    if len(updates) < 3 or AgglomerativeClustering is None:
        return vector_to_update(clipped.mean(dim=0), updates[0]), new_norm_history

    normed = clipped / torch.norm(clipped, dim=1, keepdim=True).clamp_min(1e-12)
    cosine_sim = (normed @ normed.T).clamp(-1.0, 1.0)
    cosine_dist = (1.0 - cosine_sim).cpu().numpy()
    np.fill_diagonal(cosine_dist, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=2, metric='precomputed', linkage='average'
    ).fit_predict(cosine_dist)
    counts = np.bincount(labels)
    chosen = int(np.argmax(counts))
    return vector_to_update(clipped[labels == chosen].mean(dim=0), updates[0]), new_norm_history


def fltrust_normalize_and_score(updates, server_update):
    """Trust bootstrapping shared by FLTrust (Cao et al., NDSS 2021) and RFLPA
    (arXiv:2405.15182): TS_i = ReLU(cos(g_i, g_0)), g_i_bar = (||g_0||/||g_i||) * g_i."""
    server_vec = flatten_state(server_update)
    server_norm = torch.norm(server_vec).clamp_min(1e-12)
    trust_scores = []
    normalized_updates = []
    for update in updates:
        vec = flatten_state(update)
        vec_norm = torch.norm(vec).clamp_min(1e-12)
        cos = F.cosine_similarity(vec, server_vec, dim=0).item()
        trust_scores.append(max(0.0, cos))
        scale = (server_norm / vec_norm).item()
        normalized_updates.append({k: v.float() * scale for k, v in update.items()})
    return normalized_updates, trust_scores


def aggregate_fltrust(updates, server_update):
    normalized_updates, trust_scores = fltrust_normalize_and_score(updates, server_update)
    total = sum(trust_scores)
    if total <= 1e-12:
        weights = [1.0 / len(updates)] * len(updates)
    else:
        weights = [w / total for w in trust_scores]
    return average_updates(normalized_updates, weights)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TD3Policy:
    """Minimal TD3 actor-critic (Fujimoto et al., 2018) learning the AdaAggRL
    (arXiv:2406.14217) aggregation-weighting policy online, one transition per
    federated round. Action = (a in simplex over the 4 similarity metrics,
    b in [0,1] threshold fraction)."""

    def __init__(self, state_dim, device, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9, tau=0.01,
                 policy_noise=0.1, noise_clip=0.2, policy_delay=2, buffer_size=2000):
        self.device = device
        self.action_dim = 5
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip, self.policy_delay = policy_noise, noise_clip, policy_delay
        self.actor = MLP(state_dim, self.action_dim).to(device)
        self.actor_target = MLP(state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        in_dim = state_dim + self.action_dim
        self.critic1 = MLP(in_dim, 1).to(device)
        self.critic2 = MLP(in_dim, 1).to(device)
        self.critic1_target = MLP(in_dim, 1).to(device)
        self.critic2_target = MLP(in_dim, 1).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )
        self.buffer = []
        self.buffer_size = buffer_size
        self.updates_done = 0

    @staticmethod
    def _raw_to_action(raw):
        a = torch.softmax(raw[..., :4], dim=-1)
        b = torch.sigmoid(raw[..., 4:5])
        return torch.cat([a, b], dim=-1)

    def act(self, state_vec, explore=True):
        with torch.no_grad():
            raw = self.actor(state_vec.to(self.device).unsqueeze(0)).squeeze(0)
            if explore:
                raw = raw + torch.randn_like(raw) * 0.1
            action = self._raw_to_action(raw)
        return action.cpu()

    def remember(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self, batch_size=16):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([b[0] for b in batch]).to(self.device)
        actions = torch.stack([b[1] for b in batch]).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack([b[3] for b in batch]).to(self.device)

        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            noise = (torch.randn_like(next_raw) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self._raw_to_action(next_raw + noise)
            target_q1 = self.critic1_target(torch.cat([next_states, next_action], dim=1))
            target_q2 = self.critic2_target(torch.cat([next_states, next_action], dim=1))
            target_q = rewards + self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        self.updates_done += 1
        if self.updates_done % self.policy_delay == 0:
            action = self._raw_to_action(self.actor(states))
            actor_loss = -self.critic1(torch.cat([states, action], dim=1)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            for target, source in (
                (self.actor_target, self.actor),
                (self.critic1_target, self.critic1),
                (self.critic2_target, self.critic2),
            ):
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * sp.data)


def mmd_rbf(x, y, sigma=1.0):
    if x is None or y is None or x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    kxx = torch.exp(-torch.cdist(x, x) ** 2 / (2 * sigma ** 2)).mean()
    kyy = torch.exp(-torch.cdist(y, y) ** 2 / (2 * sigma ** 2)).mean()
    kxy = torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2)).mean()
    return (kxx + kyy - 2 * kxy).clamp_min(0).sqrt().item()


def reconstruct_client_distribution(global_state, update, model_factory, channels, image_size, device,
                                     num_images=16, max_iters=30, lr=0.1):
    """Gradient-inversion reconstruction of the client's local data (AdaAggRL,
    Sec. 3.1). NOTE: the paper inverts a single-step gradient; our clients run
    multiple local epochs, so the cumulative update is used in its place as
    the closest available proxy for the target gradient -- an approximation
    forced by the multi-step local training used throughout this repro."""
    model = model_factory().to(device)
    model.load_state_dict(global_state, strict=True)
    model.eval()
    target_grad = {k: -v.to(device) for k, v in update.items()}

    num_classes = model.classifier[-1].out_features
    dummy_x = torch.randn(num_images, channels, image_size, image_size, device=device, requires_grad=True)
    dummy_y_logits = torch.randn(num_images, num_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([dummy_x, dummy_y_logits], lr=lr)

    param_items = [(name, p) for name, p in model.named_parameters() if name in target_grad]
    param_names = [name for name, _ in param_items]
    params = [p for _, p in param_items]

    grad_diff = torch.tensor(0.0, device=device)
    for _ in range(max_iters):
        opt.zero_grad(set_to_none=True)
        dummy_y = F.softmax(dummy_y_logits, dim=1)
        out = model(dummy_x)
        loss = -(dummy_y * F.log_softmax(out, dim=1)).sum(dim=1).mean()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_diff = sum(F.mse_loss(g, target_grad[name]) for name, g in zip(param_names, grads))
        grad_diff.backward()
        opt.step()

    reconstruction_similarity = 1.0 / (1.0 + float(grad_diff.detach().item()))
    return dummy_x.detach(), reconstruction_similarity


def extract_features(model, images):
    with torch.no_grad():
        return model.features(images).flatten(1)


def adaaggrl_similarities(current_embeddings, history_embeddings, global_embeddings):
    def to_score(d):
        return 2.0 * math.cos(math.tanh(d / 2.0)) - 1.0
    d_cl = mmd_rbf(current_embeddings, history_embeddings)
    d_cg = mmd_rbf(current_embeddings, global_embeddings)
    d_lg = mmd_rbf(history_embeddings, global_embeddings)
    return to_score(d_cl), to_score(d_cg), to_score(d_lg)


def aggregate_adaaggrl(global_state, updates, model_factory, server_loader, channels, image_size, device, state,
                        lam=1.5):
    """AdaAggRL (arXiv:2406.14217, AAAI 2025): reconstruct each client's data
    distribution via gradient inversion, score clients with MMD-based
    similarity to their own history and to the round's global distribution,
    let a TD3 policy set metric weights + a filtering threshold, and apply an
    exponential penalty lambda^h to clients repeatedly flagged as suspicious."""
    n = len(updates)
    if state.get('policy') is None:
        state['policy'] = TD3Policy(state_dim=8, device=device)
        state['history_embeddings'] = [None] * n
        state['h'] = [0] * n
        state['prev_loss'] = None
        state['prev_state_vec'] = None
        state['prev_action'] = None

    policy = state['policy']
    feature_model = model_factory().to(device)
    feature_model.load_state_dict(global_state, strict=True)
    feature_model.eval()

    recon_sims, embeddings = [], []
    for update in updates:
        images, sim = reconstruct_client_distribution(global_state, update, model_factory, channels, image_size, device)
        recon_sims.append(sim)
        embeddings.append(extract_features(feature_model, images))
    global_embeddings = torch.cat(embeddings, dim=0)

    raw_metrics = []
    for idx in range(n):
        hist = state['history_embeddings'][idx]
        s_cl, s_cg, s_lg = adaaggrl_similarities(embeddings[idx], hist, global_embeddings)
        raw_metrics.append((recon_sims[idx], s_cl, s_cg, s_lg))

    metrics_arr = np.array(raw_metrics, dtype=float)
    mn, mx = metrics_arr.min(axis=0), metrics_arr.max(axis=0)
    normalized = np.clip((metrics_arr - mn) / np.maximum(mx - mn, 1e-6), 1e-6, 1.0)

    state_vec = torch.tensor(
        np.concatenate([normalized.mean(axis=0), normalized.std(axis=0)]), dtype=torch.float32
    )
    action = policy.act(state_vec, explore=True)
    a_weights = action[:4].numpy()
    b_threshold = float(action[4].item())

    w_hat = normalized @ a_weights
    w_tilde = w_hat / (w_hat.max() if w_hat.max() > 0 else 1.0)
    delta = w_tilde.max() * b_threshold
    filtered = np.where(w_tilde > delta, w_tilde, 0.0)

    for idx in range(n):
        if filtered[idx] <= 0:
            state['h'][idx] += 1
        else:
            state['h'][idx] = max(0, state['h'][idx] - 1)

    penalized = np.array([
        filtered[idx] / (lam ** state['h'][idx]) if filtered[idx] > 0 else 0.0
        for idx in range(n)
    ])
    if penalized.sum() <= 1e-12:
        weights = [1.0 / n] * n
    else:
        weights = (penalized / penalized.sum()).tolist()

    aggregated = average_updates(updates, weights)

    candidate_model = model_factory().to(device)
    candidate_model.load_state_dict(add_update(global_state, aggregated), strict=True)
    new_loss, _ = evaluate(candidate_model, server_loader, device)
    if state['prev_loss'] is not None:
        reward = state['prev_loss'] - new_loss
        policy.remember(state['prev_state_vec'], state['prev_action'], reward, state_vec)
        policy.train_step()
    state['prev_loss'] = new_loss
    state['prev_state_vec'] = state_vec
    state['prev_action'] = action
    state['history_embeddings'] = embeddings

    return aggregated, {
        'selected_clients': [i for i in range(n) if filtered[i] > 0],
        'penalty_counts': list(state['h']),
        'action_weights': a_weights.tolist(),
        'threshold_b': b_threshold,
    }


def aggregate_fedlad(updates, tol=1e-6):
    """FedLAD (arXiv:2508.02136): stack client updates as columns of a
    parameters x clients matrix, run Gauss-Jordan elimination with partial
    pivoting to compute its RREF, and keep only the clients whose columns
    become pivot columns (i.e. are linearly independent of previously kept
    columns) -- this discards redundant/duplicated (Sybil) updates. Selected
    clients are then merged with plain unweighted FedAvg."""
    vectors = torch.stack([flatten_state(update) for update in updates]).double()
    norms = torch.norm(vectors, dim=1).clamp_min(1e-12)
    matrix = vectors.T.clone()  # rows = parameters, columns = clients
    num_rows, num_cols = matrix.shape

    pivot_row = 0
    selected_indices = []
    for col in range(num_cols):
        if pivot_row >= num_rows:
            break
        column_tail = matrix[pivot_row:, col]
        rel_idx = int(torch.argmax(torch.abs(column_tail)))
        pivot_value = column_tail[rel_idx]
        eps = tol * norms[col]
        if pivot_value.abs().item() < eps.item():
            continue  # column is (numerically) a linear combination of already-selected columns

        abs_row = pivot_row + rel_idx
        if abs_row != pivot_row:
            matrix[[pivot_row, abs_row]] = matrix[[abs_row, pivot_row]]

        matrix[pivot_row] = matrix[pivot_row] / matrix[pivot_row, col].clone()
        factors = matrix[:, col].clone()
        factors[pivot_row] = 0.0
        matrix -= torch.outer(factors, matrix[pivot_row])

        selected_indices.append(col)
        pivot_row += 1

    fallback_used = False
    if not selected_indices:
        selected_indices = list(range(num_cols))
        fallback_used = True

    selected_updates = [updates[idx] for idx in selected_indices]
    return average_updates(selected_updates), {
        'selected_clients': selected_indices,
        'selected_count': len(selected_indices),
        'fallback_fedavg': fallback_used,
    }


def client_shap_attribution(candidate_weights, model_factory, background_x, probe_x, device):
    """Mean |SHAP value| per output class for one client's candidate global
    model, using a gradient-based explainer (DeepLIFT-style expected
    gradients) over the server's reference data."""
    model = model_factory().to(device)
    model.load_state_dict(candidate_weights, strict=True)
    model.eval()
    explainer = shap.GradientExplainer(model, background_x)
    shap_values = explainer.shap_values(probe_x)
    if isinstance(shap_values, list):
        per_class = np.stack(
            [np.abs(sv).mean(axis=tuple(range(1, sv.ndim))) for sv in shap_values], axis=1
        )
    else:
        arr = np.asarray(shap_values)
        per_class = np.abs(arr).reshape(arr.shape[0], arr.shape[1], -1).mean(axis=2)
    return per_class.mean(axis=0)


def aggregate_sherpa(global_state, updates, model_factory, server_loader, device):
    """SHERPA (IEEE S&P 2024): identify poisoners via SHAP feature-attribution
    clustering rather than raw loss statistics -- for each client's candidate
    global model, compute per-class mean |SHAP value| attributions, then
    bipartition clients on those attribution vectors and discard the minority
    (poisoner) cluster, averaging the majority with equal weights."""
    if shap is None:
        raise RuntimeError('Il metodo "sherpa" richiede il pacchetto "shap" (pip install shap).')
    if KMeans is None:
        raise RuntimeError('Il metodo "sherpa" richiede scikit-learn (pip install scikit-learn).')

    background_x, _ = next(iter(server_loader))
    background_x = background_x.to(device)
    probe_x, _ = next(iter(server_loader))
    probe_x = probe_x.to(device)

    attributions = [
        client_shap_attribution(add_update(global_state, update), model_factory, background_x, probe_x, device)
        for update in updates
    ]
    attribution_matrix = np.stack(attributions)

    if len(updates) < 3:
        return average_updates(updates), {'fallback_fedavg': True}

    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(attribution_matrix)
    counts = np.bincount(labels)
    chosen = int(np.argmax(counts))
    selected_indices = [i for i, label in enumerate(labels) if label == chosen]
    selected_updates = [updates[i] for i in selected_indices]

    return average_updates(selected_updates), {
        'selected_clients': selected_indices,
        'attribution_shape': list(attribution_matrix.shape),
        'fallback_fedavg': False,
    }


def aggregate_fedgreed(global_state, updates, model_factory, server_loader, device):
    """FedGreed (arXiv:2508.18060): rank clients once by individual loss on the
    server's reference dataset, then test fixed-order prefixes of the sorted
    list, stopping as soon as extending the prefix stops improving the loss."""
    individual_losses = []
    for update in updates:
        model = model_factory().to(device)
        model.load_state_dict(add_update(global_state, update), strict=True)
        loss, _ = evaluate(model, server_loader, device)
        individual_losses.append(loss)
    order = sorted(range(len(updates)), key=lambda i: individual_losses[i])

    selected = []
    best_loss = float('inf')
    for idx in order:
        candidate = selected + [idx]
        update = average_updates([updates[i] for i in candidate])
        model = model_factory().to(device)
        model.load_state_dict(add_update(global_state, update), strict=True)
        loss, _ = evaluate(model, server_loader, device)
        if not selected or loss <= best_loss:
            selected = candidate
            best_loss = loss
        else:
            break
    return average_updates([updates[i] for i in selected]), selected


def aggregate(method, global_state, local_states, server_state, model_factory, server_loader, device, state,
              byzantine_f=0, channels=None, image_size=None):
    updates = [state_update(global_state, s) for s in local_states]
    server_update = state_update(global_state, server_state) if server_state is not None else None
    info = {}
    if method in {'fedavg', 'fedsgd'}:
        return average_updates(updates), state, info
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
        return average_updates(selected), state, {'confidence_scores': scores, 'selected_clients': len(selected), 'fallback_fedavg': fallback_used}
    if method == 'krum':
        return aggregate_krum(updates, byzantine_f), state, info
    if method == 'trimmed_mean':
        return aggregate_trimmed_mean(updates), state, info
    if method == 'fltrust':
        return aggregate_fltrust(updates, server_update), state, {'server_guided': True}
    if method == 'clipped_clustering':
        norm_history = state.get('norm_history', [])
        update, norm_history = aggregate_clipped_clustering(updates, norm_history)
        state['norm_history'] = norm_history
        return update, state, info
    if method == 'rflpa':
        return aggregate_fltrust(updates, server_update), state, {'server_guided': True}
    if method == 'adaaggrl':
        adaaggrl_state = state.get('adaaggrl', {})
        update, adaaggrl_info = aggregate_adaaggrl(
            global_state, updates, model_factory, server_loader, channels, image_size, device, adaaggrl_state
        )
        state['adaaggrl'] = adaaggrl_state
        return update, state, adaaggrl_info
    if method == 'fedgreed':
        update, selected = aggregate_fedgreed(global_state, updates, model_factory, server_loader, device)
        return update, state, {'selected_clients': selected}
    if method == 'sherpa':
        update, sherpa_info = aggregate_sherpa(global_state, updates, model_factory, server_loader, device)
        return update, state, sherpa_info
    if method == 'fedlad':
        update, fedlad_info = aggregate_fedlad(updates)
        return update, state, fedlad_info
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
    parser.add_argument('--krum-f', type=int, default=-1, help='Numero di client Byzantine assunto da Krum (n-f-2 vicini). -1 = auto dal numero di malicious-clients configurati.')
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'args.json').write_text(json.dumps(vars(args), indent=2) + '\n', encoding='utf-8')

    train_ds, test_ds, channels, num_classes, image_size = load_datasets(args.dataset, Path(args.data_root), download=not args.no_download)
    client_loaders, server_loader, test_loader, active_malicious = make_loaders(train_ds, test_ds, args, num_classes)
    device = torch.device(args.device)

    def model_factory():
        return VGG16(channels, num_classes, image_size)

    global_model = model_factory().to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    aggregator_state = {}
    metrics = []

    if args.krum_f < 0:
        malicious_ids = {int(x) for x in args.malicious_clients.split(',') if x.strip()}
        byzantine_f = min(len(malicious_ids), max(0, (args.num_clients - 3) // 2))
    else:
        byzantine_f = args.krum_f

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

        update, aggregator_state, info = aggregate(
            args.method, global_state, local_states, server_state, model_factory, server_loader, device,
            aggregator_state, byzantine_f=byzantine_f, channels=channels, image_size=image_size,
        )
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
