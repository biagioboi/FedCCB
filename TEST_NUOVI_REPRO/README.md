# Reproducible experiments for the journal submission

This folder separates the new experimental campaign from the historical results in `TEST_NUOVI`.

## Goal

- Generalize beyond CIFAR-10: `cifar10`, `cifar100`, `svhn`, `fashionmnist`.
- Compare the proposed method against classic and robust baselines: `fedavg`, `fedsgd`, `krum`, `trimmed_mean`, `fltrust`.
- Track the recent methods still to be implemented: `clipped_clustering`, `rflpa`, `adaaggrl`, `fedgreed`.
- Make every run identifiable by dataset, non-IID alpha, attack intensity, method and seed.

## Structure

- `configs/experiment_matrix.json`: the official experiment matrix.
- `scripts/materialize_runs.py`: generates `runs/...`, `config.json`, `run.sh` and `run_manifest.csv`.
- `scripts/clean_legacy_results.py`: archives or deletes the old results.
- `runs/`: newly generated results area, not hand-edited.

## Usage

Generate the matrix:

```bash
python TEST_NUOVI_REPRO/scripts/materialize_runs.py
```

Check what is already runnable:

```bash
python - <<'PY'
import csv
from collections import Counter
with open('TEST_NUOVI_REPRO/run_manifest.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(Counter(row['status'] for row in rows))
for row in rows:
    if row['status'] == 'ready':
        print(row['run_dir'])
PY
```

Run a ready experiment:

```bash
bash TEST_NUOVI_REPRO/runs/cifar10/alpha_0p9/medio/fltrust/seed_42/run.sh
```

Before deleting old results, do a dry run:

```bash
python TEST_NUOVI_REPRO/scripts/clean_legacy_results.py
```

Recommended: archive before deleting:

```bash
python TEST_NUOVI_REPRO/scripts/clean_legacy_results.py --archive
```

Permanent deletion:

```bash
python TEST_NUOVI_REPRO/scripts/clean_legacy_results.py --delete --yes
```

## Model

All runs use the same classifier, `SmallCNN` (defined in `scripts/robust_fl_experiment.py`,
not VGG16): 3 conv blocks (32/64/128 channels, BatchNorm + ReLU + MaxPool, final
`AdaptiveAvgPool2d`) followed by a fully-connected classifier (128 -> 128 -> num_classes,
with 0.2 dropout). It is the same architecture for both client and server across all
datasets (`cifar10`, `cifar100`, `svhn`, `fashionmnist`); only the input channels and the
number of classes change. The only other network in the file is the `MLP` used internally
as the TD3 policy/critic by the `adaaggrl` method, not as a classifier.

## Methods implemented in the single runner

The runner `scripts/robust_fl_experiment.py` executes all new runs for 50 rounds and saves `results/metrics.csv`, `results/args.json` and `results/last_checkpoint.pt` inside every experiment folder.

Available methods:

- `proposed_confidence`: generalized version of the confidence-based method, with scoring on a clean root dataset and selection via clustering of the scores.
- `fedavg`, `fedsgd`: classic federated baselines.
- `krum`: Blanchard et al., NeurIPS 2017. Picks the update with the minimum distance to its `n-f-2` closest neighbors (`f` = assumed number of Byzantine clients, `--krum-f`, by default estimated from the number of configured `--malicious-clients`).
- `trimmed_mean`: Yin et al., ICML 2018. Coordinate-wise trimmed mean.
- `fltrust`: Cao et al., NDSS 2021. Trust score `ReLU(cos(g_i, g_0))` **after** normalizing every update to the norm of the server update (`g_i_bar = (||g_0||/||g_i||) * g_i`), then a weighted average of the `g_i_bar`.
- `rflpa`: arXiv:2405.15182 (NeurIPS 2024). Same trust-bootstrapping formula as FLTrust (with secure aggregation in the original paper, omitted here as it is out of scope for the simulation).
- `clipped_clustering`: arXiv:2302.07173. Clips the norm to the **historical median value** (not the median of the current round only), then bipartitions clients via agglomerative clustering on cosine similarity (Sattler et al., ICASSP 2020), keeping the majority cluster.
- `adaaggrl`: arXiv:2406.14217 (AAAI 2025). Reconstructs each client's data distribution via gradient inversion, MMD similarity (client history vs. round-level global distribution), a TD3 policy that learns the weights of the 4 metrics plus the filtering threshold, and an exponential `lambda^h` penalty for repeatedly suspicious clients. NOTE: the paper's gradient inversion assumes a single SGD step; here clients run multiple local epochs, so the cumulative update is used as a proxy for the target gradient (a necessary approximation). The "pre-trained" feature extractor is the convolutional backbone of the current global model (no external pre-trained network is available in this pipeline).
- `fedgreed`: arXiv:2508.18060. Sorts clients once by individual loss on the root dataset, then tests prefixes in this fixed order, stopping at the first prefix that does not improve the loss.
- `sherpa`: IEEE S&P 2024. Identifies poisoners via clustering of SHAP attributions (not via loss statistics): for each candidate model it computes the average per-class |SHAP value| attribution with a gradient explainer, bipartitions clients on these vectors and discards the minority cluster.
- `fedlad`: arXiv:2508.02136. Gauss-Jordan elimination with partial pivoting (RREF) on the parameters x clients matrix to identify the linearly independent columns (clients), then unweighted FedAvg over only the selected clients.

All datasets in the matrix are now runnable: `cifar10`, `cifar100`, `svhn`, `fashionmnist`.

## Client parallelism

Runs train clients in parallel via `--client-parallelism 0`, where `0` means use all clients in the run. To go back to serial behavior and reduce peak GPU memory, pass `--client-parallelism 1`.

## Operational status

- `run_manifest.csv` contains 360 `ready` runs.
- Every `run.sh` uses `--num-rounds 50`.
- The old results in `TEST_NUOVI` have been deleted upon request.

## Downloading with curl

If `torchvision` downloads slowly, you can pre-download the public archives with `curl` and leave them in the cache used by the runs:

```bash
bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh all
```

Parallel download:

```bash
JOBS=4 bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh cifar10 cifar100 svhn fashionmnist
```

Custom cache, if you want to download to an external disk and then move/symlink it:

```bash
DATA_ROOT=/fast/path/data bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh all
```

The datasets are public: no token needed. If you use a private mirror, you can download elsewhere and then copy into the same structure as `TEST_NUOVI_REPRO/data`.
