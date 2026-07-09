#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
JOBS="${JOBS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh [dataset ...]

Datasets:
  cifar10 cifar100 svhn fashionmnist all

Env:
  DATA_ROOT=/path/to/data   default: TEST_NUOVI_REPRO/data
  JOBS=4                    parallel downloads, default: 1

Examples:
  bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh all
  JOBS=4 bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh cifar10 cifar100 svhn fashionmnist
EOF
}

curl_get() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "[skip] $out already exists"
    return 0
  fi
  echo "[curl] $url"
  curl -L --fail --retry 10 --retry-delay 5 --connect-timeout 30 -C - -o "$out" "$url"
}

curl_get_any() {
  local out="$1"
  shift
  local url
  for url in "$@"; do
    if curl_get "$url" "$out"; then
      return 0
    fi
    echo "[warn] failed: $url"
  done
  return 1
}

extract_tgz() {
  local archive="$1"
  local dest="$2"
  local marker="$3"
  if [[ -e "$dest/$marker" ]]; then
    echo "[skip] $dest/$marker already extracted"
    return 0
  fi
  echo "[tar] $archive -> $dest"
  tar -xzf "$archive" -C "$dest"
}

download_cifar10() {
  local root="$DATA_ROOT/cifar10"
  local archive="$root/cifar-10-python.tar.gz"
  local fastai_archive="$DATA_ROOT/cifar10_fastai.tgz"
  mkdir -p "$root"
  echo "[fallback] Toronto unavailable; using FastAI CIFAR-10 ImageFolder mirror"
  curl_get "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz" "$fastai_archive"
  extract_tgz "$fastai_archive" "$DATA_ROOT" "cifar10/train"
}

download_cifar100() {
  local root="$DATA_ROOT/cifar100"
  local archive="$root/cifar-100-python.tar.gz"
  local fastai_archive="$DATA_ROOT/cifar100_fastai.tgz"
  mkdir -p "$root"
  echo "[fallback] Toronto unavailable; using FastAI CIFAR-100 ImageFolder mirror"
  curl_get "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz" "$fastai_archive"
  extract_tgz "$fastai_archive" "$DATA_ROOT" "cifar100/train"
}

download_svhn() {
  local root="$DATA_ROOT/svhn"
  mkdir -p "$root"
  curl_get "http://ufldl.stanford.edu/housenumbers/train_32x32.mat" "$root/train_32x32.mat"
  curl_get "http://ufldl.stanford.edu/housenumbers/test_32x32.mat" "$root/test_32x32.mat"
}

download_fashionmnist() {
  local root="$DATA_ROOT/fashionmnist/FashionMNIST/raw"
  mkdir -p "$root"
  local s3_base="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
  local github_base="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
  curl_get_any "$root/train-images-idx3-ubyte.gz" "$s3_base/train-images-idx3-ubyte.gz" "$github_base/train-images-idx3-ubyte.gz"
  curl_get_any "$root/train-labels-idx1-ubyte.gz" "$s3_base/train-labels-idx1-ubyte.gz" "$github_base/train-labels-idx1-ubyte.gz"
  curl_get_any "$root/t10k-images-idx3-ubyte.gz" "$s3_base/t10k-images-idx3-ubyte.gz" "$github_base/t10k-images-idx3-ubyte.gz"
  curl_get_any "$root/t10k-labels-idx1-ubyte.gz" "$s3_base/t10k-labels-idx1-ubyte.gz" "$github_base/t10k-labels-idx1-ubyte.gz"
}

run_one() {
  case "$1" in
    cifar10) download_cifar10 ;;
    cifar100) download_cifar100 ;;
    svhn) download_svhn ;;
    fashionmnist) download_fashionmnist ;;
    all)
      download_cifar10
      download_cifar100
      download_svhn
      download_fashionmnist
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown dataset: $1" >&2; usage; exit 2 ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 0
fi

mkdir -p "$DATA_ROOT"
echo "DATA_ROOT=$DATA_ROOT"

if [[ "$JOBS" -gt 1 && "$#" -gt 1 ]]; then
  printf '%s\n' "$@" | xargs -n1 -P "$JOBS" bash -c '"$0" "$1"' "$0"
else
  for dataset in "$@"; do
    run_one "$dataset"
  done
fi

echo "Done. The runner will read datasets from: $DATA_ROOT"
