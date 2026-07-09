#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts' / 'robust_fl_experiment.py'


def load_matrix(config_path: Path) -> dict:
    with config_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def experiment_id(dataset: str, alpha: float, attack: str, method: str, seed: int) -> str:
    alpha_text = str(alpha).replace('.', 'p')
    return f'{dataset}__alpha_{alpha_text}__{attack}__{method}__seed_{seed}'


def write_launcher(run_dir: Path, config: dict) -> None:
    defaults = config['defaults']
    attack = config['attack']
    malicious_clients = ','.join(str(client) for client in attack['malicious_clients'])
    launcher = run_dir / 'run.sh'
    launcher.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'cd "$(dirname "$0")"\n'
        f'python "{RUNNER}" \\\n'
        f'  --dataset "{config["dataset"]["name"]}" \\\n'
        f'  --method "{config["method"]["name"]}" \\\n'
        f'  --alpha "{config["alpha"]}" \\\n'
        f'  --attack "{attack["name"]}" \\\n'
        f'  --num-rounds "{defaults["num_rounds"]}" \\\n'
        f'  --num-clients "{defaults["num_clients"]}" \\\n'
        f'  --server-fraction "{defaults["server_data_fraction"]}" \\\n'
        f'  --batch-size "{defaults["batch_size"]}" \\\n'
        f'  --local-epochs "{defaults["local_epochs"]}" \\\n'
        f'  --seed "{defaults["seed"]}" \\\n'
        f'  --label-swap-fraction "{attack["label_swap_fraction"]}" \\\n'
        f'  --malicious-probability "{attack["malicious_activation_probability"]}" \\\n'
        f'  --malicious-clients "{malicious_clients}" \\\n'
        '  --output-dir "results"\n',
        encoding='utf-8',
    )
    launcher.chmod(0o755)


def materialize(config_path: Path, dry_run: bool) -> list[dict]:
    matrix = load_matrix(config_path)
    output_root = ROOT / matrix['output_root']
    defaults = matrix['defaults']
    rows = []

    for dataset in matrix['datasets']:
        for alpha in matrix['non_iid_alphas']:
            for attack in matrix['attack_intensities']:
                for method in matrix['methods']:
                    seed = defaults['seed']
                    exp_id = experiment_id(dataset['name'], alpha, attack['name'], method['name'], seed)
                    run_dir = output_root / dataset['name'] / f"alpha_{str(alpha).replace('.', 'p')}" / attack['name'] / method['name'] / f'seed_{seed}'
                    config = {
                        'experiment_id': exp_id,
                        'dataset': dataset,
                        'alpha': alpha,
                        'attack': attack,
                        'method': method,
                        'defaults': defaults,
                        'runner': str(RUNNER),
                        'status': 'ready',
                    }
                    if not dry_run:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        (run_dir / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                        write_launcher(run_dir, config)
                    rows.append({
                        'experiment_id': exp_id,
                        'dataset': dataset['name'],
                        'alpha': alpha,
                        'attack': attack['name'],
                        'method': method['name'],
                        'status': 'ready',
                        'run_dir': str(run_dir),
                        'runner': str(RUNNER),
                        'reason': '',
                    })
    return rows


def write_manifest(rows: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Materializza la matrice esperimenti in run riproducibili.')
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'experiment_matrix.json'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    matrix = load_matrix(config_path)
    rows = materialize(config_path, args.dry_run)
    manifest_path = ROOT / matrix['manifest_path']
    if not args.dry_run:
        write_manifest(rows, manifest_path)

    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    print(json.dumps({'total': len(rows), 'counts': counts, 'manifest': str(manifest_path)}, indent=2))


if __name__ == '__main__':
    main()
