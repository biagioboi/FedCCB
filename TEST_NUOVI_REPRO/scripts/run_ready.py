#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "run_manifest.csv"


def load_ready_rows(manifest_path: Path, dataset: str | None, alpha: str | None, attack: str | None, method: str | None) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ready"]
    if dataset:
        rows = [row for row in rows if row["dataset"] == dataset]
    if alpha:
        rows = [row for row in rows if row["alpha"] == alpha]
    if attack:
        rows = [row for row in rows if row["attack"] == attack]
    if method:
        rows = [row for row in rows if row["method"] == method]
    return rows


def run_one(row: dict, slot: int, gpus: list[str] | None) -> tuple[dict, int, Path]:
    run_dir = Path(row["run_dir"])
    run_script = run_dir / "run.sh"
    log_path = run_dir / "runner.log"
    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus[slot % len(gpus)]
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"experiment_id={row['experiment_id']}\n")
        log_file.write(f"run_script={run_script}\n")
        if gpus:
            log_file.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        log_file.flush()
        completed = subprocess.run(
            ["bash", str(run_script)],
            cwd=str(run_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return row, completed.returncode, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue le run ready elencate in TEST_NUOVI_REPRO/run_manifest.csv.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--alpha", default=None, help="Esempio: 0.1, 0.5, 0.9")
    parser.add_argument("--attack", default=None, choices=[None, "basso", "medio", "alto"])
    parser.add_argument("--method", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Numero di run da eseguire in parallelo.")
    parser.add_argument("--gpus", default=None, help="Lista GPU separate da virgola, esempio: 0,1. Assegna CUDA_VISIBLE_DEVICES per slot.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continua anche se una run fallisce.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_ready_rows(MANIFEST, args.dataset, args.alpha, args.attack, args.method)
    if not rows:
        raise SystemExit("Nessuna run ready trovata con questi filtri.")

    jobs = max(1, args.jobs)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()] if args.gpus else None

    print(f"Run ready selezionate: {len(rows)}")
    print(f"Jobs paralleli: {jobs}")
    if gpus:
        print(f"GPU slots: {gpus}")

    for row in rows:
        print(Path(row["run_dir"]) / "run.sh")
    if args.dry_run:
        return

    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_one, row, index, gpus): row
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            row, returncode, log_path = future.result()
            status = "OK" if returncode == 0 else f"FAILED({returncode})"
            print(f"[{status}] {row['experiment_id']} log={log_path}")
            if returncode != 0:
                failures.append((row, returncode, log_path))
                if not args.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    break

    if failures:
        print("\nRun fallite:")
        for row, returncode, log_path in failures:
            print(f"- {row['experiment_id']} returncode={returncode} log={log_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
