#!/usr/bin/env python3
import argparse
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
LEGACY_ROOT = PROJECT_ROOT / "TEST_NUOVI"


LEGACY_PATTERNS = [
    "distribuzione *",
    "resultsagenti",
    "datatest_test",
    ".ipynb_checkpoints",
    "dist_0.5_nodataset.zip"
]


def collect_targets() -> list[Path]:
    targets = []
    for pattern in LEGACY_PATTERNS:
        targets.extend(sorted(LEGACY_ROOT.glob(pattern)))
    return [target for target in targets if target.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Archivia o cancella i risultati legacy in TEST_NUOVI.")
    parser.add_argument("--archive", action="store_true", help="Sposta i risultati in TEST_NUOVI_REPRO/legacy_archive/<timestamp>.")
    parser.add_argument("--delete", action="store_true", help="Cancella definitivamente i risultati legacy.")
    parser.add_argument("--yes", action="store_true", help="Richiesto insieme a --delete.")
    args = parser.parse_args()

    targets = collect_targets()
    if not args.archive and not args.delete:
        print("Dry run. Target che verrebbero archiviati/cancellati:")
        for target in targets:
            print(target)
        print("\nUsa --archive per spostare in archivio, oppure --delete --yes per cancellare.")
        return

    if args.archive and args.delete:
        raise SystemExit("Scegli solo una modalita': --archive oppure --delete.")

    if args.delete and not args.yes:
        raise SystemExit("Per cancellare devi passare anche --yes.")

    if args.archive:
        archive_dir = ROOT / "legacy_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir.mkdir(parents=True, exist_ok=True)
        for target in targets:
            shutil.move(str(target), str(archive_dir / target.name))
        print(f"Archiviati {len(targets)} target in {archive_dir}")
        return

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    print(f"Cancellati {len(targets)} target legacy da {LEGACY_ROOT}")


if __name__ == "__main__":
    main()
