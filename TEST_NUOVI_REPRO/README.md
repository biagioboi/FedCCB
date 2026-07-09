# Esperimenti riproducibili per rivista

Questa cartella separa la nuova campagna sperimentale dai risultati storici in `TEST_NUOVI`.

## Obiettivo

- Generalizzare oltre CIFAR-10: `cifar10`, `cifar100`, `svhn`, `fashionmnist`.
- Confrontare il metodo proposto con baseline classiche e robuste: `fedavg`, `fedsgd`, `krum`, `trimmed_mean`, `fltrust`.
- Tenere traccia dei metodi recenti da implementare: `clipped_clustering`, `rflpa`, `adaaggrl`, `fedgreed`.
- Rendere ogni run identificabile da dataset, alpha non-IID, intensita' di attacco, metodo e seed.

## Struttura

- `configs/experiment_matrix.json`: matrice ufficiale degli esperimenti.
- `scripts/materialize_runs.py`: genera `runs/...`, `config.json`, `run.sh` e `run_manifest.csv`.
- `scripts/clean_legacy_results.py`: archivia o cancella i risultati vecchi.
- `runs/`: nuova area risultati generata, non compilata a mano.

## Uso

Genera la matrice:

```bash
python TEST_NUOVI_REPRO/scripts/materialize_runs.py
```

Controlla cosa e' gia' eseguibile:

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

Esegui una run pronta:

```bash
bash TEST_NUOVI_REPRO/runs/cifar10/alpha_0p9/medio/fltrust/seed_42/run.sh
```

Prima di cancellare i vecchi risultati, fai un dry run:

```bash
python TEST_NUOVI_REPRO/scripts/clean_legacy_results.py
```

Consigliato: archivia prima di cancellare:

```bash
python TEST_NUOVI_REPRO/scripts/clean_legacy_results.py --archive
```

Cancellazione definitiva:

```bash
python TEST_NUOVI_REPRO/scripts/clean_legacy_results.py --delete --yes
```

## Metodi implementati nel runner unico

Il runner `scripts/robust_fl_experiment.py` esegue tutte le nuove run a 50 round e salva `results/metrics.csv`, `results/args.json` e `results/last_checkpoint.pt` dentro ogni cartella esperimento.

Metodi disponibili:

- `proposed_confidence`: versione generalizzata del metodo a confidenza, con scoring su root dataset pulito e selezione tramite clustering dei punteggi.
- `fedavg`, `fedsgd`: baseline federated classiche.
- `krum`, `trimmed_mean`: baseline robuste classiche.
- `fltrust`: aggregazione guidata da update pulito lato server.
- `clipped_clustering`: clipping della norma degli update e clustering dei client prima dell'aggregazione.
- `rflpa`: weighting robusto server-guided basato su similarita' dell'update client-server.
- `adaaggrl`: aggregazione adattiva con fiducia server-guided e stabilita' storica del client.
- `fedgreed`: selezione greedy dei client che migliorano la loss sul root dataset pulito.
- `sherpa`: aggregazione robusta explainable con profilo di loss per classe su root dataset pulito, anomaly score per client e media pesata dei client non anomali.
- `fedlad`: aggregazione linear-algebra-based che seleziona update non ridondanti/linearmente indipendenti e pesa i contributi selezionati in modo robusto.

Tutti i dataset della matrice sono ora eseguibili: `cifar10`, `cifar100`, `svhn`, `fashionmnist`.

## Stato operativo

- `run_manifest.csv` contiene 360 run `ready`.
- Ogni `run.sh` usa `--num-rounds 50`.
- I risultati vecchi in `TEST_NUOVI` sono stati cancellati su richiesta.

## Download con curl

Se `torchvision` scarica lentamente, puoi pre-scaricare gli archivi pubblici con `curl` e lasciarli nella cache usata dalle run:

```bash
bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh all
```

Download parallelo:

```bash
JOBS=4 bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh cifar10 cifar100 svhn fashionmnist
```

Cache custom, se vuoi scaricare su disco esterno e poi spostare/symlinkare:

```bash
DATA_ROOT=/path/veloce/data bash TEST_NUOVI_REPRO/scripts/download_datasets_curl.sh all
```

I dataset sono pubblici: non serve token. Se usi un mirror privato, puoi scaricare altrove e poi copiare nella stessa struttura di `TEST_NUOVI_REPRO/data`.

