# Esperimenti riproducibili per rivista - variante VGG16

Copia di `TEST_NUOVI_REPRO` con lo stesso protocollo sperimentale, ma con il
modello globale sostituito da **VGG16** (Simonyan & Zisserman, ICLR 2015, con
BatchNorm), il modello usato nelle campagne precedenti di questo progetto, al
posto della `SmallCNN` usata in `TEST_NUOVI_REPRO`.

## Differenze rispetto a `TEST_NUOVI_REPRO`

- `scripts/robust_fl_experiment.py`: la classe `SmallCNN` e' sostituita da
  `VGG16` (13 conv + BatchNorm + ReLU, 5 max-pool con `ceil_mode=True` per
  gestire input piccoli come i 28x28 di FashionMNIST, seguiti da
  `AdaptiveAvgPool2d((1,1))` e un singolo `Linear(512, num_classes)`).
  Tutta la logica di federated learning, gli attacchi e i metodi di
  aggregazione (incluso `robust_fl_experiment.py`) sono invariati.
- `data/`: symlink a `../TEST_NUOVI_REPRO/data` per riusare i dataset gia'
  scaricati senza duplicarli su disco.
- `configs/experiment_matrix.json`: stessa matrice (dataset, alpha, attacchi,
  metodi), con `project` e `model` aggiornati per riflettere la variante
  VGG16.
- `runs/`, `run_manifest.csv`: generati da zero per questa campagna, separati
  da quelli di `TEST_NUOVI_REPRO`.

Per tutto il resto (uso, metodi disponibili, download dataset, parallelismo
client) vale quanto documentato in `TEST_NUOVI_REPRO/README.md`, sostituendo
`TEST_NUOVI_REPRO` con `TEST_NUOVI_REPRO_VGG16` nei comandi.

## Uso rapido

Genera la matrice:

```bash
python TEST_NUOVI_REPRO_VGG16/scripts/materialize_runs.py
```

Esegui una run pronta:

```bash
bash TEST_NUOVI_REPRO_VGG16/runs/cifar10/alpha_0p9/medio/fltrust/seed_42/run.sh
```

Esegui tutte le run pronte (o un sottoinsieme filtrato):

```bash
python TEST_NUOVI_REPRO_VGG16/scripts/run_ready.py --jobs 1
```

## Stato operativo

- `run_manifest.csv` contiene 432 run `ready` (4 dataset x 3 alpha x 3
  intensita' di attacco x 12 metodi).
- Ogni `run.sh` usa `--num-rounds 50`.
- Smoke test manuale eseguito su `fashionmnist` (1 canale, 28x28) e `cifar10`
  (3 canali, 32x32) per verificare che `VGG16` produca shape corrette in
  tutti i percorsi che ispezionano `model.features` / `model.classifier`
  (es. `proposed_confidence`, `adaaggrl`).
