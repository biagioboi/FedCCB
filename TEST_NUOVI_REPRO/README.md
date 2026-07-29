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
- `krum`: Blanchard et al., NeurIPS 2017. Sceglie l'update con distanza minima ai suoi `n-f-2` vicini piu' prossimi (`f` = numero assunto di client Byzantine, `--krum-f`, di default stimato dal numero di `--malicious-clients` configurati).
- `trimmed_mean`: Yin et al., ICML 2018. Trimmed mean coordinata-per-coordinata.
- `fltrust`: Cao et al., NDSS 2021. Trust score `ReLU(cos(g_i, g_0))` **dopo** aver normalizzato ogni update alla norma dell'update server (`g_i_bar = (||g_0||/||g_i||) * g_i`), poi media pesata dei `g_i_bar`.
- `rflpa`: arXiv:2405.15182 (NeurIPS 2024). Stessa formula di trust bootstrapping di FLTrust (con secure aggregation nel paper originale, qui omessa perche' fuori scope simulativo).
- `clipped_clustering`: arXiv:2302.07173. Clip della norma al **valore storico mediano** (non al mediano del solo round corrente), poi bipartizione dei client via clustering agglomerativo su similarita' coseno (Sattler et al., ICASSP 2020) mantenendo il cluster maggioritario.
- `adaaggrl`: arXiv:2406.14217 (AAAI 2025). Ricostruzione della distribuzione dati di ogni client via gradient inversion, similarita' MMD (storia del client, distribuzione globale del round), policy TD3 che apprende i pesi delle 4 metriche + soglia di filtro, penalita' esponenziale `lambda^h` per client ripetutamente sospetti. NOTA: la gradient inversion del paper assume un singolo step di SGD; qui i client fanno piu' epoche locali, quindi l'update cumulativo e' usato come proxy del gradiente target (approssimazione necessaria). L'estrattore di feature "pre-addestrato" e' il backbone convoluzionale del modello globale corrente (nessuna rete pre-addestrata esterna e' disponibile in questa pipeline).
- `fedgreed`: arXiv:2508.18060. Ordina i client una sola volta per loss individuale sul root dataset, poi testa i prefissi in quest'ordine fisso fermandosi al primo prefisso che non migliora la loss.
- `sherpa`: IEEE S&P 2024. Identifica i poisoner via clustering delle attribuzioni SHAP (non tramite statistiche di loss): per ogni modello candidato calcola l'attribuzione media |SHAP value| per classe con un explainer a gradiente, bipartiziona i client su questi vettori e scarta il cluster minoritario.
- `fedlad`: arXiv:2508.02136. Eliminazione di Gauss-Jordan con pivoting parziale (RREF) sulla matrice parametri x client per individuare le colonne (client) linearmente indipendenti, poi FedAvg non pesato sui soli client selezionati.

Tutti i dataset della matrice sono ora eseguibili: `cifar10`, `cifar100`, `svhn`, `fashionmnist`.

## Parallelismo client

Le run addestrano i client in parallelo tramite `--client-parallelism 0`, dove `0` significa usare tutti i client della run. Per tornare al comportamento seriale e ridurre il picco di memoria GPU, passa `--client-parallelism 1`.

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

