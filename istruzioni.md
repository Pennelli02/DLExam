# Istruzioni per testare il progetto

## Pre-requisiti
Prima di poter far partire il programma bisogna avere delle accortezze:

- bisogna creare un'account su synapse e richiedere l'accesso al [dataset](https://www.synapse.org/Synapse:syn3193805/wiki/217789). Una volta approvato crea un api-token di synapse per l'accesso. Poi copia il token in un .env file del progetto
    
    ```SYNAPSE_TOKEN=il_tuo_token_qui```
- nel progetto è possibile usare tre modelli in base alle impostazioni del file config.yaml tra cui il modello preso da un checkpoint ufficiale. Per poterlo usare però va prima scaricato al seguente [link](https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz). Finito di scaricare inserirlo nella cartella _PreTrainedModels/imagenet21k_


## Scaricare il dataset
Per scaricare e avere a disposizione il dataset basta chiamare delle funzioni che gestiscono tutto il processo
```bash
# Con valori di default (no seed, train_ratio=0.6) ufficiale paper
python setup.py

# Con parametri personalizzati
python setup.py --random_seed 42 --train_ratio 0.7
```
Questo scarica i file da Synapse, estrae la zip e preprocessa i dati in train/val/test automaticamente.

## Lanciare il training
Per avviare il training lancia:
```bash
python train.py config.yaml
```

Il file `config.yaml` contiene tutte le impostazioni del modello. Nel progetto è possibile usare tre modelli diversi, configurabili tramite il file yaml:

- **NPT_TransUNet** – modello addestrato da zero, senza pesi pre-addestrati
- **PT_TransUNet** – modello con pesi pre-addestrati
- **CheckpointNet** – modello che usa il checkpoint ufficiale scaricato in precedenza

## Visualizzare i risultati
Per monitorare il training in tempo reale avvia TensorBoard:
```bash
tensorboard --logdir=tensorboard
```

## Testare un modello
Per eseguire l'inferenza sul test set lancia:
```bash
python test.py config.yaml
```
L'inferenza viene eseguita automaticamente con entrambi i metodi di resize (`scipy` e `v2`) e stampa un confronto finale tra i due, con le metriche Dice e HD95 per ogni organo e per ogni paziente.