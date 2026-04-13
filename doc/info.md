# Appunti su paper TransUNET
## Paper
### Dataset
2 dataset:
 - Synapse multi-organ segmentation dataset (9 classi) si ricade in 9 classi perché seguono le procedure del paper [Domain Adaptive Relational Reasoning for 3D
Multi-Organ Segmentation](https://arxiv.org/abs/2005.09120)
 - The ACDC challenge
### Architettura
(tutto pretrained però proviamo a implementare a mano)
- resnet50 pretrained su imagenet per CNN FEATURES EXTRACTOR
- ViT16 pretrained su imagenet per transformer.

## Code

### Linear Projection
```
class PT_Encoder(nn.Module):
    def __init__(self, img_size: int = 224):
        super().__init__()

        self.cnn = PTResnet()

        # Proiezione canali (come da paper: 1024 -> 768)
        self.embedding_proj = nn.Conv2d(1024, 768, kernel_size=1)
        self.flatten = nn.Flatten()
        self.vit = PreTrainedVit(img_size=img_size)
```

1. La scelta del modello ViT-BaseNel paper (Sezione 4.1. Implementation Details), gli autori dichiarano di utilizzare la configurazione "Base" del Vision Transformer (ViT-B/16). Per definizione, il modello ViT-Base ha una dimensione nascosta ($D$ o embed_dim) di 768. Tutti i token che entrano nel Transformer devono avere questa dimensione per poter essere sommati ai positional embeddings e processati dai blocchi di self-attention.
2. L'output della ResNet-50 (R50)Il paper utilizza una ResNet-50 come "backbone" per estrarre le feature map (Sezione 3.1. Transformer Encoder). Analizzando la struttura della ResNet-50, l'ultimo strato utilizzato (il terzo blocco, "Layer 3") produce una mappa di caratteristiche con 1024 canali. Poiché il Transformer "Base" accetta solo vettori di dimensione 768, è matematicamente necessario un passaggio di riduzione.
3. La "Linear Projection" (Equazione 1) Nell'equazione (1) del paper: $z_0 = [x^1_p E; x^2_p E; \dots; x^N_p E] + E_{pos}$, la matrice $E$ è definita come la patch embedding projection.Nello scenario ibrido (CNN + Transformer), i "patch" sono in realtà i pixel della feature map della CNN. Ogni "pixel" è un vettore di 1024 elementi. Per trasformare quel vettore da 1024 a 768 elementi (la dimensione $D$ del ViT), serve appunto una proiezione lineare. Sebbene il paper parli di "linear projection", nel deep learning su immagini si usa una convoluzione 1x1 perché:È computazionalmente identica a un livello lineare applicato a ogni pixel.Mantiene la struttura spaziale della mappa fino al momento del flatten.
*In sintesi: I 1024 canali sono lo standard della ResNet-50, mentre i 768 sono lo standard del ViT-Base scelto dagli autori. La proiezione 1024 -> 768 è il "ponte" obbligatorio per farli comunicare.*

### Reshape
L'output del Transformer viene definito nel paper come $z_L$. Rappresenta lo stato finale della sequenza dopo tutti i blocchi di attenzione.

| Teoria (Paper) | Valore (Mio Progetto) | Descrizione |
|---------------|----------------------|-------------|
| H, W          | 224, 224             | Risoluzione immagine di input |
| P             | 16                   | Dimensione della Patch |
| D             | 768                  | Dimensione di Embedding (ViT-Base) |
| z_L           | [batch, 196, 768]    | Output finale del Transformer |

$$$z_L \in \mathbb{R}^{\frac{H \cdot W}{P^2} \times D}$$$

Per passare dall'Encoder (Transformer) al Decoder (CUP), i dati devono cambiare dominio: da Sequenza (1D) a Griglia (2D). Senza questo passaggio, le convoluzioni non potrebbero funzionare.
```
# 1. Output dal ViT: [B, 196, 768]
# Rappresenta 196 "vettori descrittivi" in fila.
x = vit_output 

# 2. Reshape (Spatial Recovery): [B, 14, 14, 768]
# Riordiniamo i 196 token in una griglia spaziale 14x14.
x = x.view(-1, 14, 14, 768)

# 3. Permute (Channel First): [B, 768, 14, 14]
# Spostiamo l'embedding nella dimensione dei canali per le Conv2d.
x = x.permute(0, 3, 1, 2).contiguous()
```

### Upsample
nel codice si è riscontrato un problema dovuto alla scelta di quale operatore per fare upsampling:
- nn.convTraspose2d() trainable
- nn.upSample() not trainable 

Sembra dai forum di pytorch che per la segmentazione sia meglio upSample() per la segmentazione quindi procedo.

La ConvTranspose2d soffre spesso di un problema chiamato effetto a scacchiera. Poiché ha pesi addestrabili e sovrappone i calcoli durante l'espansione, può creare dei pattern ripetitivi che disturbano i bordi della segmentazione. In medicina, dove la precisione del bordo di un organo è vitale, questo è un grosso rischio.
Secondo l'articolo ["Deconvolution and Checkerboard Artifacts" (Odena et al.)](https://distill.pub/2016/deconv-checkerboard/?ref=mlq-ai) risulta molto più vantaggioso usare l'unsampling con metodi nearest e bilinear dato che non crea artefatti nell'immagine.
L'algoritmo di interpolazione bilineare è meno efficiente dal punto di vista computazionale rispetto al metodo "near neighbor", ma offre un'approssimazione più precisa. Il valore di un singolo pixel viene calcolato come media ponderata di tutti gli altri valori in base alle distanze.

Nel codice verrà utilizzato **nn.Upsample(mode="bilinear")**

### Test model paper

Le skip connections vengono salvate a diversi livelli del ResNet50 encoder:
```python
# Dal ResNet50.forward():
x1 = self.relu(x)           # Skip 1: [B, 64, 112, 112]  ← 64 canali!
x2 = self.layer1_block3(x)  # Skip 2: [B, 256, 56, 56]   ← 256 canali
x3 = self.layer2_block4(x)  # Skip 3: [B, 512, 28, 28]   ← 512 canali

return x4, [x1, x2, x3]
```
La **skip1** viene salvata **subito dopo Conv1 + BatchNorm + ReLU**, quando l'immagine è stata ridotta spazialmente a 112x112 ma i canali sono ancora **64**. Non è ancora passata attraverso il layer1 che espande i canali a 256.

#### Architettura Dettagliata ResNet50:
```
Input: 224x224x3
   ↓ Conv1 (kernel=7, stride=2, padding=3)
112x112x64  
   ↓ BatchNorm + ReLU
112x112x64 ← Skip1 salvata qui (64 canali)
   ↓ MaxPool (kernel=3, stride=2, padding=1)
56x56x64
   ↓ Layer1 (3x Bottleneck blocks)
56x56x256  ← Skip2 salvata qui (256 canali)
   ↓ Layer2 (4x Bottleneck blocks, stride=2 nel primo)
28x28x512  ← Skip3 salvata qui (512 canali)
   ↓ Layer3 (6x Bottleneck blocks, stride=2 nel primo)
14x14x1024  ← Output CNN per Transformer
```
pretrained su ImageNet1k.v1 e passeremo adesso su una versione Imagenet1k.v2. In teoria risulta essere imagenet21 però ricerco i pesi
#### Dimensioni Corrette del Decoder CUP
```python
class CUP(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 64):
        super().__init__()
        
        # Block 1: 768 → 512 (no skip)
        self.layer1 = CUPBlock(in_channels=768, out_channels=512)
        
        # Block 2: 512 + 512 (skip3) = 1024 totali → 256 out
        self.layer2 = CUPBlock(in_channels=1024, out_channels=256)
        
        # Block 3: 256 + 256 (skip2) = 512 totali → 128 out
        self.layer3 = CUPBlock(in_channels=512, out_channels=128)
        
        # Block 4: 128 + 64 (skip1) = 192 totali → 64 out
        self.layer4 = CUPBlock(in_channels=192, out_channels=64)
```
Flusso Completo del Decoder:
```
14x14x768 (dal Transformer)
   ↓ CUP Block 1 (no skip)
28x28x512
   ↓ CUP Block 2 + Skip3 (28x28x512)
   | Concatenazione: 512 + 512 = 1024 canali
56x56x256
   ↓ CUP Block 3 + Skip2 (56x56x256)
   | Concatenazione: 256 + 256 = 512 canali
112x112x128
   ↓ CUP Block 4 + Skip1 (112x112x64)
   | Concatenazione: 128 + 64 = 192 canali
224x224x64
   ↓ Last Conv Layer
224x224x16
   ↓ Segmentation Head
224x224x9 (output finale)
```

| Layer Decoder | Input Decoder | Skip Connection | Canali Totali | Output     |
|---------------|---------------|-----------------|-----------|------------|
| CUP Block 1   | 14x14x768     | Nessuna         | 768       | 28x28x512  |
| CUP Block 2   | 28x28x512     | 28x28x512       | 1024      | 56x56x256  |
| CUP Block 3   | 56x56x256     | 56x56x256       | 512       | 112x112x128|
| CUP Block 4   | 112x112x128   | 112x112x64      | 192       | 224x224x64 |


### BatchNorm2d
osservando nei forum e dagli appunti che dopo una relu e quasi sempre richiesto una batchnorm cosa che nel paper non risulta che nel decoder si usi però verrà implementata lo stesso 
### Pre-Processing

Il dataset Synapse contiene 30 scansioni CT volumetriche 3D che richiedono preprocessing per essere utilizzabili nel training di TransUNet.

#### 1. Download Dataset
```python
getDataset()
```
Scarica automaticamente il dataset dal repository Synapse usando l'API con autenticazione token personale (richiede registrazione su synapse.org).

#### 2. Estrazione File
```python
setup_synapse_dataset()
```
Estrae `RawData.zip` e organizza i file NIfTI nella struttura:
```
dataset/RawData/RawData/Training/
├── img/    # 30 volumi CT (.nii.gz)
└── label/  # 30 maschere di segmentazione
```

#### 3. Preprocessing
```python
preprocess_synapse()
```

**Operazioni principali:**

- **HU Windowing**: `clip(image, -125, 275)` - finestra addominale standard per tessuti molli
- **Normalization**: `(image + 125) / 400` - normalizzazione in range [0, 1]
- **Dataset Split**: 18 training cases / 12 validation cases
- **Format Conversion**:
  - Training → 2D slices (`.npz`) per memory efficiency (~2212 slices)
  - Validation → 3D volumes (`.h5`) per metriche volumetriche (Dice Score, Hausdorff Distance)

**Output:**
```
dataset/project_transunet/
├── train_npz/      # ~2212 slice 2D (512×512)
└── validation_vol_h5/    # 12 volumi 3D (per validation)
```

**Note**: La cartella `RawData/Testing/` (20 volumi senza label) non viene utilizzata, seguendo l'implementazione del paper originale.

*alcune informazioni sono riprese da altri paper citati*

### Dataset Augmentation
per una maggiore efficienza e perché più recente utilizzeremo v2 per l'augmentation

### Inference
L'inferenza è stata implementata seguendo un protocollo slice-by-slice coerente con lo stato dell'arte (Zhou et al. [19]) . Per garantire la massima fedeltà anatomica, è stata utilizzata una pipeline di trasformazione differenziata: interpolazione bilineare con antialiasing per il ricampionamento delle scansioni CT e interpolazione nearest-neighbor per le maschere di segmentazione, preservando l'integrità delle etichette categoriali durante il volume-stacking.

### Loss
Risulta dall'articolo che il modello nel training usa come loss una funzione combinata tra DiceLoss e CrossEntropy Loss

### Training and Validation
La validation avverrà non ad ogni iterazione di batch, ma ad ogni 2 epoche perché essendo costosa e consiste in una valutazione delle metriche per volume diventa pesante
Inoltre, è stato deciso di ridurre per una questione di memoria il batch a 8 immagini
La validazione sarà sia globale (avg) e singola per ogni organo per avere una visione più dettagliata

### Numero di iterazioni
nel paper è stato fissato 14.000 iterazioni però dato che il mio computer non riesce a gestire 24 di batch ho optato per un 8 di batch e quindi i valori delle epoche cambiano e quindi per essere sicuri è stato deciso di aumentare il numero di iterazioni (52 epoche * lenght(train) circa 14k)

### Learning Rate
nel paper si usa 0.01 come learning rate però dato il cambiamento del batch utilizzeremo:  Square Root Scaling Facebook Research (Goyal et al. 2017) passando a 0.006

## Primo Training 
Loss: 0.0639 DiceScore: 0.8953 CE: 0.0231

non mi torna al momento la validazione e nella visualizzazione di tensorboard spesso immagini tutte nere
Inoltre, non riesce a identificare il Liver e Right Kidney
## Secondo Training
64% e 39% rispettivamente  in DSC e HD95 nella validation

## Terzo Training 
70% e 37%

## Quarto training
70% e 39% la normalizzazione statistica non ha colmato il gap

## Commenti
al momento otteniamo risultati buoni, ma non sufficienti. Valutiamo cosa può comportare queste grandi perdite. Optiamo per un batch di 24

Al momento ottengo in validazione  MEDIA TOTALE    -> Dice: 0.6658 | HD95: 47.26 epoca 96 su 152

Proviamo con una normalizzazione statistica su imagenet se non riusciamo a colmare il gap allora si deve passare all'utilizzo di imagenet21k pre trained

### Velocità training
 accellerazione del training usando GradScalar() e num_workers per la validation possibile collo di bottiglia con dataloaders
 
### Modelli Pretrained
seguendo il paper abbiamo recuperato il checkpoint R50+ViT-B_16.npz perché loro usano un modello tutto insieme da fine-tunare
cartella è : PreTrainedModels\imagenet21k\R50+ViT-B_16.npz

da valutare: *tale modello pretrained utilizza la group norm e non la classica batch norm*

### Possibili soluzioni
uso del checkpoint non sembra migliorare come funziona il modello poi a fine training valuterò.

- valutare se pesi caricati correttamente caricati in modo giusto

L'errore sembra essere che dato che sto usando un modello di Big Transfer devo attuare delle pre trasformazioni per far si che il transfer learning funzioni

### Osservazioni
notiamo che con un seed fissato e l'utilizzo del modello checkpointnet che un aumento dei layer convoluzionali (conv3x3->bn->relu) diminuisce il valore di HD95 medio, ma non influenza i valori di DSC
notiamo che in genere il modello PT_Transunet genera valori con alto DSC a discapito della HD95 questo suggerisce di provare con un seed fisso e con più layer convoluzionali

**Ovviamente bisogna capire cosa abbassi di 3% la DSC media nel test**

**Gli organi che influiscono tanto sono Gallbladder e Pancreas**

### checkpoint 1
CheckpointNet con doppio layer convoluzionale

           INFO      RISULTATI VALIDAZIONE (resize_type='v2')      train.py:189
           INFO     ============================================== train.py:190
                    ==============                                             
           INFO     Aorta           -> Dice: 0.8161 | HD95: 11.22  train.py:192
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.5194 | HD95: 26.38  train.py:192
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.7876 | HD95: 47.24  train.py:192
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.7474 | HD95: 24.66  train.py:192
                    mm                                                         
           INFO     Liver           -> Dice: 0.9276 | HD95: 36.13  train.py:192
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.5313 | HD95: 15.48  train.py:192
                    mm                                                         
           INFO     Spleen          -> Dice: 0.8533 | HD95: 43.00  train.py:192
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7368 | HD95: 16.44  train.py:192
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.7399 | HD95: 27.57  train.py:193
                    mm                                                         
           INFO      Validazione volumetrica avviata               train.py:164
                    (resize_type='scipy')...                                   
                                train.py:181
           INFO                                                    train.py:188
                    ==============================================             
                    ==============                                             
           INFO      RISULTATI VALIDAZIONE (resize_type='scipy')   train.py:189
           INFO     ============================================== train.py:190
                    ==============                                             
           INFO     Aorta           -> Dice: 0.8323 | HD95: 7.02   train.py:192
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.5019 | HD95: 28.12  train.py:192
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.7981 | HD95: 47.44  train.py:192
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.7510 | HD95: 24.73  train.py:192
                    mm                                                         
           INFO     Liver           -> Dice: 0.9301 | HD95: 36.03  train.py:192
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.5339 | HD95: 15.41  train.py:192
                    mm                                                         
           INFO     Spleen          -> Dice: 0.8570 | HD95: 25.91  train.py:192
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7427 | HD95: 16.32  train.py:192
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.7434 | HD95: 25.12  train.py:193
                    mm


### checkpoint test

    [18:38:31] INFO      img0038 processato                            train.py:180
    [18:38:32] INFO                                                    train.py:187
                    ==============================================             
                    ==============                                             
           INFO      RISULTATI VALIDAZIONE (resize_type='v2')      train.py:188
           INFO     ============================================== train.py:189
                    ==============                                             
           INFO     Aorta           -> Dice: 0.8083 | HD95: 8.72   train.py:191
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.5747 | HD95: 11.65  train.py:191
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.7893 | HD95: 61.92  train.py:191
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.7433 | HD95: 59.30  train.py:191
                    mm                                                         
           INFO     Liver           -> Dice: 0.9306 | HD95: 35.54  train.py:191
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.5757 | HD95: 11.72  train.py:191
                    mm                                                         
           INFO     Spleen          -> Dice: 0.8356 | HD95: 40.19  train.py:191
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7324 | HD95: 20.02  train.py:191
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.7488 | HD95: 31.13  train.py:192
                    mm                                                         
           INFO      Validazione volumetrica avviata               train.py:163
                    (resize_type='scipy')...                                    

                    ==============================================             
                    ==============                                             
           INFO      RISULTATI VALIDAZIONE (resize_type='scipy')   train.py:188
           INFO     ============================================== train.py:189
                    ==============                                             
           INFO     Aorta           -> Dice: 0.8242 | HD95: 8.32   train.py:191
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.5678 | HD95: 10.11  train.py:191
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.7994 | HD95: 53.98  train.py:191
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.7463 | HD95: 50.10  train.py:191
                    mm                                                         
           INFO     Liver           -> Dice: 0.9338 | HD95: 35.74  train.py:191
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.5754 | HD95: 13.76  train.py:191
                    mm                                                         
           INFO     Spleen          -> Dice: 0.8411 | HD95: 40.22  train.py:191
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7396 | HD95: 19.79  train.py:191
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.7535 | HD95: 29.00  train.py:192
                    mm                                                         
    [19:56:20] INFO     Saved checkpoint transunet_ckp_test\e_00152.chp train.py:44
           INFO      End training 
### checkpoint rep
architettura classica con una sola conv ma con split diverso
  
  INFO      RISULTATI VALIDAZIONE (resize_type='v2')      train.py:188
           INFO     ============================================== train.py:189
                    ==============                                             
           INFO     Aorta           -> Dice: 0.8322 | HD95: 11.85  train.py:191
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.6953 | HD95: 7.31   train.py:191
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.7862 | HD95: 87.86  train.py:191
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.7581 | HD95: 53.08  train.py:191
                    mm                                                         
           INFO     Liver           -> Dice: 0.9409 | HD95: 15.66  train.py:191
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.4826 | HD95: 21.77  train.py:191
                    mm                                                         
           INFO     Spleen          -> Dice: 0.8689 | HD95: 71.83  train.py:191
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7316 | HD95: 15.80  train.py:191
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.7620 | HD95: 35.65  train.py:192
                    mm                                                         
           INFO      Validazione volumetrica avviata               train.py:163
                    (resize_type='scipy')...                                   
        INFO                                                    train.py:187
                    ==============================================             
                    ==============                                             
           INFO      RISULTATI VALIDAZIONE (resize_type='scipy')   train.py:188
           INFO     ============================================== train.py:189
                    ==============                                             
           INFO     Aorta           -> Dice: 0.8454 | HD95: 11.77  train.py:191
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.6809 | HD95: 8.06   train.py:191
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.7861 | HD95: 88.03  train.py:191
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.7553 | HD95: 75.67  train.py:191
                    mm                                                         
           INFO     Liver           -> Dice: 0.9396 | HD95: 16.22  train.py:191
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.4785 | HD95: 21.21  train.py:191
                    mm                                                         
           INFO     Spleen          -> Dice: 0.8678 | HD95: 70.96  train.py:191
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7283 | HD95: 16.37  train.py:191
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.7602 | HD95: 38.54  train.py:192
                    mm


---
## Situazione attuale
- **training**: 18 scan mediche suddivise in slices
- **validation**: 6 scan mediche in 3d 
- **testing**: 6 scan mediche in 3d

Miglior risultato ottenuto *(però non ho seguito la situazione attuale)*: transunet_ckp_v2/e_00076.chp 



## Risultati
### checkpoint
otteniamo soprattutto a causa della divisione dei risultati sovrastimati con il paper
architettura classica con doppia convoluzione nel decoder

    INFO     scipy  -> Dice: 0.6591 | HD95: 36.88 mm         test.py:174
    INFO     v2     -> Dice: 0.6560 | HD95: 36.93 mm         test.py:174
### checkpointv1
architettura classica con singola convoluzione nel decoder

    INFO     scipy  -> Dice: 0.6689 | HD95: 41.63 mm         test.py:174
    INFO     v2     -> Dice: 0.6682 | HD95: 41.81 mm         test.py:174
### checkpointv2
architettura del checkpoint però a differenza degli altri checkpoint non è presente la separazione tra validation e testing, ma solo training e validation per avere un confronto con il paper

### checkpoint test
Architettura: PT_transunet

    INFO     scipy  -> Dice: 0.6574 | HD95: 43.20 mm         test.py:179
    INFO     v2     -> Dice: 0.6797 | HD95: 42.36 mm         test.py:179

---
## Testing singolo paziente
osservando i risultati del singolo paziente notiamo che otteniamo valori sottostimati perché essendo che abbiamo pochi casi di test (6) e tanta variabilità il valore medio ne risente
Infatti, si osserva in questo mio contesto:

    ────────────────────────────────────────────────           
    ──                                                         
          Paziente: img0002                               test.py:69
         ──────────────────────────────────────────────── test.py:70
                Organo          |     Dice |  HD95 (mm)          test.py:71
                ────────────────────────────────────────         test.py:72
                Aorta           |   0.8371 |       5.74          test.py:78
                Gallbladder     |   0.6582 |       5.39          test.py:78
                Left Kidney     |   0.9329 |       1.73          test.py:78
                Right Kidney    |   0.8760 |       3.61          test.py:78
                Liver           |   0.9510 |       3.46          test.py:78
                Pancreas        |   0.5432 |       9.27          test.py:78
                Spleen          |   0.9345 |       1.41          test.py:78
                Stomach         |   0.8638 |       4.90          test.py:78
                ────────────────────────────────────────         test.py:83
                MEDIA PAZIENTE  |   0.8246 |       4.44          test.py:84
    [14:44:55] INFO                                                      test.py:68
                    ────────────────────────────────────────────────           

           INFO      Paziente: img0003                               test.py:69
           INFO     ──────────────────────────────────────────────── test.py:70
                    ──                                                         
           INFO     Organo          |     Dice |  HD95 (mm)          test.py:71
           INFO     ────────────────────────────────────────         test.py:72
           INFO     Aorta           |   0.7402 |      20.35          test.py:78
           INFO     Gallbladder     |   0.1580 |      24.12          test.py:78
           INFO     Left Kidney     |   0.2964 |     153.39          test.py:78
           INFO     Right Kidney    |   0.4716 |     155.01          test.py:78
           INFO     Liver           |   0.8642 |     124.14          test.py:78
           INFO     Pancreas        |   0.3642 |      41.70          test.py:78
           INFO     Spleen          |   0.6236 |     220.88          test.py:78
           INFO     Stomach         |   0.6946 |      33.67          test.py:78
           INFO     ────────────────────────────────────────         test.py:83
           INFO     MEDIA PAZIENTE  |   0.5266 |      96.66          test.py:84
    [14:46:51] INFO                                                      test.py:68
                    ────────────────────────────────────────────────           
                    ──                                                         
           INFO      Paziente: img0009                               test.py:69
           INFO     ──────────────────────────────────────────────── test.py:70
                    ──                                                         
           INFO     Organo          |     Dice |  HD95 (mm)          test.py:71
           INFO     ────────────────────────────────────────         test.py:72
           INFO     Aorta           |   0.7299 |      27.33          test.py:78
           INFO     Gallbladder     |   0.0000 |     163.81          test.py:78
           INFO     Left Kidney     |   0.0643 |     121.88          test.py:78
           INFO     Right Kidney    |   0.0000 |     148.46          test.py:78
           INFO     Liver           |   0.4642 |      90.53          test.py:78
           INFO     Pancreas        |   0.2041 |      22.11          test.py:78
           INFO     Spleen          |   0.0025 |     152.93          test.py:78
           INFO     Stomach         |   0.6575 |      33.73          test.py:78
           INFO     ────────────────────────────────────────         test.py:83
           INFO     MEDIA PAZIENTE  |   0.2653 |      95.10          test.py:84
    [14:48:05] INFO                                                      test.py:68
                    ────────────────────────────────────────────────           
                    ──                                                         
           INFO      Paziente: img0024                               test.py:69
           INFO     ──────────────────────────────────────────────── test.py:70
                    ──                                                         
           INFO     Organo          |     Dice |  HD95 (mm)          test.py:71
           INFO     ────────────────────────────────────────         test.py:72
           INFO     Aorta           |   0.8853 |       2.45          test.py:78
           INFO     Gallbladder     |   0.0000 |      66.82          test.py:78
           INFO     Left Kidney     |   0.9121 |       2.00          test.py:78
           INFO     Right Kidney    |   0.9091 |       4.00          test.py:78
           INFO     Liver           |   0.9262 |      20.27          test.py:78
           INFO     Pancreas        |   0.5580 |      20.35          test.py:78
           INFO     Spleen          |   0.9515 |       2.00          test.py:78
           INFO     Stomach         |   0.6903 |      10.82          test.py:78
           INFO     ────────────────────────────────────────         test.py:83
           INFO     MEDIA PAZIENTE  |   0.7291 |      16.09          test.py:84
    [14:48:52] INFO                                                      test.py:68
                    ────────────────────────────────────────────────           
                    ──                                                         
           INFO      Paziente: img0035                               test.py:69
           INFO     ──────────────────────────────────────────────── test.py:70
                    ──                                                         
           INFO     Organo          |     Dice |  HD95 (mm)          test.py:71
           INFO     ────────────────────────────────────────         test.py:72
           INFO     Aorta           |   0.5507 |      14.77          test.py:78
           INFO     Gallbladder     |   1.0000 |       0.00          test.py:78
           INFO     Left Kidney     |   0.9041 |       2.00          test.py:78
           INFO     Right Kidney    |   0.8808 |       2.00          test.py:78
           INFO     Liver           |   0.9412 |       5.00          test.py:78
           INFO     Pancreas        |   0.6016 |       9.00          test.py:78
           INFO     Spleen          |   0.9463 |       1.41          test.py:78
           INFO     Stomach         |   0.6966 |       7.87          test.py:78
           INFO     ────────────────────────────────────────         test.py:83
           INFO     MEDIA PAZIENTE  |   0.8152 |       5.26          test.py:84
    [14:49:42] INFO                                                      test.py:68
                    ────────────────────────────────────────────────           
                    ──                                                         
           INFO      Paziente: img0039                               test.py:69
           INFO     ──────────────────────────────────────────────── test.py:70
                    ──                                                         
           INFO     Organo          |     Dice |  HD95 (mm)          test.py:71
           INFO     ────────────────────────────────────────         test.py:72
           INFO     Aorta           |   0.5752 |       4.00          test.py:78
           INFO     Gallbladder     |   0.7300 |       3.00          test.py:78
           INFO     Left Kidney     |   0.8243 |       3.00          test.py:78
           INFO     Right Kidney    |   0.8016 |       3.00          test.py:78
           INFO     Liver           |   0.9523 |       2.24          test.py:78
           INFO     Pancreas        |   0.6388 |      10.77          test.py:78
           INFO     Spleen          |   0.9494 |       1.41          test.py:78
           INFO     Stomach         |   0.8812 |       2.24          test.py:78
           INFO     ────────────────────────────────────────         test.py:83
           INFO     MEDIA PAZIENTE  |   0.7941 |       3.71          test.py:84
    [14:49:43] INFO                                                      test.py:97
                    ================================================           
                    ============                                               
           INFO      RISULTATI TESTING (resize_type='scipy')         test.py:98
           INFO     ================================================ test.py:99
                    ============                                               
           INFO     Aorta           -> Dice: 0.7197 | HD95: 12.44   test.py:104
                    mm                                                         
           INFO     Gallbladder     -> Dice: 0.4244 | HD95: 43.86   test.py:104
                    mm                                                         
           INFO     Left Kidney     -> Dice: 0.6557 | HD95: 47.33   test.py:104
                    mm                                                         
           INFO     Right Kidney    -> Dice: 0.6565 | HD95: 52.68   test.py:104
                    mm                                                         
           INFO     Liver           -> Dice: 0.8499 | HD95: 40.94   test.py:104
                    mm                                                         
           INFO     Pancreas        -> Dice: 0.4850 | HD95: 18.87   test.py:104
                    mm                                                         
           INFO     Spleen          -> Dice: 0.7346 | HD95: 63.34   test.py:104
                    mm                                                         
           INFO     Stomach         -> Dice: 0.7473 | HD95: 15.54   test.py:104
                    mm                                                         
           INFO     MEDIA TOTALE    -> Dice: 0.6591 | HD95: 36.88   test.py:106
                    mm                                                         
           INFO 