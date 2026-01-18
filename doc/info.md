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
- nn.unSample() not trainable 

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
112x112x64  ← Skip1 salvata QUI! (64 canali)
   ↓ BatchNorm + ReLU
112x112x64
   ↓ MaxPool (kernel=3, stride=2, padding=1)
56x56x64
   ↓ Layer1 (3x Bottleneck blocks)
56x56x256  ← Skip2 salvata qui (256 canali)
   ↓ Layer2 (4x Bottleneck blocks, stride=2 nel primo)
28x28x512  ← Skip3 salvata qui (512 canali)
   ↓ Layer3 (6x Bottleneck blocks, stride=2 nel primo)
14x14x1024  ← Output CNN per Transformer
```

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