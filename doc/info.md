# Appunti su paper 3d e progetto
## Paper
### Dataset
Scegliere tra due dataset
- Beyond the Cranial
Vault (BTCV) (più plausibile) [link dataset](https://www.synapse.org/Synapse:syn3193805/files/)
-  Medical Segmentation Decathlon
(MSD) (tante varianti pesanti da scegliere solo una)--> nel paper stato dell'arte in cervello e spleen [link dataset](http://medicaldecathlon.com/)
 ### Architettura 
Il transformer encoder segue le caratteristiche di ViT-B16 no pretrained


# Appunti su paper TransUNET
## Paper
### Dataset
2 dataset:
 - Synapse multi-organ segmentation dataset
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