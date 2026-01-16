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