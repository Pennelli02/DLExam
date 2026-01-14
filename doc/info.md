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
