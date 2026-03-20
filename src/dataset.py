import os
import random

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Importa i tipi speciali di tensori: Image, Mask, BoundingBox, etc.
# Servono per dire a v2 come trattare ogni tipo di dato
from torchvision import tv_tensors

# Importa le trasformazioni moderne (v2) di torchvision
# v2 è la versione aggiornata che gestisce meglio immagini + maschere
from torchvision.transforms import v2

class SynapseDataset(Dataset):
    def __init__(self, opts, data_dir, split='train', transform=None):
        self.opts = opts
        self.split = split
        self.transform = transform
        self.base_dir = Path(data_dir)
        self.sample_list = self._get_sample_list()

    def _get_sample_list(self):
        """
           Scansiona la cartella e ottiene la lista dei file
        """
        if self.split == "train":
            # Trova tutti i .npz e rimuovi l'estensione
            files = sorted(self.base_dir.glob('*.npz'))
            sample_list = [f.stem for f in files]  # .stem rimuove l'estensione
        else:  # test
            # Trova tutti i .h5
            files = sorted(self.base_dir.glob('*.npy.h5'))
            sample_list = [f.name.replace('.npy.h5', '') for f in files]

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Metodo che restituisce un singolo campione dato l'indice
        # PyTorch chiama questo metodo per ogni batch
        # idx: indice del campione da caricare (0, 1, 2, ..., len-1)
        if self.split == "train":
            # MODALITÀ TRAINING: carica slice 2D da file .npz

            # Prende il nome del file dalla lista e rimuove il carattere '\n'
            slice_name = self.sample_list[idx].strip('\n')

            # Costruisce il percorso completo del file
            data_path = os.path.join(self.base_dir, slice_name + '.npz')

            # Carica il file .npz (archivio NumPy compresso)
            # Contiene 2 array: 'image' e 'label'
            data = np.load(data_path)

            # Estrae l'immagine e la maschera dall'archivio
            # image: array [H, W] con valori di intensità
            # label: array [H, W] con classi degli organi
            image, label = data['image'], data['label']

        else:
            #MODALITA' VALIDATION:  carica volume 3D da file .h5

            # Prende il nome del volume e rimuove '\n'
            vol_name = self.sample_list[idx].strip('\n')

            filepath = os.path.join(self.base_dir, f"{vol_name}.npy.h5")

            data = h5py.File(filepath, 'r')

            # Carica i dati dal file H5:
            # [:] carica TUTTO l'array in memoria
            # image: [D, H, W] dove D = numero di slice nel volume
            # label: [D, H, W] con le maschere 3D
            image, label = data['image'][:], data['label'][:]

        # CONVERSIONE IN TENSORI PYTORCH
        #print(f"DEBUG - Max: {image.max()}, Min: {image.min()}")
        # 1. .astype(np.float32) → converte in float32 (richiesto da PyTorch)
        # 2. torch.from_numpy() → converte NumPy array in tensore PyTorch
        # 3. .unsqueeze(0) → aggiunge dimensione canale → [1, H, W]
        #    (PyTorch vuole sempre [C, H, W])
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': tv_tensors.Image(image), 'label': tv_tensors.Mask(label)}

        # APPLICAZIONE DELLE TRASFORMAZIONI

        if self.transform:
            # Se è stata passata una pipeline di trasformazioni

            sample = self.transform(sample)

        return {
            # Restituisce l'immagine trasformata [C, H, W]
            'image': sample['image'],

            # Restituisce la label trasformata [H, W] convertita in long (int64)
            'label': sample['label'].long(),

            # Restituisce il nome del caso per tracking
            # Utile per salvare le predizioni con il nome originale
            'case_name': self.sample_list[idx].strip('\n')
        }
    # PIPELINE DI TRAINING
def get_train_transform(opts):
    # Funzione che crea la pipeline di augmentation per il training
    # output_size: dimensione finale dell'immagine (TransUNet usa 224x224)
    return v2.Compose([
        # Ridimensiona immagine e maschera a 224x224
        # antialias=True: applica anti-aliasing per immagini più smooth
        v2.Resize(size=(opts.image_size, opts.image_size), antialias=True),

        # gira l'immagine di 90 180 270 gradi
        RandomRot90(p=opts.flip_prob),
        # Ribalta orizzontalmente con probabilità 50%
        v2.RandomHorizontalFlip(p=opts.flip_prob),

        # Ribalta verticalmente con probabilità 50%
        v2.RandomVerticalFlip(p=opts.flip_prob),

        # Ruota casualmente l'immagine tra -20° e +20°
        v2.RandomRotation(degrees=opts.degrees, expand=False),

        # Converte il tipo di dato in float32 compatibile con tv_tensor
        v2.ToDtype(torch.float32, scale=False),
    ])

class RandomRot90:
    def __init__(self, p=0.5):
        self.p = p
    """Ruota di k×90° casuale, applicata sia a image che a mask."""
    def __call__(self, sample):
        if random.random() > self.p:
            k = np.random.randint(0, 4)
            # torch.rot90 funziona su tensori, dims=(H,W) = (-2,-1)
            sample['image'] = tv_tensors.wrap(
                torch.rot90(sample['image'], k, dims=[-2, -1]),
                like=sample['image']  # mantiene il tipo tv_tensors.Image
            )
            sample['label'] = tv_tensors.wrap(
                torch.rot90(sample['label'], k, dims=[-2, -1]),
                like=sample['label']  # mantiene il tipo tv_tensors.Mask
            )
        return sample


# non sono sicuro di questa scelta di creare questa classe
# class MakeDataloader:
#     def __init__(self, opts):
#         self.opts = opts
#         self.train_dataloader = DataLoader(SynapseDataset(opts, opts.train_dir,'train', transform=get_train_transform(opts)), batch_size=opts.batch_size, shuffle=True)
#         self.validation_dataloader = DataLoader(SynapseDataset(opts, opts.validation_dir, 'val', transform=None), batch_size=1, shuffle=False)
