import random
import zipfile
from pathlib import Path
import os
import glob
import numpy as np
import nibabel as nib
import h5py
import torch
from tqdm import tqdm
import synapseclient
import synapseutils
import SimpleITK as sitk
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from medpy import metric

from src.transUNet import PT_TransUNet


def getDataset():
    # Percorso dove salvare i file
    dataset_dir = os.path.join("src", "dataset")

    # Crea la cartella se non esiste
    os.makedirs(dataset_dir, exist_ok=True)

    # Login a Synapse
    syn = synapseclient.Synapse()
    syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc2ODc2MDU3NywiaWF0IjoxNzY4NzYwNTc3LCJqdGkiOiIzMTE1NyIsInN1YiI6IjM1NzEwMzUifQ.oLL7_Syvjxzwb-pNnUAZ0sqoORTheECa7F04eiNc2qSOA2h-ITIPEKGbZOoTp8IZ-Kd1yRICqV4A_p7wlE7Rsh28_bOwgoDXN-265ZV-srXpvCrokkbWtz7bMwkNYtMO4DxAnoMzs9XNQI32-2Xu_RwVbVcEzmrtaLpPd5PxryRtnywzCLEm2lJBKsDekAS_Gib3eE9JZs_TZ6VoLKKqYmkynCinH_t2v3meGFxTP23BKRu_9ReSU6Q0CPlxwsKtiA7DSsywx4T7wiGVon5Rlib099UaieGF41RKJTT3ScJmiMrE9Fhd9q4l1fmpymx0eMT3FIqHLDKMwIe4qboLKg")

    # Scarica i file nella cartella src/dataset
    files = synapseutils.syncFromSynapse(
        syn,
        'syn3193805',
        path=dataset_dir
    )

    return files

def setup_synapse_dataset():
    """Estrae e organizza il dataset Synapse"""

    # Percorsi
    base_dir = Path("dataset/Abdomen")
    rawdata_zip = base_dir / "RawData.zip"
    output_dir = Path("dataset/RawData")

    print("=" * 60)
    print("SYNAPSE DATASET SETUP")
    print("=" * 60)

    # Verifica che il file zip esista
    if not rawdata_zip.exists():
        print(f" ERROR: {rawdata_zip} not found!")
        print("\nAvailable files:")
        for f in base_dir.glob("*.zip"):
            print(f"  - {f.name}")
        return False

    print(f"\n✓ Found: {rawdata_zip}")
    print(f"  Size: {rawdata_zip.stat().st_size / (1024 ** 3):.2f} GB")

    # Estrai il file
    print(f"\n Extracting to: {output_dir}")
    print("   This may take a few minutes...")

    try:
        with zipfile.ZipFile(rawdata_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(" Extraction complete!")
    except Exception as e:
        print(f" Extraction failed: {e}")
        return False

    # Verifica la struttura estratta
    print("\n Checking extracted structure...")

    expected_paths = [
        output_dir / "RawData" / "Training" / "img",
        output_dir / "RawData" / "Training" / "label",
    ]

    for path in expected_paths:
        if path.exists():
            files = list(path.glob("*.nii.gz"))
            print(f"  {path.relative_to(output_dir)}: {len(files)} files")
        else:
            print(f"  {path.relative_to(output_dir)}: NOT FOUND")

    # Conta i file totali
    img_dir = output_dir / "RawData" / "Training" / "img"
    label_dir = output_dir / "RawData" / "Training" / "label"

    if img_dir.exists() and label_dir.exists():
        img_files = sorted(img_dir.glob("*.nii.gz"))
        label_files = sorted(label_dir.glob("*.nii.gz"))

        print(f"\n{'=' * 60}")
        print(f" SETUP COMPLETE!")
        print(f"{'=' * 60}")
        print(f"Images: {len(img_files)}")
        print(f"Labels: {len(label_files)}")
        print(f"\nSample files:")
        for img in img_files[:3]:
            print(f"  - {img.name}")
        print(f"  ... and {len(img_files) - 3} more")
        return True
    else:
        print("\n Setup incomplete - check directory structure")
        return False


def preprocess_synapse(random_seed=None, train_ratio=0.6):
    """
    Preprocessing del dataset Synapse con MAPPING CORRETTO delle classi
    """
    raw_data_dir = os.path.join("dataset", "RawData", "RawData", "Training", "img")
    label_data_dir = os.path.join("dataset", "RawData", "RawData", "Training", "label")
    output_dir = os.path.join("dataset", "project_transunet")

    train_out_dir = os.path.join(output_dir, "train_npz")
    test_out_dir = os.path.join(output_dir, "validation_vol_h5")

    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    # Lista immagini
    image_list = sorted(glob.glob(os.path.join(raw_data_dir, "*.nii.gz")))

    if len(image_list) == 0:
        print(f" No files found in {raw_data_dir}")
        return

    print(f"✓ Found {len(image_list)} volumes")

    # ========== MAPPING CORRETTO ==========
    # Synapse original -> TransUNet standard
    LABEL_MAPPING = {
        0: 0,  # Background
        1: 7,  # Spleen
        2: 4,  # Right Kidney
        3: 3,  # Left Kidney
        4: 2,  # Gallbladder
        6: 5,  # Liver
        7: 8,  # Stomach
        8: 1,  # Aorta
        11: 6,  # Pancreas
    }

    if random_seed is None:
        # Split ufficiale del paper
        train_ids = [5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40]
        train_cases = [f"img{i:04d}" for i in train_ids]
        all_cases = [os.path.basename(p).replace(".nii.gz", "") for p in image_list]
        test_cases = [c for c in all_cases if c not in train_cases]

    else:
        random.seed(random_seed)
        all_cases = [os.path.basename(p).replace(".nii.gz", "") for p in image_list]
        random.shuffle(all_cases)
        n_train = int(len(all_cases) * train_ratio)
        train_cases = all_cases[:n_train]
        test_cases = all_cases[n_train:]

    train_slice_count = 0
    val_volume_count = 0

    for img_path in tqdm(image_list, desc="Processing volumes"):
        case_name = os.path.basename(img_path).replace(".nii.gz", "")
        case_num = case_name.replace("img", "")
        label_path = os.path.join(label_data_dir, f"label{case_num}.nii.gz")

        if not os.path.exists(label_path):
            print(f"\n  Warning: Label not found for {case_name}")
            continue

        # Carica volumi
        image = nib.load(img_path).get_fdata()
        label_raw = nib.load(label_path).get_fdata()

        # ========== PREPROCESSING IMMAGINE ==========
        image = np.clip(image, -125, 275)
        image = (image + 125) / 400.0  # Normalizza [0, 1]

        # ========== RIMAPPATURA LABEL ==========
        label = np.zeros_like(label_raw, dtype=np.uint8)

        print(f"\n[{case_name}] Original label values: {np.unique(label_raw)}")

        for original_class, target_class in LABEL_MAPPING.items():
            mask = (label_raw == original_class)
            label[mask] = target_class

        print(f"[{case_name}] Remapped label values: {np.unique(label)}")

        # Verifica mapping
        unique_labels = np.unique(label)
        if unique_labels.max() > 8:
            print(f" ERROR {case_name}: Labels still > 8: {unique_labels}")
            continue

        # Verifica che Liver (5) sia presente (dovrebbe esserci nella maggior parte dei casi)
        if 5 not in unique_labels:
            print(f"  WARNING {case_name}: Liver (class 5) not found after remapping!")

        # Count voxels per class
        for cls in range(9):
            count = (label == cls).sum()
            if count > 0 and cls > 0:  # Skip background
                organ_names = ["", "Aorta", "Gallbladder", "L.Kidney",
                               "R.Kidney", "Liver", "Pancreas", "Spleen", "Stomach"]
                print(f"  Class {cls} ({organ_names[cls]}): {count:,} voxels")

        # Determina train o validation
        is_train = case_name in train_cases

        if is_train:
            # Salva slice 2D
            for slice_idx in range(image.shape[2]):
                img_slice = image[:, :, slice_idx]
                lab_slice = label[:, :, slice_idx]

                slice_name = f"{case_name}_slice{slice_idx:03d}.npz"
                np.savez(
                    os.path.join(train_out_dir, slice_name),
                    image=img_slice.astype(np.float32),
                    label=lab_slice.astype(np.uint8)
                )
                train_slice_count += 1
        else:
            # Salva volume 3D
            image_vol = image.transpose(2, 0, 1)
            label_vol = label.transpose(2, 0, 1)

            vol_name = f"{case_name}.npy.h5"
            #print(f"Saving H5 {vol_name} - Range: [{image_vol.min()}, {image_vol.max()}]")
            with h5py.File(os.path.join(test_out_dir, vol_name), 'w') as f:
                f.create_dataset("image", data=image_vol.astype(np.float32), compression="gzip")
                f.create_dataset("label", data=label_vol.astype(np.uint8), compression="gzip")

            val_volume_count += 1

    print(f"PREPROCESSING COMPLETE!")
    print(f"Training slices: {train_slice_count}")
    print(f"Validation volumes: {val_volume_count}")
    print(f"\nOutput:")
    print(f"  Train: {train_out_dir}")
    print(f"  Validation: {test_out_dir}")


# def debug_disk_data():
#     train_dir = "dataset/project_transunet/train_npz"
#     val_dir = "dataset/project_transunet/validation_vol_h5"
#
#     print("=" * 50)
#     print("DEBUG DATI PROCESSATI SU DISCO")
#     print("=" * 50)
#
#     # 1. Controllo Campione Training (.npz)
#     train_files = list(Path(train_dir).glob("*.npz"))
#     if train_files:
#         sample_train = train_files[0]
#         data = np.load(sample_train)
#         img, lab = data['image'], data['label']
#         print(f"\n[TRAINING SLICE] {sample_train.name}")
#         print(f"  Shape: {img.shape}")
#         print(f"  Range Pixel: [{img.min():.4f}, {img.max():.4f}]")
#         print(f"  Classi Label: {np.unique(lab)}")
#
#         if img.max() > 1.0 or img.min() < 0.0:
#             print("  (!) ATTENZIONE: I dati di training non sono nel range [0, 1]!")
#     else:
#         print("\n[!] Nessun file .npz trovato in training.")
#
#     # 2. Controllo Campione Validazione (.h5)
#     val_files = list(Path(val_dir).glob("*.h5"))
#     if val_files:
#         sample_val = val_files[0]
#         with h5py.File(sample_val, 'r') as f:
#             img = f['image'][:]
#             lab = f['label'][:]
#         print(f"\n[VALIDATION VOLUME] {sample_val.name}")
#         print(f"  Shape: {img.shape} (Z, H, W)")
#         print(f"  Range Pixel: [{img.min():.4f}, {img.max():.4f}]")
#         print(f"  Classi Label: {np.unique(lab)}")
#
#         if img.min() < 0 or img.max() > 1:
#             print("  (!) ERRORE CRITICO: Il volume di validazione ha range HU grezzo!")
#             print("      Il training vede [0,1], la validazione vede HU. Il Dice sarà bassissimo.")
#     else:
#         print("\n[!] Nessun file .h5 trovato in validazione.")
#
#     # 3. Verifica Coerenza Mapping
#     # Se il fegato è la classe 5 nel training, deve esserlo anche nel test
#     print("\n" + "=" * 50)

def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        """
        Calcola Dice e HD95 assicurandosi che le maschere non siano vuote.
        """
        # Convertiamo in binario (0 sfondo, 1 qualsiasi organo)
        pred = (pred > 0).astype(float)
        gt = (gt > 0).astype(float)

        # Caso 1: Entrambi hanno l'organo (Situazione standard)
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, hd95

        # Caso 2: Il modello trova un organo che non esiste (Falso Positivo)
        # Il Dice è 0, la distanza HD95 non è definibile (spesso si mette un valore alto di default)
        if pred.sum() > 0 and gt.sum() == 0:
            return 0.0, 100.0  # 100mm è una penalità standard

        # Caso 3: L'organo esiste ma il modello non vede nulla (Falso Negativo)
        if pred.sum() == 0 and gt.sum() > 0:
            return 0.0, 100.0

        # Caso 4: Entrambi vuoti (Il modello ha predetto correttamente lo sfondo)
        return 1.0, 0.0


def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, test_mode=False ,z_spacing=1):
    """
        Esegue l'inferenza slice-by-slice su un volume 3D usando un modello 2D
        e calcola le metriche per ogni classe.
        Replicando il protocollo di Zhou et al. e Yu et al.

        Args:
            image (Tensor): volume di input [1, Z, H, W]
            label (Tensor): ground truth [1, Z, H, W]
            net (nn.Module): modello di segmentazione 2D
            classes (int): numero totale di classi (incluso background)
            patch_size (tuple): dimensione di input richiesta dal modello
            test_save_path (str): path per il salvataggio dei risultati (.nii.gz)
            case (str): identificativo del paziente
            z_spacing (float): spacing assiale (mm)
        """
    # Spostiamo i dati su CPU e li convertiamo in NumPy
    # Il loader ti dà [1, 147, 512, 512] -> B, C, H, W
    # Ma per noi C sono le fette (Z).
    # Rimuoviamo la dimensione batch: [1, Z, H, W] -> [Z, H, W]
    # Questo facilita il loop slice-by-slice

    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    # ================== BLOCCO DEBUG INPUT ==================
    print(f"\n{'=' * 40}")
    print(f"DEBUG DATA RANGE - Caso: {case}")
    print(f"{'=' * 40}")

    # Statistiche Immagine
    img_min, img_max = image.min(), image.max()
    print(f"INPUT IMAGE:")
    print(f"  -> Range: [{img_min:.2f}, {img_max:.2f}]")
    print(f"  -> Media: {image.mean():.2f} | Std: {image.std():.2f}")

    # Diagnosi Automatica
    if img_min < -500:
        print("\n[!] DIAGNOSI: Valori in Hounsfield Units (HU) rilevati.")
        print("    Il modello si aspetta dati clippati e normalizzati (es. [0, 1]).")
        print("    AZIONE: Applica np.clip(image, -125, 275) e poi normalizza.")
    elif img_min >= 0 and img_max <= 1:
        print("\n[v] DIAGNOSI: Dati nel range [0, 1]. Corretto.")
    else:
        print("\n[?] DIAGNOSI: Range inusuale. Verifica il pre-processing del training.")

    # Verifica Ground Truth
    unique_labels = np.unique(label)
    print(f"\nGROUND TRUTH:")
    print(f"  -> Classi presenti: {unique_labels}")
    if max(unique_labels) > classes:
        print(f"  [!] ERRORE: Trovate label ({max(unique_labels)}) superiori al num_classes configurato!")
    print(f"{'=' * 40}\n")

    # Se dopo lo squeeze image è ancora 4D (raro ma possibile), forziamo:
    if image.ndim == 4:
        image = image[0]

    # Inizializziamo il volume delle predizioni
    # Stessa shape della label per facilitare il confronto voxel-wise

    prediction = np.zeros_like(label)

    #Definizione delle trasformazioni (torchvision v2)

    resize_input = v2.Resize(
        size=patch_size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    )

    resize_output = v2.Resize(
        size=label.shape[1:],  # dimensioni originali (H, W)
        interpolation=InterpolationMode.NEAREST
    )

    net.eval()

    # Recuperiamo il device del modello
    device = next(net.parameters()).device

    with torch.inference_mode():
        # Loop su ogni slice assiale
        for z in range(image.shape[0]):
            # Estrazione della singola fetta 2D
            slice_2d = image[z, :, :]
            x, y = slice_2d.shape[0], slice_2d.shape[1]

            # Conversione in tensore PyTorch [H, W] -> [1, H, W]
            slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0)

            # Resize dell'input SOLO se necessario
            if x != patch_size[0] or y != patch_size[1]:
                slice_resized = resize_input(slice_tensor)
            else:
                slice_resized = slice_tensor

            # Aggiungi dimensione batch: [1, H, W] -> [1, 1, H, W]
            input_tensor = slice_resized.unsqueeze(0).to(device)

            # Inferenza
            outputs = net(input_tensor)

            # Softmax + Argmax (come nel paper originale)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)

            # Resize della predizione alle dimensioni originali SOLO se necessario
            if x != patch_size[0] or y != patch_size[1]:
                out_resized = resize_output(out.unsqueeze(0)).squeeze(0)
            else:
                out_resized = out

            # Salviamo la slice nel volume finale
            prediction[z] = out_resized.cpu().numpy()

            # Libera memoria GPU
            del input_tensor, outputs, out, out_resized

        # Calcolo metriche per ogni classe (si salta la classe 0 = background)

    metric_list = []
    # Si salta la classe 0 (background)
    for cls in range(1, classes):
        metric_list.append(
            calculate_metric_percase(
                prediction == cls,
                label == cls
            )
        )

    #Salvataggio opzionale dei risultati in formato NIfTI
    #     SimpleITK usa l'ordine:
    #     - array:  [Z, Y, X]
    #     - spacing:(X, Y, Z)

    if test_save_path is not None and test_mode:
        # Crea directory se non esiste
        os.makedirs(test_save_path, exist_ok=True)

        # # ======== VERIFICA VALORI PRIMA DI SALVARE ========
        # print(f"\n[DEBUG {case}] Verifica valori pre-salvataggio:")
        # print(f"  Prediction shape: {prediction.shape}")
        # print(f"  Prediction dtype: {prediction.dtype}")
        # print(f"  Prediction unique values: {np.unique(prediction)}")
        # print(f"  Prediction min/max: {prediction.min():.2f} / {prediction.max():.2f}")

        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        # Imposta spacing (in mm)
        img_itk.SetSpacing((1.0, 1.0, z_spacing))
        prd_itk.SetSpacing((1.0, 1.0, z_spacing))
        lab_itk.SetSpacing((1.0, 1.0, z_spacing))

        # Salva file NIfTI
        sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_pred.nii.gz")
        sitk.WriteImage(img_itk, f"{test_save_path}/{case}_img.nii.gz")
        sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_gt.nii.gz")

    return metric_list


def mock_test():
     # 1. Configurazione Parametri
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     num_classes = 9
     img_size = 224
     num_slices = 10  # Volume piccolo per il test

     print(f"--- Test su Device: {device} ---")
#
     # 2. Creazione Modello Mock (o il tuo PT_TransUNet reale)
     # Se vuoi testare il TUO modello reale:
     model = PT_TransUNet(img_size=img_size).to(device)

     # Per il test rapido usiamo una classe minima che simula l'errore
#     class MockNet(torch.nn.Module):
#         def __init__(self, n_cls):
#             super().__init__()
#             self.conv = torch.nn.Conv2d(3, n_cls, kernel_size=1)
#
#         def forward(self, x):
#             # Simuliamo la logica incriminata del PTResnet
#             if x.dim() == 3: x = x.unsqueeze(0)
#             if x.shape[1] == 1:
#                 print(f"   [Debug] Input shape prima di expand: {x.shape}")
#                 x = x.expand(-1, 3, -1, -1)
#                 print(f"   [Debug] Input shape dopo expand: {x.shape}")
#             return self.conv(x)
#
#     model = MockNet(num_classes).to(device)
     model.eval()
#
#     # 3. Creazione Dati Sintetici (Simuliamo un volume Synapse)
#     # Shape attesa da test_single_volume: [1, Z, H, W]
     dummy_image = torch.randn(1, num_slices, 512, 512).to(device)
     dummy_label = torch.randint(0, num_classes, (1, num_slices, 512, 512)).to(device)
#
     print(f"Input Image Shape: {dummy_image.shape}")
     print(f"Input Label Shape: {dummy_label.shape}")
#
#     # 4. Esecuzione Funzione
     try:
         metrics = test_single_volume(
             image=dummy_image,
             label=dummy_label,
             net=model,
             classes=num_classes,
             patch_size=[img_size, img_size],
             test_mode=False
         )
#
         print("\n--- TEST COMPLETATO CON SUCCESSO ---")
         print(f"Numero di organi valutati: {len(metrics)}")
         print(f"Esempio Metrica Organo 1 (Dice, HD95): {metrics[0]}")
#
     except Exception as e:
         print("\n--- TEST FALLITO ---")
         print(f"Errore riscontrato: {e}")
         import traceback
         traceback.print_exc()


def inspect_h5_file(filepath):
    print("=" * 50)
    print(f"ISPEZIONE FILE: {filepath}")
    print("=" * 50)

    try:
        with h5py.File(filepath, 'r') as f:
            #  Vediamo quali dataset ci sono (dovrebbero essere 'image' e 'label')
            keys = list(f.keys())
            print(f"Chiavi trovate: {keys}\n")

            for key in keys:
                data = f[key]
                print(f"--- DATASET: {key} ---")
                print(f"  Shape: {data.shape}  (Z, H, W)")
                print(f"  Dtype: {data.dtype}")

                # Calcoliamo alcune statistiche veloci
                # Carichiamo i dati in memoria per le statistiche
                arr = data[:]
                print(f"  Valore Min: {np.min(arr):.4f}")
                print(f"  Valore Max: {np.max(arr):.4f}")
                print(f"  Valori unici: {np.unique(arr) if key == 'label' else 'N/A'}")
                print("-" * 30)

    except Exception as e:
        print(f"Errore durante l'apertura del file: {e}")


def inspect_label_distribution(h5_file_path):
    """Verifica quali classi sono presenti in un file di validation"""
    import h5py

    with h5py.File(h5_file_path, 'r') as f:
        label = f['label'][:]

    unique, counts = np.unique(label, return_counts=True)

    organ_names = {
        0: "Background",
        1: "Aorta",
        2: "Gallbladder",
        3: "Left Kidney",
        4: "Right Kidney",
        5: "Liver",
        6: "Pancreas",
        7: "Spleen",
        8: "Stomach"
    }

    print(f"\n{'=' * 70}")
    print(f"FILE: {h5_file_path}")
    print(f"{'=' * 70}")
    print(f"{'Class':<5} | {'Organ':<15} | {'Voxels':<12} | {'Percentage':<10}")
    print("-" * 70)

    total_voxels = label.size
    for cls, count in zip(unique, counts):
        organ = organ_names.get(int(cls), "UNKNOWN")
        percentage = (count / total_voxels) * 100
        print(f"{int(cls):<5} | {organ:<15} | {count:<12,} | {percentage:>6.2f}%")

    print("=" * 70 + "\n")

    return unique




if __name__ == "__main__":
    inspect_h5_file("dataset/project_transunet/validation_vol_h5/img0001.npy.h5")
    #mock_test()
    #debug_disk_data()
    #preprocess_synapse()
    # Test su tutti i file validation
    # validation_dir = "dataset/project_transunet/validation_vol_h5"
    # for h5_file in sorted(Path(validation_dir).glob("*.h5")):
    #     classes_present = inspect_label_distribution(str(h5_file))
    # #
    #     # VERIFICA: Liver (classe 5) dovrebbe essere presente in TUTTI i file
    #     if 5 not in classes_present:
    #         print(f"️  WARNING: Liver (class 5) NOT FOUND in {h5_file.name}!")
    # #
    #     # VERIFICA: Right Kidney (classe 4) dovrebbe essere presente
    #     if 4 not in classes_present:
    #         print(f"️  WARNING: Right Kidney (class 4) NOT FOUND in {h5_file.name}!")
