import random
import zipfile
from pathlib import Path
import os
import glob
import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm
import synapseclient
import synapseutils

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
    # Percorsi corretti dopo estrazione
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
        print("Did you extract RawData.zip?")
        return

    print(f" Found {len(image_list)} volumes")

    if random_seed is None:
        # Split secondo TransUNet paper (18 training / 12 validation)
        # Questi sono gli ID esatti usati nel paper
        train_ids = [5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40]
        train_cases = [f"img{i:04d}" for i in train_ids]

        all_cases = [os.path.basename(p).replace(".nii.gz", "") for p in image_list]
        test_cases = [c for c in all_cases if c not in train_cases]

        print(f"Training: {len(train_cases)} cases")
        print(f"Validation: {len(test_cases)} cases\n")
    elif random_seed is not None:
        random.seed(random_seed)
        all_cases = [os.path.basename(p).replace(".nii.gz", "") for p in image_list]
        # Shuffle e split
        random.shuffle(all_cases)
        n_train = int(len(all_cases) * train_ratio)

        train_cases = all_cases[:n_train]
        test_cases = all_cases[n_train:]

        print(f"Training: {len(train_cases)} cases - {sorted(train_cases)}")
        print(f"Testing: {len(test_cases)} cases - {sorted(test_cases)}\n")

    train_slice_count = 0
    val_volume_count = 0

    for img_path in tqdm(image_list, desc="Processing volumes"):
        case_name = os.path.basename(img_path).replace(".nii.gz", "")

        # Costruisci il path della label
        case_num = case_name.replace("img", "")
        label_path = os.path.join(label_data_dir, f"label{case_num}.nii.gz")

        if not os.path.exists(label_path):
            print(f"\n  Warning: Label not found for {case_name}")
            continue

        # Carica i volumi
        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Preprocessing (TransUNet paper)
        image = np.clip(image, -125, 275)
        image = (image + 125) / 400.0  # Normalizza a [0, 1]

        # Determina train o validation
        is_train = case_name in train_cases

        if is_train:
            # Salva slice 2D per training
            for slice_idx in range(image.shape[2]):
                img_slice = image[:, :, slice_idx]
                lab_slice = label[:, :, slice_idx]

                #  salta slice completamente vuote
                # if np.sum(lab_slice) == 0:
                #     continue

                slice_name = f"{case_name}_slice{slice_idx:03d}.npz"
                np.savez(
                    os.path.join(train_out_dir, slice_name),
                    image=img_slice.astype(np.float32),
                    label=lab_slice.astype(np.uint8)
                )
                train_slice_count += 1
        else:
            # Salva volume 3D per test
            image_vol = image.transpose(2, 0, 1)  # [Z, H, W]
            label_vol = label.transpose(2, 0, 1)

            vol_name = f"{case_name}.npy.h5"
            with h5py.File(os.path.join(test_out_dir, vol_name), 'w') as f:
                f.create_dataset("image", data=image_vol.astype(np.float32), compression="gzip")
                f.create_dataset("label", data=label_vol.astype(np.uint8), compression="gzip")

            val_volume_count += 1

    print(f"PREPROCESSING COMPLETE!\n")
    print(f"Training slices: {train_slice_count}")
    print(f" Validation  volumes: {val_volume_count}")
    print(f"\nOutput:")
    print(f"  Train: {train_out_dir}")
    print(f"  Validation:  {test_out_dir}")


if __name__ == "__main__":
    preprocess_synapse()

