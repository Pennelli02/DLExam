import argparse
from types import SimpleNamespace

import numpy as np
import torch
import logging
import os

import yaml
from scipy.ndimage import zoom
from torch import nn

from src.transUNet import CheckpointNet
from utils import test_single_volume, test_single_volumeSy
from train import load_checkpoint

LOG = logging.getLogger(__name__)

# Nomi degli organi secondo il paper TransUNet
ORGAN_NAMES = [
    "Aorta", "Gallbladder", "Left Kidney", "Right Kidney",
    "Liver", "Pancreas", "Spleen", "Stomach"
]


def inference(net, valid_loader, opts, resize_type="scipy"):
    """
    Testing volumetrico con metriche dettagliate per-organo.
    Replica esatta del protocollo di validazione ma su test set.

    Args:
        net: modello di segmentazione
        valid_loader: DataLoader del test set
        opts: configurazione (n_classes, save_dir, ...)
        resize_type: "scipy" per zoom scipy (paper originale)
                     "v2"    per torchvision v2.Resize
    """
    net.eval()
    performance_buffer = []

    # Seleziona la funzione di inferenza in base al tipo di resize
    if resize_type == "scipy":
        inference_fn = test_single_volumeSy
        LOG.info("Resize: scipy.ndimage.zoom")
    elif resize_type == "v2":
        inference_fn = test_single_volume
        LOG.info("  Resize: torchvision v2.Resize")
    else:
        raise ValueError(f"resize_type deve essere 'scipy' o 'v2', ricevuto: '{resize_type}'")

    LOG.info(f" Testing volumetrico avviato (resize_type='{resize_type}')...")

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valid_loader):
            image = sampled_batch['image']
            label = sampled_batch['label']
            case_name = sampled_batch['case_name'][0]

            case_metrics = inference_fn(
                image, label, net,
                classes=opts.n_classes,
                test_mode=True,
                test_save_path=opts.save_dir,
                case=case_name
            )
            performance_buffer.append(case_metrics)
            LOG.info(f" {case_name} processato")

    # [Num_Pazienti, Num_Organi, 2]
    performance_buffer = np.array(performance_buffer)

    # Media per organo
    mean_per_organ = np.mean(performance_buffer, axis=0)

    # Statistiche globali
    avg_dice = np.mean(mean_per_organ[:, 0])
    avg_hd95 = np.mean(mean_per_organ[:, 1])

    # Log dettagliato per-organo
    LOG.info("\n" + "=" * 60)
    LOG.info(f" RISULTATI TESTING (resize_type='{resize_type}')")
    LOG.info("=" * 60)

    for i, organ in enumerate(ORGAN_NAMES):
        dice = mean_per_organ[i, 0]
        hd95 = mean_per_organ[i, 1]
        LOG.info(f"{organ:15s} -> Dice: {dice:.4f} | HD95: {hd95:.2f} mm")

    LOG.info(f"{'MEDIA TOTALE':15s} -> Dice: {avg_dice:.4f} | HD95: {avg_hd95:.2f} mm")

    return avg_dice, avg_hd95, mean_per_organ


if __name__ == "__main__":
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)-8s %(message)s',
            datefmt='%H:%M:%S'
        )

        parser = argparse.ArgumentParser()
        parser.add_argument("config", help='YAML Configuration file')
        parser.add_argument(
            "--checkpoint",
            default=None,
            help="Path esplicito al checkpoint (opzionale, altrimenti prende l'ultimo)"
        )
        args = parser.parse_args()

        opts = yaml.load(open(args.config), Loader=yaml.Loader)
        opts = SimpleNamespace(**opts)
        opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Costruzione modello
        model = CheckpointNet(npz_path="PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")

        # Caricamento corretto tramite load_checkpoint
        # optimizer e scheduler sono None perché siamo in inference, non training
        checkpoint = load_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            opts=opts,
            checkpoint_path=opts.testing_chp  # None = prende l'ultimo automaticamente
        )

        if checkpoint is None:
            LOG.error("Nessun checkpoint trovato. Aborting.")
            exit(1)

        LOG.info(f"Modello caricato dall'epoca {checkpoint['epoch']} "
                 f"(step {checkpoint.get('global_step', 'N/A')})")

        model.eval()

        # Dataset e loader
        from dataset import SynapseDataset
        from torch.utils.data import DataLoader

        test_dataset = SynapseDataset(
            opts=opts,
            data_dir=opts.testing_dir,
            split="test",
            transform=None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=opts.num_workers,
            pin_memory=True  # velocizza trasferimento CPU→GPU
        )

        # Doppio run scipy vs v2
        all_results = {}
        for resize_type in ["scipy", "v2"]:
            LOG.info(f"\n{'=' * 60}")
            LOG.info(f" Run con resize_type='{resize_type}'")
            LOG.info(f"{'=' * 60}")

            avg_dice, avg_hd95, per_organ = inference(
                net=model,
                valid_loader=test_loader,
                opts=opts,
                resize_type=resize_type
            )
            all_results[resize_type] = (avg_dice, avg_hd95)

        # Confronto finale tra i due metodi
        LOG.info("\n" + "=" * 60)
        LOG.info(" CONFRONTO FINALE scipy vs v2")
        LOG.info("=" * 60)
        for resize_type, (dice, hd95) in all_results.items():
            LOG.info(f"{resize_type:6s} -> Dice: {dice:.4f} | HD95: {hd95:.2f} mm")