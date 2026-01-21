import os
from types import SimpleNamespace
import argparse
import logging
import glob
from ipdb import launch_ipdb_on_exception
import yaml
from rich.logging import RichHandler
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import src.dataset
from src.transUNet import PT_TransUNet, NPT_TransUNet
from src.dataset import SynapseDataset
from src.utils import test_single_volume


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()

def save_checkpoint(model, optimizer, epoch, loss, opts):
    fname = os.path.join(opts.checkpoint_dir,f'e_{epoch:05d}.chp')
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch, loss=loss)
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')


def load_checkpoint(model, optimizer, opts, epoch=None, checkpoint_path=None):
    """
    Carica automaticamente l'ultimo checkpoint o uno specifico.

    Args:
        model: Il modello già istanziato.
        optimizer: L'optimizer già istanziato sui parametri del modello.
        opts: Namespace con opts.checkpoint_dir e opts.device.
        epoch: (Opzionale) Epoca specifica da caricare.
        checkpoint_path: (Opzionale) Path diretto al file .chp.
    """

    # LOGICA DI RICERCA AUTOMATICA DEL FILE
    if checkpoint_path is not None:
        fname = checkpoint_path
    elif epoch is not None:
        fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    else:
        # Cerca tutti i file .chp nella cartella
        chk_files = glob.glob(os.path.join(opts.checkpoint_dir, "*.chp"))
        if not chk_files:
            LOG.warning(" Nessun checkpoint trovato. Il training partirà da zero.")
            return None

        # Prende l'ultimo basandosi sul nome del file (e_00001.chp, e_00002.chp...)
        fname = max(chk_files, key=os.path.getctime)  # Il più recente per data di creazione

    # CARICAMENTO EFFETTIVO
    LOG.info(f"Caricamento checkpoint: {fname}")

    # Carichiamo sempre su CPU inizialmente per evitare errori di memoria GPU
    checkpoint = torch.load(fname, map_location='cpu')

    # Ripristiniamo i pesi del modello
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ripristiniamo lo stato dell'optimizer (es. momentum, medie Adam)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # SPOSTAMENTO STATO OPTIMIZER SU GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(opts.device)

    # SPOSTAMENTO MODELLO SUL DEVICE
    model.to(opts.device)

    loaded_epoch = checkpoint['epoch']
    LOG.info(f" Checkpoint caricato correttamente! (Epoca: {loaded_epoch})")

    return checkpoint


class DiceLoss(nn.Module):
    """
    Dice Loss multi-classe ottimizzata.
    Basata sul paper TransUNet ma con implementazione vettorizzata.
    """

    def __init__(self, n_classes: int = 9, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, softmax: bool = True) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] - output del modello
            target: [B, H, W] - ground truth (valori 0-8)
            softmax: applica softmax ai logits

        Returns:
            loss: Dice loss mediato su tutte le classi
        """
        if softmax:
            probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        else:
            probs = logits

        # One-hot encoding del target: [B, H, W] -> [B, C, H, W]
        target_one_hot = F.one_hot(target.long(), num_classes=self.n_classes)  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Flatten spaziale per ogni classe
        probs_flat = probs.contiguous().view(probs.size(0), self.n_classes, -1)  # [B, C, H*W]
        target_flat = target_one_hot.contiguous().view(target_one_hot.size(0), self.n_classes, -1)  # [B, C, H*W]

        # Calcolo Dice per classe (vettorizzato)
        intersection = (probs_flat * target_flat).sum(dim=2)  # [B, C]
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]

        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        dice_loss = 1 - dice_per_class  # [B, C]

        # Media su batch e classi
        return dice_loss.mean()

# LOGICA DI VALIDAZIONE (Chiamata nel Training Loop)
def validate_model(model, valid_loader, opts):
    """
    Validazione volumetrica con metriche dettagliate per-organo
    """
    model.eval()
    performance_buffer = []

    # Nomi degli organi secondo il paper TransUNet
    organ_names = [
        "Aorta", "Gallbladder", "Left Kidney", "Right Kidney",
        "Liver", "Pancreas", "Spleen", "Stomach"
    ]

    LOG.info(" Validazione volumetrica avviata...")

    with torch.no_grad():  # per risparmiare memoria
        for i_batch, sampled_batch in enumerate(valid_loader):
            image = sampled_batch['image']
            label = sampled_batch['label']
            case_name = sampled_batch['case_name'][0]

            case_metrics = test_single_volume(
                image, label, model,
                classes=opts.n_classes
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
    LOG.info(" RISULTATI VALIDAZIONE")
    LOG.info("=" * 60)

    for i, organ in enumerate(organ_names):
        dice = mean_per_organ[i, 0]
        hd95 = mean_per_organ[i, 1]
        LOG.info(f"{organ:15s} -> Dice: {dice:.4f} | HD95: {hd95:.2f} mm")

    LOG.info(f"{'MEDIA TOTALE':15s} -> Dice: {avg_dice:.4f} | HD95: {avg_hd95:.2f} mm")

    return avg_dice, avg_hd95, mean_per_organ


def train_loop(model, train, valid, opts):
    import tensorflow as tf
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_name}/train')
    val_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_name}/validation')

    # Nomi organi per TensorBoard
    organ_names = [
        "Aorta", "Gallbladder", "Left_Kidney", "Right_Kidney",
        "Liver", "Pancreas", "Spleen", "Stomach"
    ]

    # Optimizer
    if opts.type_tr == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opts.lr,
            momentum=opts.momentum,
            weight_decay=opts.weight_decay,
        )
    #else:
        #da valutare se testare con altri tipi di optimizer come adam
    model.train()

    # Loss function (Dice + CE secondo paper)
    dice_loss_fn = DiceLoss(n_classes=opts.n_classes)

    # Resume da checkpoint
    start_epoch = 1
    if opts.resume:
        checkpoint = load_checkpoint(model, optimizer, opts, checkpoint_path=opts.checkpoint_dir)
        if checkpoint is not None:
            start_epoch = checkpoint['epoch'] + 1

    step = 0

    LOG.info(f" Training da epoca {start_epoch} a {opts.n_epoch_sy}")

    for epoch in range(start_epoch, opts.n_epoch_sy + 1):
        model.train()
        epoch_losses = []
        epoch_dice_losses = []
        epoch_ce_losses = []

        for batch_i, batch in enumerate(train, 1):
            images = batch['image'].to(opts.device)  # [B, 1, H, W]
            labels = batch['label'].to(opts.device)  # [B, H, W]

            optimizer.zero_grad()

            # Forward
            outputs = model(images)  # [B, 9, 224, 224]

            # Loss: Dice + CE (come nel paper)
            dice_loss = dice_loss_fn(outputs, labels, softmax=True)
            ce_loss = F.cross_entropy(outputs, labels.long())
            loss = 0.5 * dice_loss + 0.5 * ce_loss

            # Backward
            loss.backward()
            optimizer.step()

            # Metriche
            epoch_losses.append(loss.item())
            epoch_dice_losses.append(dice_loss.item())
            epoch_ce_losses.append(ce_loss.item())

            # Logging
            if batch_i % opts.log_every == 0:
                train_loss = np.mean(epoch_losses[-opts.batch_window:])
                train_dice_loss = np.mean(epoch_dice_losses[-opts.batch_window:])
                train_ce_loss = np.mean(epoch_ce_losses[-opts.batch_window:])


                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'loss={train_loss:.4f} (dice={train_dice_loss:.4f}, ce={train_ce_loss:.4f}) | '
                LOG.info(msg)

                # TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar('total_loss', train_loss, step=step)
                    tf.summary.scalar('dice_loss', train_dice_loss, step=step)
                    tf.summary.scalar('ce_loss', train_ce_loss, step=step)

                step += 1

        # --- FASE DI VALIDAZIONE (Ogni fine 2 epoche) ---
        # Usiamo la validazione volumetrica per monitorare i progressi "reali"

            if epoch % opts.validation_frequency == 0:
                val_dice, val_hd95, per_organ_metrics = validate_model(model, valid, opts)

                with val_writer.as_default():
                    # Metriche globali
                    tf.summary.scalar('val_dice_avg', val_dice, step=epoch)
                    tf.summary.scalar('val_hd95_avg', val_hd95, step=epoch)

                    # Metriche per-organo
                    for i, organ_name in enumerate(organ_names):
                        tf.summary.scalar(f'val_dice/{organ_name}',
                                              per_organ_metrics[i, 0], step=epoch)
                        tf.summary.scalar(f'val_hd95/{organ_name}',
                                              per_organ_metrics[i, 1], step=epoch)

         # Checkpoint periodico
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), opts)


def main(opts):
    from visualizer import visualize
    input_data = torch.randn(opts.batch, 1, opts.image_size, opts.image_size)
    if opts.pre_trained:
        model = PT_TransUNet()
    else:
        model = NPT_TransUNet()

    LOG.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    #CL visualization
    visualize(model, opts.model_name, input_data)
    model = model.to(opts.device)

    #creiamo i dataset di training e testing
    if opts.dataset_type == "Synapse":
        #prendiamo le immagini di training
        train_dataset = SynapseDataset(opts, opts.train_dir, "train", src.dataset.get_train_transform(opts))
        print(train_dataset.__len__())
        train_loader = DataLoader(train_dataset, batch_size=opts.batch, shuffle=True)

        #prendiamo le immagini di validation
        val_dataset = SynapseDataset(opts, opts.validation_dir, "validation", None)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    elif opts.dataset_type == "ACDC":
        #TODO
        print("lavori in corso")

    LOG.info(f"Train batches: {len(train_loader)}, Valid batches: {len(val_loader)}")

    #training
    train_loop(model, train_loader, val_loader, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Crea checkpoint directory
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    with launch_ipdb_on_exception():
        main(opts)