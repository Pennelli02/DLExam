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
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from src.transUNet import NPT_TransUNet, CheckpointNet, PT_TransUNet
from src.dataset import SynapseDataset, get_train_transform
from src.utils import test_single_volume


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()

def save_checkpoint(model, optimizer, scheduler, epoch, loss, global_step ,opts):
    fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), # Salviamo lo stato dello scheduler
        'epoch': epoch,
        'loss': loss,
        'global_step': global_step
    }
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')


def load_checkpoint(model, optimizer, scheduler, opts, epoch=None, checkpoint_path=None):
    if checkpoint_path is not None:
        fname = checkpoint_path
    elif epoch is not None:
        fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    else:
        chk_files = glob.glob(os.path.join(opts.checkpoint_dir, "*.chp"))
        if not chk_files:
            LOG.warning(" Nessun checkpoint trovato.")
            return None
        chk_files.sort()
        fname = chk_files[-1]

    LOG.info(f" Caricamento checkpoint: {fname}")

    checkpoint = torch.load(fname, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Sposta stato optimizer su GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(opts.device)

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Lo scheduler NON ha bisogno di essere spostato su GPU

    model.to(opts.device)

    loaded_epoch = checkpoint['epoch']
    loaded_step = checkpoint.get('global_step', 0)
    LOG.info(f" Checkpoint caricato! (Epoca: {loaded_epoch})")

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
                classes=opts.n_classes, test_mode=True, test_save_path=opts.save_dir, case=case_name
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
    import time

    # -----------------------------
    # TensorBoard Writers
    # -----------------------------
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_name}/train')
    val_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_name}/validation')

    # Nomi degli organi per logging
    organ_names = [
        "Aorta", "Gallbladder", "Left_Kidney", "Right_Kidney",
        "Liver", "Pancreas", "Spleen", "Stomach"
    ]

    # -----------------------------
    # Optimizer e scheduler
    # -----------------------------
    if opts.type_tr == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opts.lr,
            momentum=opts.momentum,
            weight_decay=opts.weight_decay,
        )

    # Poly LR Scheduler
    max_iterations = len(train) * opts.n_epoch_sy
    poly_lr_lambda = lambda iteration: (1.0 - iteration / max_iterations) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    # -----------------------------
    # Loss function
    # -----------------------------
    dice_loss_fn = DiceLoss(n_classes=opts.n_classes)

    # -----------------------------
    # Mixed Precision (AMP)
    # -----------------------------
    scaler = GradScaler(device='cuda')

    # -----------------------------
    # Resume da checkpoint
    # -----------------------------
    start_epoch = 1
    global_step = 0

    if opts.resume:
        checkpoint = load_checkpoint(model, optimizer, scheduler, opts)
        if checkpoint is not None:
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint['epoch'] + 1

    step = global_step

    # -----------------------------
    # CUDNN benchmark
    # -----------------------------
    torch.backends.cudnn.benchmark = True

    LOG.info(f" Training da epoca {start_epoch} a {opts.n_epoch_sy} | Step: {step}")

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(start_epoch, opts.n_epoch_sy + 1):
        model.train()
        epoch_losses = []
        epoch_dice_losses = []
        epoch_ce_losses = []

        # Timing diagnostics
        data_load_times = []
        compute_times = []
        batch_end_time = time.time()

        for batch_i, batch in enumerate(train, 1):
            # ===== TIMING: Data Loading =====
            data_time = time.time() - batch_end_time
            data_load_times.append(data_time)

            compute_start = time.time()

            # ===== GPU Transfer =====
            images = batch['image'].to(opts.device, non_blocking=True)
            labels = batch['label'].to(opts.device, non_blocking=True)

            # Controllo sicurezza
            unique_labels = torch.unique(labels)
            if unique_labels.max() >= opts.n_classes:
                LOG.error(f" ERRORE: Label fuori range! Valori: {unique_labels.cpu().numpy()}")
                raise ValueError("Label fuori range nel dataset!")

            optimizer.zero_grad(set_to_none=True)

            # ===== Forward + Loss =====
            with autocast(device_type='cuda'):
                outputs = model(images)
                dice_loss = dice_loss_fn(outputs, labels, softmax=True)
                ce_loss = F.cross_entropy(outputs, labels.long())
                loss = 0.5 * dice_loss + 0.5 * ce_loss

            # ===== Backward =====
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ===== TIMING: Compute =====
            compute_time = time.time() - compute_start
            compute_times.append(compute_time)

            # Salvataggio metriche
            epoch_losses.append(loss.item())
            epoch_dice_losses.append(dice_loss.item())
            epoch_ce_losses.append(ce_loss.item())

            # ===== LOGGING CON TIMING =====
            if batch_i % opts.log_every == 0:
                train_loss = np.mean(epoch_losses[-opts.batch_window:])
                train_dice_loss = np.mean(epoch_dice_losses[-opts.batch_window:])
                train_ce_loss = np.mean(epoch_ce_losses[-opts.batch_window:])
                current_dice_score = 1 - train_dice_loss

                # Calcola efficienza GPU
                avg_data_time = np.mean(data_load_times[-opts.log_every:])
                avg_compute_time = np.mean(compute_times[-opts.log_every:])
                gpu_efficiency = avg_compute_time / (avg_data_time + avg_compute_time) * 100

                # Console logging
                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'Loss={train_loss:.4f} | Dice={current_dice_score:.4f} | CE={train_ce_loss:.4f} | '
                msg += f'Data={avg_data_time * 1000:.0f}ms | GPU={gpu_efficiency:.0f}%  |  LR={optimizer.param_groups[0]['lr']}'
                LOG.info(msg)

                # TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar('loss/total', train_loss, step=step)
                    tf.summary.scalar('loss/dice', train_dice_loss, step=step)
                    tf.summary.scalar('loss/ce', train_ce_loss, step=step)
                    tf.summary.scalar('metrics/dice_score', current_dice_score, step=step)

            # ===== Visualizzazione Immagini (ogni 500 step) =====
            if step % 500 == 0:
                with train_writer.as_default():
                    idx = 0
                    # Immagine normalizzata
                    img_vis = (images[idx] - images[idx].min()) / (images[idx].max() - images[idx].min() + 1e-8)
                    tf.summary.image('train/Image',
                                     img_vis.cpu().numpy().transpose(1, 2, 0)[np.newaxis, ...],
                                     step=step)

                    # Predizione
                    pred = torch.argmax(torch.softmax(outputs[idx], dim=0), dim=0).unsqueeze(0).float()
                    tf.summary.image('train/Prediction',
                                     (pred * 25).cpu().numpy().transpose(1, 2, 0)[np.newaxis, ...],
                                     step=step)

                    # Ground Truth
                    gt = labels[idx].unsqueeze(0).float()
                    tf.summary.image('train/GroundTruth',
                                     (gt * 25).cpu().numpy().transpose(1, 2, 0)[np.newaxis, ...],
                                     step=step)

            # Reset timer
            batch_end_time = time.time()
            step += 1

        # ===== VALIDAZIONE =====
        if epoch % opts.validation_frequency == 0:
            LOG.info(f" VALIDAZIONE Volumetrica")

            val_dice, val_hd95, per_organ_metrics = validate_model(model, valid, opts)


            # TensorBoard validation metrics
            with val_writer.as_default():
                tf.summary.scalar('metrics/dice_avg', val_dice, step=epoch)
                tf.summary.scalar('metrics/hd95_avg', val_hd95, step=epoch)

                # Metriche per organo
                for i, organ_name in enumerate(organ_names):
                    tf.summary.scalar(f'dice/{organ_name}', per_organ_metrics[i, 0], step=epoch)
                    tf.summary.scalar(f'hd95/{organ_name}', per_organ_metrics[i, 1], step=epoch)



            # Pulizia memoria GPU
            torch.cuda.empty_cache()

        # ===== CHECKPOINT PERIODICI =====
        if epoch % opts.save_every == 0 or epoch == opts.n_epoch_sy:
            save_checkpoint(model, optimizer, scheduler, epoch,
                                      loss.item(), step, opts)

    LOG.info(f" End training")





def main(opts):
    from visualizer import visualize
    input_data = torch.randn(opts.batch, 1, opts.image_size, opts.image_size)
    if opts.pre_trained:
        if opts.checkpoint_net:
            model= CheckpointNet("PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")
        else:
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
        train_dataset = SynapseDataset(opts, opts.train_dir, "train", get_train_transform(opts))
        print(train_dataset.__len__())
        train_loader = DataLoader(train_dataset, batch_size=opts.batch, shuffle=True, num_workers=opts.num_workers, pin_memory=True)

        #prendiamo le immagini di validation
        val_dataset = SynapseDataset(opts, opts.validation_dir, "validation", None)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

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
    print(opts.device)

    # Crea checkpoint directory
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    with launch_ipdb_on_exception():
        main(opts)