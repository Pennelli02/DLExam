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


from src.transUNet import PT_TransUNet, NPT_TransUNet
from src.dataset import SynapseDataset


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


def train_loop(model, train, valid, opts):
    import tensorflow as tf
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_name}/train')
    val_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model_name}/validation')

def main(opts):
    from visualizer import visualize
    input_data = torch.randn(opts.batch_size, 1, opts.image_size, opts.image_size)
    if opts.pre_trained:
        model = PT_TransUNet()
    else:
        model = NPT_TransUNet()
    #CL visualization
    visualize(model, opts.model_name, input_data)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with launch_ipdb_on_exception():
        main(opts)