import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchinfo import summary
from rich.console import Console
import torchvision.utils as vutils
console = Console()

def visualize(model, model_name, input_data):
    out = model(input_data)
    console.print(f'Computed output, shape = {out.shape=}')
    model_stats = summary(model,
                          input_data=input_data,
                          col_names=[
                              "input_size",
                              "output_size",
                              "num_params",
                              # "params_percent",
                              # "kernel_size",
                              # "mult_adds",
                          ],
                          row_settings=("var_names",),
                          col_width=18,
                          depth=8,
                          verbose=0,
                          )
    console.print(model_stats)

def visualize_segm(image, prediction, label):
    """
    image: fetta 2D della CT [H, W]
    prediction: maschera predetta dalla rete [H, W]
    label: maschera reale (Ground Truth) [H, W]
    """
    plt.figure(figsize=(12, 4))

    # 1. Immagine Originale
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("CT Originale")
    plt.axis('off')

    # 2. Predizione con Colori (usiamo 'jet' o 'nipy_spectral' per distinguere gli organi)
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='nipy_spectral') # Colori diversi per classi diverse
    plt.title("Predizione IA")
    plt.axis('off')

    # 3. Verità (Ground Truth)
    plt.subplot(1, 3, 3)
    plt.imshow(label, cmap='nipy_spectral')
    plt.title("Verità (Ground Truth)")
    plt.axis('off')

    plt.show() # Apre una finestra pop-up in PyCharm


def visualize_batch(dataloader, num_images=16, save_path=None):
    """
    Visualizza una griglia di immagini e relative label da un DataLoader.

    Args:
        dataloader: Il DataLoader di Synapse
        num_images: Quante coppie (Immagine/Label) mostrare (max 32 per leggibilità)
        save_path: Se fornito, salva l'immagine su disco
    """
    # 1. Estrae il primo batch disponibile
    batch = next(iter(dataloader))
    images = batch['image']
    labels = batch['label']

    # 2. Prepariamo la lista per la griglia
    # Affianchiamo immagine e label per un confronto immediato
    display_list = []
    for i in range(min(num_images, len(images))):
        # Portiamo l'immagine nel range [0, 1] per la visualizzazione
        img = images[i]

        # Rendiamo la label visibile: la normalizziamo o usiamo una colormap
        # Qui la normalizziamo scalarmente per farla risaltare nella griglia gray
        lab = labels[i].unsqueeze(0).float() / 8.0

        display_list.append(img)
        display_list.append(lab)

    # 3. Creazione della griglia
    grid = vutils.make_grid(display_list, nrow=8, padding=2, normalize=False)

    # 4. Visualizzazione con Matplotlib
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Visualizzazione Batch: CT (Sinistra) vs Label (Destra)")

    # Trasponiamo da [C, H, W] a [H, W, C]
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)), cmap='nipy_spectral')

    if save_path:
        plt.savefig(save_path)
        print(f"Griglia salvata in: {save_path}")

    plt.show()


def show_single_slice(image, label, title="Controllo Qualità"):
    """Visualizza una singola fetta e la sua maschera sovrapposta."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Immagine CT")

    plt.subplot(1, 2, 2)
    # Overlay: l'immagine sotto, la label colorata sopra con trasparenza
    plt.imshow(image.squeeze(), cmap='gray')
    masked_label = np.ma.masked_where(label == 0, label)
    plt.imshow(masked_label.squeeze(), cmap='rainbow', alpha=0.5)
    plt.title("Overlay Organi")

    plt.suptitle(title)
    plt.show()


def save_and_display_segmentation(data_path, output_dir="outputs"):
    """
    Carica una slice, applica il colore alla label, salva e visualizza.
    """
    # 1. Caricamento dei dati
    # Carica il file .npz creato durante il preprocessing
    data = np.load(data_path)
    image_original = data['image'][:]  # Pixel della CT normalizzati [0, 1]
    label_image = data['label'][:]  # Maschera con valori interi 0-8

    # 2. Trasformazione in colori (Versione ottimizzata)
    # Invece di un ciclo for, creiamo una palette di colori RGB
    # Ogni riga corrisponde a una classe: [R, G, B]
    colors = np.array([
        [0, 0, 0],  # 0: Sfondo (Nero)
        [1, 0, 0],  # 1: Aorta (Rosso)
        [0, 1, 0],  # 2: Cistifellea (Verde)
        [0, 0, 1],  # 3: Rene Sinistro (Blu)
        [1, 1, 0],  # 4: Rene Destro (Giallo)
        [1, 0, 1],  # 5: Fegato (Magenta)
        [0, 1, 1],  # 6: Pancreas (Ciano)
        [1, 0, 0.5],  # 7: Milza (Rosa)
        [0, 0.5, 1]  # 8: Stomaco (Azzurro)
    ])

    # Mappatura istantanea: ogni valore in label_image diventa il colore corrispondente
    coloured_label = colors[label_image.astype(int)]

    # 3. Conversione in formato immagine (0-255 uint8)
    # Moltiplichiamo per 255 perché Image.fromarray richiede byte, non float
    lb_uint8 = (coloured_label * 255).astype(np.uint8)
    im_uint8 = (image_original * 255).astype(np.uint8)

    # Crea l'oggetto immagine PIL
    lb_pil = Image.fromarray(lb_uint8)
    im_pil = Image.fromarray(im_uint8)

    # 4. Salvataggio su disco
    import os
    os.makedirs(output_dir, exist_ok=True)
    lb_path = os.path.join(output_dir, 'label_colored.bmp')
    im_path = os.path.join(output_dir, 'image_original.bmp')

    lb_pil.save(lb_path)
    im_pil.save(im_path)

    # 5. Visualizzazione
    # Usiamo matplotlib per vederle affiancate in PyCharm
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(im_pil, cmap='gray')
    plt.title("CT Originale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lb_pil)
    plt.title("Label Colorata")
    plt.axis('off')

    plt.show()

    return im_path, lb_path

def npz_summary(npz_path: str, max_depth: int = 5) -> None:
    """
    Stampa la struttura del file .npz in stile torchinfo:
    gerarchia indentata con layer name, shape e conteggio parametri.
    """

    w     = dict(np.load(npz_path, allow_pickle=False))
    keys  = sorted(w.keys())
    total = sum(v.size for v in w.values())

    # ── intestazione ─────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 88 + "┐")
    print(f"│  {'NPZ SUMMARY':^86}│")
    print(f"│  File : {npz_path:<77}│")
    print(f"│  Chiavi: {len(keys):<9}  Parametri totali: {total:<52,}│")
    print("├" + "─" * 42 + "┬" + "─" * 28 + "┬" + "─" * 16 + "┤")
    print(f"│  {'Layer name':<40}│ {'Shape':<27}│ {'Params':>14} │")
    print("├" + "─" * 42 + "┼" + "─" * 28 + "┼" + "─" * 16 + "┤")

    # ── costruzione albero ───────────────────────────────────────────────────
    # Ogni nodo ha:
    #   "_leaves": lista di chiavi complete che appartengono a questo nodo
    #   "children": dizionario {nome: nodo_figlio}
    def _new_node():
        return {"_leaves": [], "children": {}}

    root = _new_node()
    for k in keys:
        # Splitta la chiave gerarchica: "Transformer/block_0/Dense/kernel"
        # -> ["Transformer", "block_0", "Dense", "kernel"]
        parts = k.split("/")
        node  = root
        # Naviga/crea i nodi intermedi (tutti tranne l'ultimo che è il nome del tensore)
        for part in parts[:-1]:
            node["children"].setdefault(part, _new_node())
            node = node["children"][part]
        # L'ultimo elemento è il tensore foglia: lo aggiungiamo al nodo corrente
        node["_leaves"].append(k)

    # ── conteggio parametri ricorsivo ────────────────────────────────────────
    def _count_params(node) -> int:
        # Somma i parametri delle foglie dirette
        total = sum(w[k].size for k in node["_leaves"])
        # Ricorre nei figli
        for child in node["children"].values():
            total += _count_params(child)
        return total

    # ── stampa riga tensore foglia ────────────────────────────────────────────
    def _print_leaf(key: str, prefix: str, is_last: bool) -> None:
        # Prende solo il nome finale della chiave (dopo l'ultimo "/")
        name      = key.split("/")[-1]
        connector = "└─ " if is_last else "├─ "
        row_name  = (prefix + connector + name)[:40]
        shape_str = str(w[key].shape)
        params    = w[key].size
        print(f"│  {row_name:<40}│ {shape_str:<27}│ {params:>14,} │")

    # ── stampa nodo ricorsivo ─────────────────────────────────────────────────
    def _print_node(node: dict, prefix: str, depth: int) -> None:
        # Se abbiamo superato la profondità massima, non stampiamo nulla
        if depth > max_depth:
            return

        leaves   = node["_leaves"]
        children = list(node["children"].items())  # lista di (nome, nodo)

        # Calcola quanti elementi totali ha questo nodo (foglie + figli)
        # per determinare correttamente quale è l'ultimo
        all_items = leaves + [name for name, _ in children]
        n_items   = len(all_items)

        # Stampa prima le foglie dirette di questo nodo
        for i, key in enumerate(leaves):
            # È l'ultimo elemento solo se non ci sono figli dopo
            is_last = (i == len(leaves) - 1) and len(children) == 0
            _print_leaf(key, prefix, is_last)

        # Poi stampa i nodi figli
        for i, (name, child_node) in enumerate(children):
            is_last   = (i == len(children) - 1)
            connector = "└─ " if is_last else "├─ "

            # Calcola statistiche del sottoalbero figlio
            child_params = _count_params(child_node)
            n_leaves     = sum(1 for _ in _iter_leaves(child_node))

            # Riga di intestazione del nodo figlio
            sec_label = (prefix + connector + name)[:40]
            sec_info  = f"[{n_leaves} tensori]"
            print(f"│  {sec_label:<40}│ {sec_info:<27}│ {child_params:>14,} │")

            # Calcola il nuovo prefisso per i figli di questo nodo:
            # se è l'ultimo figlio usiamo "   " (spazio), altrimenti "│  " (barra)
            new_prefix = prefix + ("   " if is_last else "│  ")

            # Ricorre nel figlio solo se non abbiamo superato max_depth
            _print_node(child_node, new_prefix, depth + 1)

    # ── iteratore foglie ─────────────────────────────────────────────────────
    def _iter_leaves(node):
        # Genera tutte le chiavi foglia nel sottoalbero
        yield from node["_leaves"]
        for child in node["children"].values():
            yield from _iter_leaves(child)

    # ── stampa root ───────────────────────────────────────────────────────────
    # Le foglie dirette della root (chiavi senza "/" nel nome)
    for i, key in enumerate(root["_leaves"]):
        is_last = (i == len(root["_leaves"]) - 1) and not root["children"]
        _print_leaf(key, "", is_last)

    # I figli top-level della root
    top_children = list(root["children"].items())
    for i, (name, child_node) in enumerate(top_children):
        is_last      = (i == len(top_children) - 1)
        connector    = "└─ " if is_last else "├─ "
        child_params = _count_params(child_node)
        n_leaves     = sum(1 for _ in _iter_leaves(child_node))

        sec_label = (connector + name)[:40]
        sec_info  = f"[{n_leaves} tensori]"
        print(f"│  {sec_label:<40}│ {sec_info:<27}│ {child_params:>14,} │")

        new_prefix = "   " if is_last else "│  "
        _print_node(child_node, new_prefix, depth=2)

    # ── footer ────────────────────────────────────────────────────────────────
    print("├" + "─" * 42 + "┴" + "─" * 28 + "┴" + "─" * 16 + "┤")
    print(f"│  {'TOTALE PARAMETRI':>40}   {total:>42,} │")
    print("└" + "─" * 88 + "┘\n")


def load_local_weights(path):
    """
    Carica i pesi da un file .npz locale.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Il file non è stato trovato nel percorso: {path}")

    # load con allow_pickle=False è la best practice per file di pesi
    return np.load(path, allow_pickle=False)

if __name__ == "__main__":
    npz_summary("PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")

    #weights=load_local_weights("PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")
    #print(weights)