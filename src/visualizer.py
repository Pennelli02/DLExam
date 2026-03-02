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

def npz_summary(npz_path: str, max_depth: int = 3) -> None:
    """
    Stampa la struttura del file .npz in stile torchinfo:
    gerarchia indentata con layer name, shape e conteggio parametri.

    :param npz_path:  path del file .npz
    :param max_depth: livelli di indentazione gerarchica (default 3)
    """
    w     = dict(np.load(npz_path, allow_pickle=False))
    keys  = sorted(w.keys())
    total = sum(v.size for v in w.values())

    # ── intestazione ─────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 88 + "┐")
    print(f"│  {'NPZ SUMMARY':^86}│")
    print(f"│  File : {npz_path:<77}│")
    print(f"│  Layer: {len(keys):<10}  Parametri totali: {total:<52,}│")
    print("├" + "─" * 42 + "┬" + "─" * 28 + "┬" + "─" * 16 + "┤")
    print(f"│  {'Layer name':<40}│ {'Shape':<27}│ {'Params':>14} │")
    print("├" + "─" * 42 + "┼" + "─" * 28 + "┼" + "─" * 16 + "┤")

    # ── costruisci albero ────────────────────────────────────────────────────
    # nodo = {"_keys": [...], "children": {name: nodo}}
    def _new_node():
        return {"_keys": [], "children": {}}

    root = _new_node()
    for k in keys:
        parts = k.split("/")
        node  = root
        for part in parts[:-1]:
            node["children"].setdefault(part, _new_node())
            node = node["children"][part]
        node["_keys"].append(k)

    # ── stampa ricorsiva ─────────────────────────────────────────────────────
    def _params_in_node(node) -> int:
        total = sum(w[k].size for k in node["_keys"])
        for child in node["children"].values():
            total += _params_in_node(child)
        return total

    def _print_node(node, prefix: str, depth: int, is_last: bool):
        if depth > max_depth:
            return

        connector = "└─ " if is_last else "├─ "
        child_names = list(node["children"].keys())

        # Stampa i tensori foglia di questo nodo
        for ki, k in enumerate(node["_keys"]):
            leaf_last = (ki == len(node["_keys"]) - 1) and not child_names
            lc = "└─ " if leaf_last else "├─ "
            leaf_name = k.split("/")[-1]
            shape_str = str(w[k].shape)
            params    = w[k].size
            row_name  = (prefix + lc + leaf_name)[:40]
            print(f"│  {row_name:<40}│ {shape_str:<27}│ {params:>14,} │")

        # Stampa i nodi figli
        for ci, cname in enumerate(child_names):
            clast     = ci == len(child_names) - 1
            cnode     = node["children"][cname]
            cparams   = _params_in_node(cnode)
            n_tensors = sum(1 for _ in _iter_leaves(cnode))
            indent    = prefix + ("   " if is_last else "│  ")

            # riga sezione
            sec_label = (indent + ("└─ " if clast else "├─ ") + cname)[:40]
            sec_info  = f"[{n_tensors} tensori]"
            print(f"│  {sec_label:<40}│ {sec_info:<27}│ {cparams:>14,} │")

            # ricorre nei figli
            _print_node(cnode,
                        indent + ("   " if clast else "│  "),
                        depth + 1,
                        clast)

    def _iter_leaves(node):
        yield from node["_keys"]
        for child in node["children"].values():
            yield from _iter_leaves(child)

    # Stampa i top-level
    top_keys   = list(root["_keys"])
    top_childs = list(root["children"].keys())

    for k in top_keys:
        shape_str = str(w[k].shape)
        params    = w[k].size
        print(f"│  {'├─ ' + k:<40}│ {shape_str:<27}│ {params:>14,} │")

    for ci, cname in enumerate(top_childs):
        clast  = ci == len(top_childs) - 1
        cnode  = root["children"][cname]
        cp     = _params_in_node(cnode)
        ntens  = sum(1 for _ in _iter_leaves(cnode))
        lc     = "└─ " if clast else "├─ "
        sec_label = (lc + cname)[:40]
        print(f"│  {sec_label:<40}│ {'[' + str(ntens) + ' tensori]':<27}│ {cp:>14,} │")
        _print_node(cnode, "   " if clast else "│  ", 2, clast)

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
    weights=load_local_weights("PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")
    print(weights)