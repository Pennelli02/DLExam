

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


import torch.nn.functional as F


# helper
def np2th(arr: np.ndarray) -> torch.Tensor:
    """Transpose a numpy array
        Prende un array NumPy e lo converte in un tensore PyTorch float32.
    """
    return torch.tensor(np.array(arr), dtype=torch.float32)

def load_conv(m: nn.Conv2d, kernel: np.ndarray):
    """JAX (H,W,In,Out) → PyTorch (Out,In,H,W)."""
    m.weight.data = np2th(kernel).permute(3, 2, 0, 1)

def load_gn(m: nn.GroupNorm, scale: np.ndarray, bias: np.ndarray):
    """
    Carica i parametri di un GroupNorm da un checkpoint JAX nel modulo PyTorch.

    Il .npz salva scale e bias in formato JAX con shape (1, 1, 1, C),
    pensati per il broadcasting su tensori NHWC (channel-last).
    PyTorch invece si aspetta vettori piatti (C,) per GroupNorm.
    Il .reshape(-1) appiattisce qualsiasi shape in un vettore 1D.
    """
    m.weight.data = np2th(scale).reshape(-1)
    m.bias.data = np2th(bias).reshape(-1)

def load_ln(m: nn.LayerNorm, scale: np.ndarray, bias: np.ndarray):
    """
    Carica i parametri di un LayerNorm da un checkpoint JAX nel modulo PyTorch.

    A differenza di GroupNorm, i parametri LayerNorm nel .npz sono già
    salvati come vettori piatti (C,), quindi non serve il reshape.
    Basta convertire da NumPy a tensore PyTorch con trans().
    """
    m.weight.data = np2th(scale)
    m.bias.data = np2th(bias)

def reshape(x: torch.Tensor) -> torch.Tensor:
    """
    Trasforma la sequenza di token del Transformer in una mappa di feature 2D
    per il decoder convoluzionale (CUP).

    Processo:
    1. Calcola la dimensione spaziale (H, W) dalla lunghezza della sequenza.
    2. Reshape da [B, N, D] a [B, H, W, D].
    3. Permute per portare i canali in posizione 'Channel First' [B, D, H, W].

    :param x: torch.Tensor di forma [B, 196, 768] (Output del Transformer)
    :return: torch.Tensor di forma [B, 768, 14, 14] (Input per il Decoder)
    """

    batch_size, num_tokens, embed_dim = x.shape

    # Calcoliamo la dimensione della griglia (es. sqrt(196) = 14)
    grid_size = int(num_tokens ** 0.5)

    # 1. Riordina i token in una griglia spaziale (H, W)
    # Forma: [B, 14, 14, 768]
    x = x.view(batch_size, grid_size, grid_size, embed_dim)

    # 2. Sposta la dimensione dell'embedding (canali) in seconda posizione
    # Forma finale: [B, 768, 14, 14]
    # .contiguous() serve per riordinare la memoria dopo il permute
    x = x.permute(0, 3, 1, 2).contiguous()

    return x
#---------------------------------------------------------------
# Qui saranno presenti i modelli in versione no pre trained
# RESNET50


class Bottleneck(nn.Module):
    """
    Bottleneck block
    @param in_channels
    @param out_channels
    @param stride
    @param downsample
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # moltiplico per 4 perché così è scritto nel paper per espansione
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(x)  # Riduce i canali (bottleneck)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # FA IL DOWNSAMPLING SPAZIALE (se stride=2)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)  # Espande i canali (expansion)
        out = self.bn3(out)
        out = residual + out
        out = self.relu(out)
        return out


#non è la versione completa, ma quella che serve per il TransUNet
class ResNet50(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Initial convolution and max pooling layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112x64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56x56x64

        # ============ LAYER 1: 56x56x64 → 56x56x256 ============
        # Primo blocco: deve espandere da 64 a 256 canali
        downsample1_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer1_block1 = Bottleneck(64, 64, stride=1, downsample=downsample1_1)

        # Blocchi successivi: 256 → 256 (no downsample)
        self.layer1_block2 = Bottleneck(256, 64, stride=1, downsample=None)
        self.layer1_block3 = Bottleneck(256, 64, stride=1, downsample=None)

        # ============ LAYER 2: 56x56x256 → 28x28x512 ============
        # Primo blocco: stride=2 per ridurre dimensione + cambia canali
        downsample2_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        self.layer2_block1 = Bottleneck(256, 128, stride=2, downsample=downsample2_1)

        # Blocchi successivi: 512 → 512 (no downsample)
        self.layer2_block2 = Bottleneck(512, 128, stride=1, downsample=None)
        self.layer2_block3 = Bottleneck(512, 128, stride=1, downsample=None)
        self.layer2_block4 = Bottleneck(512, 128, stride=1, downsample=None)

        # ============ LAYER 3: 28x28x512 → 14x14x1024 ============
        # Primo blocco: stride=2 + cambia canali
        downsample3_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.layer3_block1 = Bottleneck(512, 256, stride=2, downsample=downsample3_1)

        # Blocchi successivi: 1024 → 1024
        self.layer3_block2 = Bottleneck(1024, 256, stride=1, downsample=None)
        self.layer3_block3 = Bottleneck(1024, 256, stride=1, downsample=None)
        self.layer3_block4 = Bottleneck(1024, 256, stride=1, downsample=None)
        self.layer3_block5 = Bottleneck(1024, 256, stride=1, downsample=None)
        self.layer3_block6 = Bottleneck(1024, 256, stride=1, downsample=None)

        # OUTPUT 14x14x1024

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)  # downsample 1/2

        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x2 = self.layer1_block3(x)  #downsample 1/4

        x = self.layer2_block1(x2)
        x = self.layer2_block2(x)
        x = self.layer2_block3(x)
        x3 = self.layer2_block4(x)  #downsample 1/8

        x = self.layer3_block1(x3)
        x = self.layer3_block2(x)
        x = self.layer3_block3(x)
        x = self.layer3_block4(x)
        x = self.layer3_block5(x)
        x4 = self.layer3_block6(x)

        return x4, [x1, x2, x3]


class MultiheadSelfAttentionBlock(nn.Module):
    # Creates a multi-head self-attention block
    # valori presente nel paper non risulti qui sia presente un dropout
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0):
        super().__init__()

        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x = self.norm1(x)
        # Self-attention layer:
        # - query, key e value sono tutti 'x' perché siamo in self-attention:
        #   ogni token (patch) dell'immagine confronta sé stesso con tutti gli altri token.
        # - multihead_attn ritorna:
        #       attn_output: i token aggiornati dopo il meccanismo di attenzione
        #       attn_weights: le matrici di attenzione (non calcolate qui)
        # - need_weights=False evita di calcolare e restituire le attention maps,
        #   risparmiando memoria e tempo di computazione.
        attn_output, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        return attn_output


class MLPBlock(nn.Module):
    # The MLP contains two layers with a GELU non-linearity
    # Dropout, when used, is applied after every dense layer except for the qkv-projections and directly after adding
    # positional- to patch embeddings
    # layer norm -> linear layer -> non-linear layer -> dropout -> linear layer -> dropout
    # paper dropout= 0.1
    def __init__(self, embed_dim: int = 768, mlp_size=3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_size=3072, dropout: float = 0.1,
                 attn_dropout: float = 0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embed_dim, num_heads, dropout=attn_dropout)

        self.mlp_block = MLPBlock(embed_dim, mlp_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


# qui creiamo e gestiamo la position patch embedding
# --- L'ENCODER IBRIDO COMPLETO (TransUNet Encoder) ---
class Encoder(nn.Module):
    def __init__(self, img_size: int = 224, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()

        # 1. CNN Part (ResNet50 ridotta come da paper)
        self.cnn = ResNet50(in_channels=in_channels)

        # 2. Patch Embedding (Adattato per feature map 1024 invece di immagine raw)
        # Il paper usa una conv 1x1 per passare da 1024 canali della CNN a 768 del ViT
        self.embedding_proj = nn.Conv2d(1024, embed_dim, kernel_size=1)

        # 3. Positional Embedding
        self.num_patches = (img_size // 16) ** 2  # 14x14 = 196
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # 4. Transformer Layers (12 blocchi)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim) for _ in range(12)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Passaggio in CNN
        # x_cnn sarà [B, 1024, 14, 14]
        x_cnn, skips = self.cnn(x)

        # Proiezione canali: 1024 -> 768
        x = self.embedding_proj(x_cnn)  # [B, 768, 14, 14]

        # Flatten e Permute per il Transformer
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]

        # Aggiunta Positional Embedding
        x = x + self.position_embedding
        x = self.dropout(x)

        # Passaggio nei 12 blocchi Transformer
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Ritorna i token del Transformer e le skip connection della CNN per il futuro decoder
        return x, skips


# ----------------------------------------------------------------------------
# pretrained models
class PTResnet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fs = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  #1/2
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  #1/4
        self.layer2 = backbone.layer2  #1/8
        self.layer3 = backbone.layer3  #output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:

        # Se x è 3D [C, H, W], aggiungiamo la dimensione batch -> [1, C, H, W] per evitare problemi relativi alla validazione
        # Qualsiasi cosa arrivi (3D o 4D), lo portiamo a [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        elif x.dim() == 5:
            x = x.squeeze(0)  # [1, 1, C, H, W] -> [1, C, H, W]

        # Se l'input è grayscale, replica i canali perché in pre trained Resnet si aspetta tre canali e non uno
        if x.shape[1] == 1:
            # .expand è meglio di .repeat: non copia i dati in memoria, crea solo viste
            # -1 indica a PyTorch di mantenere la dimensione esistente su quell'asse
            x = x.expand(-1, 3, -1, -1)

        x = self.fs(x)
        x1 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x
        x = self.layer2(x)
        x3 = x
        out = self.layer3(x)
        return out, [x1, x2, x3]


class PreTrainedVit(nn.Module):
    def __init__(self, img_size: int = 224, embed_dim: int = 768):
        # 1. Carichiamo il modello ViT-Base ufficiale di PyTorch con pesi ImageNet
        # Usiamo ViT-B/16 perché ha embed_dim=768 e 12 layer, proprio come TransUNet
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1.DEFAULT
        vit_base = vit_b_16(weights=weights)

        # 2. Embedding e Positional Embedding
        # Il TransUNet non usa il class_token, quindi dobbiamo gestire i pesi
        self.num_patches = (img_size // 16) ** 2  # 196

        # Estraiamo i pesi del positional embedding originale [1, 197, 768]
        # e scartiamo il primo (quello del class_token) per avere [1, 196, 768]
        full_pos_embed = vit_base.encoder.pos_embedding  # Parameter
        self.position_embedding = nn.Parameter(full_pos_embed[:, 1:, :].detach().clone())
        self.transformer_layers = vit_base.encoder.layers
        self.dropout = nn.Dropout(p=0.1)

        # 4. Normalizzazione finale
        self.norm = vit_base.encoder.ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arriva già proiettato a [B, 196, 768] dalla Conv 1x1 dell'Encoder
        # Aggiungiamo i pesi pre-addestrati della posizione
        x = x + self.position_embedding  # per la formula 1 del paper
        x = self.dropout(x)

        # Passiamo attraverso i 12 blocchi del Transformer di torchvision
        x = self.transformer_layers(x)

        x = self.norm(x)
        return x


class PT_Encoder(nn.Module):
    def __init__(self, img_size: int = 224):
        super().__init__()

        self.cnn = PTResnet()

        # Proiezione canali (come da paper: 1024 -> 768)
        self.embedding_proj = nn.Conv2d(1024, 768, kernel_size=1)
        self.flatten = nn.Flatten(2)
        self.vit = PreTrainedVit(img_size=img_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # CNN extractor feature
        x, skips = self.cnn(x)

        # proiezione e flatten
        x = self.embedding_proj(x)  # [B, 768, 14, 14]
        x = self.flatten(x)  # [B, 768, 196]

        #dato che il Vit si aspetta [B, 196, 768]
        x = x.transpose(1, 2)

        #transformer part
        x = self.vit(x)

        # otteniamo l'output del transformer (x) e le skip connections del resnet skip
        return x, skips


# ------------------------------------------------
# models from directory PreTrainedModels

# Dato che stiamo utilizzando modelli BiG Transfer vanno implementate delle accortezze per permettere il fine tuning
# seguito il paper

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

#blocchi con group norm
class BottleneckGN(nn.Module):
    """
    Bottleneck con GroupNorm (num_groups=32), speculare al file .npz ufficiale.
    """
    def __init__(self, in_ch: int, mid_ch: int,
                 stride: int = 1, downsample: nn.Module = None,
                 num_groups: int = 32):
        super().__init__()
        out_ch     = mid_ch * 4
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.gn1   = nn.GroupNorm(num_groups, mid_ch)
        self.conv2 = conv3x3(mid_ch, mid_ch, stride=stride, bias=False)
        self.gn2   = nn.GroupNorm(num_groups, mid_ch)
        self.conv3 = conv1x1(mid_ch, out_ch, bias=False)
        self.gn3   = nn.GroupNorm(num_groups, out_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        return self.relu(out + residual)


def _make_gn_layer(in_ch: int, mid_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    out_ch     = mid_ch * 4
    # Projection also with pre-activation according to paper.
    downsample = nn.Sequential(
        conv1x1(in_ch, out_ch, stride=stride, bias=False),
        nn.GroupNorm(out_ch, out_ch)
    )
    blocks = [BottleneckGN(in_ch, mid_ch, stride=stride, downsample=downsample)]
    for _ in range(1, n_blocks):
        blocks.append(BottleneckGN(out_ch, mid_ch))
    return nn.Sequential(*blocks)


#blocco transformer compatibile col .npz

class TransformerBlockNpz(nn.Module):
    """
    Transformer block con nomi dei sotto-moduli allineati al caricamento
    dei pesi dal file .npz (ln_1, ln_2, self_attention, mlp).
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_dim: int = 3072, dropout: float = 0.0):
        super().__init__()
        self.ln_1           = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed   = self.ln_1(x)
        attn, _  = self.self_attention(normed, normed, normed, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.ln_2(x))
        return x

# encoder dal checkpoint

class CheckpointEncoder(nn.Module):
    """
    Encoder ibrido ResNet50-GN + ViT-B/16 costruito direttamente
    dai pesi del file R50+ViT-B_16.npz.

    Differenze chiave rispetto alle versioni precedenti:
      • GroupNorm (num_groups=32) al posto di BatchNorm
      • layer3 ha 9 blocchi (non 6) come nel checkpoint ufficiale
      • embedding_proj carica anche il bias dal .npz

    Output identico a PT_Encoder e Encoder:
        forward(x) → (tokens [B,196,768], skips [x1,x2,x3])
    Compatibile drop-in con il decoder CUP esistente.
    """

    def __init__(self, img_size: int = 224, embed_dim: int = 768,
                 num_heads: int = 12, mlp_dim: int = 3072,
                 n_transformer_blocks: int = 12):
        super().__init__()

        # ResNet50-GN
        self.conv1   = StdConv2d(3,64 , kernel_size=7, stride=2, padding=3, bias=False)
        self.gn_root = nn.GroupNorm(32, 64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_gn_layer(in_ch=64,  mid_ch=64,  n_blocks=3, stride=1)  # 3 unit
        self.layer2 = _make_gn_layer(in_ch=256, mid_ch=128, n_blocks=4, stride=2)  # 4 unit
        self.layer3 = _make_gn_layer(in_ch=512, mid_ch=256, n_blocks=9, stride=2)  # 9 unit ← dal .npz

        #  proiezione 1×1: 1024 → embed_dim
        self.embedding_proj = nn.Conv2d(1024, embed_dim, kernel_size=1)

        #  positional embedding [1, 196, embed_dim]
        num_patches             = (img_size // 16) ** 2          # 196
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout            = nn.Dropout(0.1) # paper Vit Table 3

        # 12 blocchi Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockNpz(embed_dim, num_heads, mlp_dim)
            for _ in range(n_transformer_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.dim() == 3:     x = x.unsqueeze(0)
        if x.shape[1] == 1:  x = x.expand(-1, 3, -1, -1)   # grayscale → 3ch

        # CNN
        x  = self.conv1(x)
        x = self.gn_root(x)
        x = self.relu(x)
        x1 = x                                               # skip [B, 64,  112, 112]
        x  = self.maxpool(x)
        x  = self.layer1(x)
        x2 = x                                               # skip [B, 256,  56,  56]
        x  = self.layer2(x)
        x3 = x                                               # skip [B, 512,  28,  28]
        x  = self.layer3(x)                                  #      [B,1024,  14,  14]

        # Proiezione + flatten
        x = self.embedding_proj(x)                           # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)                     # [B, 196, 768]

        # Transformer
        x = self.dropout(x + self.position_embedding)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)                                     # [B, 196, 768]

        return x, [x1, x2, x3]

    def load_npz(self, npz_path: str, verbose: bool = True) -> "CheckpointEncoder":
        """
        Carica i pesi da R50+ViT-B_16.npz.
        Ritorna self per uso in catena:
            encoder = CheckpointEncoder().load_npz("R50+ViT-B_16.npz")
        """
        w = dict(np.load(npz_path, allow_pickle=False))

        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  Caricamento checkpoint: {npz_path}")
            print(f"{'─' * 55}")

        # conv_root + gn_root
        load_conv(self.conv1, w['conv_root/kernel'])
        load_gn(self.gn_root, w['gn_root/scale'], w['gn_root/bias'])
        if verbose: print("  conv_root + gn_root")

        # blocchi ResNet
        for block_i, (layer, n_units) in enumerate(
                [(self.layer1, 3), (self.layer2, 4), (self.layer3, 9)], start=1
        ):
            for unit_j in range(1, n_units + 1):
                m = layer[unit_j - 1]
                p = f'block{block_i}/unit{unit_j}'
                load_conv(m.conv1, w[f'{p}/conv1/kernel'])
                load_conv(m.conv2, w[f'{p}/conv2/kernel'])
                load_conv(m.conv3, w[f'{p}/conv3/kernel'])
                load_gn(m.gn1, w[f'{p}/gn1/scale'], w[f'{p}/gn1/bias'])
                load_gn(m.gn2, w[f'{p}/gn2/scale'], w[f'{p}/gn2/bias'])
                load_gn(m.gn3, w[f'{p}/gn3/scale'], w[f'{p}/gn3/bias'])
                if m.downsample is not None:
                    load_conv(m.downsample[0], w[f'{p}/conv_proj/kernel'])
                    load_gn(m.downsample[1], w[f'{p}/gn_proj/scale'], w[f'{p}/gn_proj/bias'])
            if verbose: print(f"  block{block_i}  ({n_units} units)")

        # embedding_proj (Conv 1×1 con bias)
        load_conv(self.embedding_proj, w['embedding/kernel'])
        self.embedding_proj.bias.data = np2th(w['embedding/bias'])
        if verbose: print("  embedding_proj (1024 → 768)")

        # positional embedding: scarta class token → [1, 196, 768]
        self.position_embedding.data = np2th(w['Transformer/posembed_input/pos_embedding'])[:, 1:, :]
        if verbose: print("  positional embedding (class token rimosso)")

        # 12 Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            p = f'Transformer/encoderblock_{i}'
            attn = block.self_attention

            load_ln(block.ln_1, w[f'{p}/LayerNorm_0/scale'], w[f'{p}/LayerNorm_0/bias'])
            load_ln(block.ln_2, w[f'{p}/LayerNorm_2/scale'], w[f'{p}/LayerNorm_2/bias'])

            for idx, name in enumerate(['query', 'key', 'value']):
                # Il .npz salva Q, K, V come tensori SEPARATI con shape (768, 12, 64)
                # PyTorch nn.MultiheadAttention li vuole UNIFICATI in in_proj_weight
                # di shape (3*768, 768) = (2304, 768), nell'ordine Q, K, V.
                #
                # Per ogni proiezione (Q=idx=0, K=idx=1, V=idx=2):
                #   kernel: (768, 12, 64) → reshape(768,768) → .T → slice [s:s+768]
                #   bias:   (12, 64)      → reshape(-1)       →     slice [s:s+768]
                #
                # s = idx * 768 è l'offset nel vettore unificato:
                #   Q occupa [0:768], K occupa [768:1536], V occupa [1536:2304]
                k = w[f'{p}/MultiHeadDotProductAttention_1/{name}/kernel']  # (768,12,64)
                b = w[f'{p}/MultiHeadDotProductAttention_1/{name}/bias']  # (12,64)
                s = idx * 768
                attn.in_proj_weight.data[s:s + 768] = np2th(k.reshape(768, 768)).T
                #print(attn.in_proj_weight.data[s:s + 768].shape)
                attn.in_proj_bias.data[s:s + 768] = np2th(b.reshape(-1))
                #print(attn.in_proj_bias.data[s:s + 768].shape)

            out_k = w[f'{p}/MultiHeadDotProductAttention_1/out/kernel']  # (12,64,768)
            attn.out_proj.weight.data = np2th(out_k.reshape(768, 768).T)
            #print(attn.out_proj.weight.data.shape)
            attn.out_proj.bias.data = np2th(w[f'{p}/MultiHeadDotProductAttention_1/out/bias'])

            block.mlp[0].weight.data = np2th(w[f'{p}/MlpBlock_3/Dense_0/kernel']).T
            block.mlp[0].bias.data = np2th(w[f'{p}/MlpBlock_3/Dense_0/bias'])
            block.mlp[3].weight.data = np2th(w[f'{p}/MlpBlock_3/Dense_1/kernel']).T
            block.mlp[3].bias.data = np2th(w[f'{p}/MlpBlock_3/Dense_1/bias'])

        if verbose: print("  Transformer blocks (12 blocchi)")

        # LayerNorm finale
        load_ln(self.norm, w['Transformer/encoder_norm/scale'], w['Transformer/encoder_norm/bias'])
        if verbose: print("  encoder_norm")

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            print(f"{'─' * 55}")
            print(f"  Parametri totali: {total:,}")
            print(f"  Checkpoint caricato con successo!\n")

        return self
#------------------------------------------------------
# Decoder
class CUPBlock(nn.Module):
    """
        Cascaded Upsampler Block per TransUNet Decoder.

        Fa:
        1. (Opzionale) Concatena con skip connection
        2. Conv 3x3 per fondere le feature
        3. ReLU
        4. BatchNorm2d
        5. Upsample 2x (bilinear interpolation)
        """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # l'uso del doppio layer convoluzionale è stato rispreso dal pytorch segmentation models che presenta le versioni aggiornate dei modelli di segmentazione
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels) # è necessaria?? nel paper non risulta presente Best practice
        # test per vedere se migliorano le prestazioni
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None, debug: bool = False, first_block: bool=None, block_name: str = "") -> torch.Tensor:
        if skip is not None:
            if debug:
                print(f"  [{block_name}] x prima cat: {x.shape}  skip: {skip.shape}")
            x = torch.cat([x, skip], dim=1)
            if debug:
                print(f"  [{block_name}] dopo cat:    {x.shape}")
        else:
            if debug:
                print(f"  [{block_name}] x (no skip): {x.shape}")

        # invertito l'ordine
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #print(x.shape)
        if first_block is None:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
        x = self.upsample(x)
        if debug:
            print(f"  [{block_name}] output:      {x.shape}")
        return x


class CUP(nn.Module):
    """
        Cascaded Upsampler (CUP) - Decoder di TransUNet.

        Prende:
        - Feature dal Transformer reshaped: [B, 196, 768]
        - Skip connections: [skip1 (112x112x64), skip2 (56x56x256), skip3 (28x28x512)]

        Output: [B, out_channels, 224, 224]
        """

    def __init__(self, in_channels: int = 768, out_channels: int = 16):
        super().__init__()
        self.layer1 = CUPBlock(in_channels=in_channels, out_channels=512)
        self.layer2 = CUPBlock(in_channels=1024, out_channels=256)
        self.layer3 = CUPBlock(in_channels=512, out_channels=128)
        self.layer4 = CUPBlock(in_channels=128 + 64, out_channels=64)

    def forward(self, x: torch.Tensor, skip_cnn: list[torch.Tensor], debug: bool = False) -> torch.Tensor:
        """
         Cascaded Upsampler (CUP) - Decoder di TransUNet (esattamente come nel paper).

            Architettura
            - Input: [B, 768, 14, 14]
            - CUP Block 1: 14x14x768 → 28x28x512 (no skip)
            - CUP Block 2: 28x28x512 + skip (28x28x512) → 56x56x256
            - CUP Block 3: 56x56x256 + skip (56x56x256) → 112x112x128
            - CUP Block 4: 112x112x128 + skip (112x112x64) → 224x224x64 siamo giusto prima della convoluzione e segmentation head
                :param x:
                :param skip_cnn [1/2, 1/4, 1/8]:
                :return: x
        """
        if debug:
            print(f"\n[CUP DEBUG]")
            print(f"  input:       {x.shape}")
            for i, s in enumerate(skip_cnn):
                print(f"  skip_cnn[{i}]: {s.shape}")
            print()
        x = self.layer1(x, skip=None, debug=debug, first_block =True, block_name="layer1")
        x = self.layer2(x, skip=skip_cnn[2], debug=debug, first_block =None , block_name="layer2")
        x = self.layer3(x, skip=skip_cnn[1], debug=debug, first_block =None, block_name="layer3")
        x = self.layer4(x, skip=skip_cnn[0], debug=debug, first_block =None, block_name="layer4")

        if debug:
            print(f"\n  output finale: {x.shape}\n")
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 16, n_classes: int = 9):
        super().__init__()
        #inizialmente ripreso dalla UNET conv1x1 però osservando le best practice di pytorch segmentation conv3x3
        self.conv1 = nn.Conv2d(in_channels, n_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x

#---------------------------------------------
#complete models

# versione pretrained
class PT_TransUNet(nn.Module):
    def __init__(self, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.encoder = PT_Encoder(img_size=img_size)
        self.decoder = CUP(in_channels=embed_dim, out_channels=64)
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True), nn.BatchNorm2d(16))
        self.head = SegmentationHead(in_channels=16, n_classes=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip = self.encoder(x)
        x = reshape(x)
        x = self.decoder(x, skip)
        x = self.last_layer(x)
        x = self.head(x)
        return x


# versione non pretrained
class NPT_TransUNet(nn.Module):
    def __init__(self, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.encoder = Encoder(img_size=img_size)
        self.decoder = CUP(in_channels=embed_dim, out_channels=64)
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True))
        self.head = SegmentationHead(in_channels=16, n_classes=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip = self.encoder(x)
        x = reshape(x)
        x = self.decoder(x, skip)
        x = self.last_layer(x)
        x = self.head(x)
        return x


# versione con il checkpoint pre trained
class CheckpointNet(nn.Module):
    """
        TransUNet con encoder dal checkpoint R50+ViT-B_16.npz.

        Uso:
            model = CPT_TransUNet(npz_path="PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")

"""

    def __init__(self, npz_path: str, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.encoder = CheckpointEncoder(img_size=img_size, embed_dim=embed_dim).load_npz(npz_path)
        self.decoder = CUP(in_channels=embed_dim, out_channels=64)
        self.last_layer = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),  nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        self.head = SegmentationHead(in_channels=16, n_classes=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip = self.encoder(x)
        #print(x.shape)
        #print(skip[0].shape)
        #print(skip[1].shape)
        #print(skip[2].shape)
        x = self.decoder(reshape(x), skip)
        #print(x.shape)
        x = self.last_layer(x)
        #print(x.shape)
        x=self.head(x)
        #print(x.shape)
        return x

#---------------------------------------------------
#test
def test_resnet50_encoder():
    """Test dell'encoder ResNet50 custom (non pretrained)"""
    print("=" * 80)
    print("TEST 1: ResNet50 Custom Encoder")
    print("=" * 80)

    model = ResNet50(in_channels=3)
    dummy_input = torch.randn(2, 1, 224, 224)

    print(f"\n Input: {dummy_input.shape}")

    with torch.no_grad():
        output, skips = model(dummy_input)

    print(f"\n Output CNN: {output.shape}")
    print(f"   Expected: torch.Size([2, 1024, 14, 14])")

    print(f"\n Skip Connections:")
    print(f"   Skip 1 (1/2): {skips[0].shape} - Expected: [2, 64, 112, 112]")
    print(f"   Skip 2 (1/4): {skips[1].shape} - Expected: [2, 256, 56, 56]")
    print(f"   Skip 3 (1/8): {skips[2].shape} - Expected: [2, 512, 28, 28]")

    # Verifica dimensioni
    success = True
    if output.shape != (2, 1024, 14, 14):
        print(f"\n Output shape errato!")
        success = False
    if skips[0].shape != (2, 64, 112, 112):
        print(f"\n Skip1 errato! Got {skips[0].shape}")
        success = False
    if skips[1].shape != (2, 256, 56, 56):
        print(f"\n Skip2 errato! Got {skips[1].shape}")
        success = False
    if skips[2].shape != (2, 512, 28, 28):
        print(f"\n Skip3 errato! Got {skips[2].shape}")
        success = False

    if success:
        print("\n ResNet50 PASSATO!")
    return success


def test_pretrained_resnet():
    """Test dell'encoder ResNet50 pretrained"""
    print("\n" + "=" * 80)
    print("TEST 2: PTResnet (Pretrained) Encoder")
    print("=" * 80)

    model = PTResnet()
    dummy_input = torch.randn(2, 1, 224, 224)

    print(f"\n Input: {dummy_input.shape}")

    with torch.no_grad():
        output, skips = model(dummy_input)

    print(f"\n Output CNN: {output.shape}")
    print(f"   Expected: torch.Size([2, 1024, 14, 14])")

    print(f"\n Skip Connections:")
    print(f"   Skip 1 (1/2): {skips[0].shape} - Expected: [2, 64, 112, 112]")
    print(f"   Skip 2 (1/4): {skips[1].shape} - Expected: [2, 256, 56, 56]")
    print(f"   Skip 3 (1/8): {skips[2].shape} - Expected: [2, 512, 28, 28]")

    # Verifica dimensioni
    success = True
    if output.shape != (2, 1024, 14, 14):
        print(f"\n Output shape errato!")
        success = False
    if skips[0].shape != (2, 64, 112, 112):
        print(f"\n Skip1 errato! Got {skips[0].shape}")
        success = False
    if skips[1].shape != (2, 256, 56, 56):
        print(f"\n Skip2 errato! Got {skips[1].shape}")
        success = False
    if skips[2].shape != (2, 512, 28, 28):
        print(f"\n Skip3 errato! Got {skips[2].shape}")
        success = False

    if success:
        print("\n PTResnet PASSATO!")
    return success


def test_custom_encoder():
    """Test dell'encoder completo (CNN + Transformer) non pretrained"""
    print("\n" + "=" * 80)
    print("TEST 3: Encoder Completo (Custom CNN + Transformer)")
    print("=" * 80)

    model = Encoder(img_size=224, in_channels=3, embed_dim=768)
    dummy_input = torch.randn(2, 3, 224, 224)

    print(f"\n Input: {dummy_input.shape}")

    with torch.no_grad():
        tokens, skips = model(dummy_input)

    print(f"\n Output Transformer: {tokens.shape}")
    print(f"   Expected: torch.Size([2, 196, 768])")

    print(f"\n Skip Connections:")
    print(f"   Skip 1: {skips[0].shape} - Expected: [2, 64, 112, 112]")
    print(f"   Skip 2: {skips[1].shape} - Expected: [2, 256, 56, 56]")
    print(f"   Skip 3: {skips[2].shape} - Expected: [2, 512, 28, 28]")

    # Verifica dimensioni
    success = True
    if tokens.shape != (2, 196, 768):
        print(f"\n Tokens shape errato! Got {tokens.shape}")
        success = False
    if skips[0].shape != (2, 64, 112, 112):
        print(f"\n Skip1 errato! Got {skips[0].shape}")
        success = False
    if skips[1].shape != (2, 256, 56, 56):
        print(f"\n Skip2 errato! Got {skips[1].shape}")
        success = False
    if skips[2].shape != (2, 512, 28, 28):
        print(f"\n Skip3 errato! Got {skips[2].shape}")
        success = False

    if success:
        print("\n Encoder Custom PASSATO!")
    return success


def test_pretrained_encoder():
    """Test dell'encoder pretrained completo"""
    print("\n" + "=" * 80)
    print("TEST 4: PT_Encoder (Pretrained ResNet + ViT)")
    print("=" * 80)

    model = PT_Encoder(img_size=224)
    dummy_input = torch.randn(2, 3, 224, 224)

    print(f"\n Input: {dummy_input.shape}")

    with torch.no_grad():
        tokens, skips = model(dummy_input)

    print(f"\n Output Transformer: {tokens.shape}")
    print(f"   Expected: torch.Size([2, 196, 768])")

    print(f"\n Skip Connections:")
    print(f"   Skip 1: {skips[0].shape} - Expected: [2, 64, 112, 112]")
    print(f"   Skip 2: {skips[1].shape} - Expected: [2, 256, 56, 56]")
    print(f"   Skip 3: {skips[2].shape} - Expected: [2, 512, 28, 28]")

    # Verifica dimensioni
    success = True
    if tokens.shape != (2, 196, 768):
        print(f"\n Tokens shape errato! Got {tokens.shape}")
        success = False
    if skips[0].shape != (2, 64, 112, 112):
        print(f"\n Skip1 errato! Got {skips[0].shape}")
        success = False
    if skips[1].shape != (2, 256, 56, 56):
        print(f"\n Skip2 errato! Got {skips[1].shape}")
        success = False
    if skips[2].shape != (2, 512, 28, 28):
        print(f"\n Skip3 errato! Got {skips[2].shape}")
        success = False

    if success:
        print("\n PT_Encoder PASSATO!")
    return success


def test_decoder():
    """Test del decoder CUP"""
    print("\n" + "=" * 80)
    print("TEST 5: Decoder CUP")
    print("=" * 80)

    decoder = CUP(in_channels=768, out_channels=64)

    # Simula input
    x = torch.randn(2, 768, 14, 14)
    skip1 = torch.randn(2, 64, 112, 112)
    skip2 = torch.randn(2, 256, 56, 56)
    skip3 = torch.randn(2, 512, 28, 28)

    print(f"\n Input decoder: {x.shape}")
    print(f"   Skip 1: {skip1.shape}")
    print(f"   Skip 2: {skip2.shape}")
    print(f"   Skip 3: {skip3.shape}")

    with torch.no_grad():
        output = decoder(x, [skip1, skip2, skip3])

    print(f"\n Output decoder: {output.shape}")
    print(f"   Expected: torch.Size([2, 64, 224, 224])")

    success = output.shape == (2, 64, 224, 224)
    if not success:
        print(f"\n Decoder output errato! Got {output.shape}")
    else:
        print("\n Decoder CUP PASSATO!")

    return success


def test_full_npt_transunet():
    """Test del modello completo NPT_TransUNet (non pretrained)"""
    print("\n" + "=" * 80)
    print("TEST 6: NPT_TransUNet COMPLETO (Non Pretrained)")
    print("=" * 80)

    model = NPT_TransUNet(img_size=224, embed_dim=768)
    dummy_input = torch.randn(2, 3, 224, 224)

    print(f"\n Input: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n Output finale: {output.shape}")
    print(f"   Expected: torch.Size([2, 9, 224, 224])")

    success = output.shape == (2, 9, 224, 224)
    if not success:
        print(f"\n Output errato! Got {output.shape}")
    else:
        print("\n NPT_TransUNet COMPLETO PASSATO!")

    return success


def test_full_pt_transunet():
    """Test del modello completo PT_TransUNet (pretrained)"""
    print("\n" + "=" * 80)
    print("TEST 7: PT_TransUNet COMPLETO (Pretrained)")
    print("=" * 80)

    model = PT_TransUNet(img_size=224, embed_dim=768)
    dummy_input = torch.randn(2, 3, 224, 224)

    print(f"\n Input: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n Output finale: {output.shape}")
    print(f"   Expected: torch.Size([2, 9, 224, 224])")

    success = output.shape == (2, 9, 224, 224)
    if not success:
        print(f"\n Output errato! Got {output.shape}")
    else:
        print("\n PT_TransUNet COMPLETO PASSATO!")

    return success


def run_all_tests():
    """Esegue tutti i test in sequenza"""
    print("\n" + "|" * 80)
    print("|" + " " * 78 + "|")
    print("|" + " " * 25 + "TEST SUITE COMPLETO" + " " * 34 + "|")
    print("|" + " " * 78 + "|")
    print("|" * 80 + "\n")

    results = []

    try:
        results.append(("ResNet50 Custom", test_resnet50_encoder()))
    except Exception as e:
        print(f"\n ERRORE in ResNet50: {e}")
        results.append(("ResNet50 Custom", False))

    try:
        results.append(("PTResnet", test_pretrained_resnet()))
    except Exception as e:
        print(f"\n ERRORE in PTResnet: {e}")
        results.append(("PTResnet", False))

    try:
        results.append(("Encoder Custom", test_custom_encoder()))
    except Exception as e:
        print(f"\n ERRORE in Encoder Custom: {e}")
        results.append(("Encoder Custom", False))

    try:
        results.append(("PT_Encoder", test_pretrained_encoder()))
    except Exception as e:
        print(f"\n ERRORE in PT_Encoder: {e}")
        results.append(("PT_Encoder", False))

    try:
        results.append(("Decoder CUP", test_decoder()))
    except Exception as e:
        print(f"\n ERRORE in Decoder: {e}")
        results.append(("Decoder CUP", False))

    try:
        results.append(("NPT_TransUNet", test_full_npt_transunet()))
    except Exception as e:
        print(f"\n ERRORE in NPT_TransUNet: {e}")
        results.append(("NPT_TransUNet", False))

    try:
        results.append(("PT_TransUNet", test_full_pt_transunet()))
    except Exception as e:
        print(f"\n ERRORE in PT_TransUNet: {e}")
        results.append(("PT_TransUNet", False))

    # Riepilogo finale
    print("\n" + "|" * 80)
    print("|" + " " * 78 + "|")
    print("|" + " " * 28 + "RIEPILOGO FINALE" + " " * 34 + "|")
    print("|" + " " * 78 + "|")
    print("|" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASSATO" if success else " FALLITO"
        print(f"\n{test_name:30s} {status}")

    print("\n" + "=" * 80)
    print(f"TOTALE: {passed}/{total} test passati")

    if passed == total:
        print("\n TUTTI I TEST PASSATI! ")
    else:
        print(f"\n  {total - passed} test falliti - controlla gli errori sopra")

    print("=" * 80 + "\n")


# ESEGUI TUTTI I TEST
if __name__ == "__main__":
    #with torch.no_grad():
    #   run_all_tests()

    #model= resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #print(model)


    #model=CheckpointEncoder()
    #print(model)

    model=CheckpointNet("PreTrainedModels/imagenet21k/R50+ViT-B_16.npz")
    print(model)
