from unittest import skipIf

import torch
import torchvision
from torch import nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

#---------------------------------------------------------------
# Qui saranno presenti i modelli in versione no pre trained
# RESNET50

def reshape (x: torch.Tensor) -> torch.Tensor:
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
    grid_size = int(num_tokens**0.5)

    # 1. Riordina i token in una griglia spaziale (H, W)
    # Forma: [B, 14, 14, 768]
    x = x.view(batch_size, grid_size, grid_size, embed_dim)

    # 2. Sposta la dimensione dell'embedding (canali) in seconda posizione
    # Forma finale: [B, 768, 14, 14]
    # .contiguous() serve per riordinare la memoria dopo il permute
    x = x.permute(0, 3, 1, 2).contiguous()

    return x


class Bottleneck(nn.Module):
    """
    Bottleneck block
    @param in_channels
    @param out_channels
    @param stride
    @param downsample
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int =1, downsample=None):
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
        out = self.conv1(x) # Riduce i canali (bottleneck)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # FA IL DOWNSAMPLING SPAZIALE (se stride=2)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out) # Espande i canali (expansion)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 112x112x64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 56x56x64

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
        x = self.relu(x)
        x1 = self.maxpool(x) # downsample 1/2

        x = self.layer1_block1(x1)
        x = self.layer1_block2(x)
        x2 = self.layer1_block3(x) #downsample 1/4

        x = self.layer2_block1(x2)
        x = self.layer2_block2(x)
        x = self.layer2_block3(x)
        x3 = self.layer2_block4(x) #downsample 1/8

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
    def __init__(self, embed_dim: int = 768, num_heads: int= 12, dropout: float = 0):
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
    def __init__(self, embed_dim: int = 768, mlp_size= 3072, dropout: float = 0.1):
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
    def __init__(self, embed_dim: int = 768, num_heads: int= 12, mlp_size= 3072, dropout: float = 0.1, attn_dropout: float = 0):
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
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fs = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu) #1/2
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1 #1/4
        self.layer2 = backbone.layer2 #1/8
        self.layer3 = backbone.layer3 #output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        vit_base = vit_b_16(weights=weights)

        # 2. Embedding e Positional Embedding
        # Il TransUNet non usa il class_token, quindi dobbiamo gestire i pesi
        self.num_patches = (img_size // 16) ** 2  # 196

        # Estraiamo i pesi del positional embedding originale [1, 197, 768]
        # e scartiamo il primo (quello del class_token) per avere [1, 196, 768]
        full_pos_embed = vit_base.encoder.pos_embedding  # Parameter
        self.position_embedding = nn.Parameter(full_pos_embed[:, 1:, :].detach().clone())

        self.dropout = nn.Dropout(p=0.1)

        # 4. Normalizzazione finale
        self.norm = vit_base.encoder.ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arriva già proiettato a [B, 196, 768] dalla Conv 1x1 dell'Encoder
        # Aggiungiamo i pesi pre-addestrati della posizione
        x = x + self.position_embedding # per la formula 1 del paper
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
        x = self.embedding_proj(x) # [B, 768, 14, 14]
        x = self.flatten(x) # [B, 768, 196]

        #dato che il Vit si aspetta [B, 196, 768]
        x = x.transpose(1, 2)

        #transformer part
        x = self.vit(x)

        # otteniamo l'output del transformer (x) e le skip connections del resnet skip
        return x, skips

class CUPBlock(nn.Module):
    """
        Cascaded Upsampler Block per TransUNet Decoder.

        Fa:
        1. (Opzionale) Concatena con skip connection
        2. Conv 3x3 per fondere le feature
        3. ReLU
        4. Upsample 2x (bilinear interpolation)
        """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            x= torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample(x)
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
        self.layer2 = CUPBlock(in_channels=512, out_channels=256)
        self.layer3 = CUPBlock(in_channels=256, out_channels=128)
        self.layer4 = CUPBlock(in_channels=128, out_channels=64)

    def forward(self, x: torch.Tensor, skip_cnn: list[torch.Tensor]) -> torch.Tensor:
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
        x = self.layer1(x)
        x = self.layer2(x, skip_cnn[2])
        x = self.layer3(x, skip_cnn[1])
        x = self.layer4(x, skip_cnn[0])
        return x



