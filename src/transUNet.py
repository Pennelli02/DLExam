
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
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fs = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  #1/2
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  #1/4
        self.layer2 = backbone.layer2  #1/8
        self.layer3 = backbone.layer3  #output

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
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
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
        self.layer2 = CUPBlock(in_channels=1024, out_channels=256)
        self.layer3 = CUPBlock(in_channels=512, out_channels=128)
        self.layer4 = CUPBlock(in_channels=128+64, out_channels=64)

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


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int = 16, n_classes: int = 9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x
# versione pretrained
class PT_TransUNet(nn.Module):
    def __init__(self, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.encoder = PT_Encoder(img_size=img_size)
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

# versione non pretrained
class NPT_TransUNet(nn.Module):
    def __init__(self, img_size: int = 224, embed_dim: int = 768):
        super().__init__()
        self.encoder = Encoder(img_size=img_size)
        self.decoder = CUP(in_channels=embed_dim, out_channels=64)
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.head = SegmentationHead(in_channels=16, n_classes=9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip = self.encoder(x)
        x = reshape(x)
        x = self.decoder(x, skip)
        x = self.last_layer(x)
        x = self.head(x)
        return x



def test_resnet50_encoder():
    """Test dell'encoder ResNet50 custom (non pretrained)"""
    print("=" * 80)
    print("TEST 1: ResNet50 Custom Encoder")
    print("=" * 80)

    model = ResNet50(in_channels=3)
    dummy_input = torch.randn(2, 3, 224, 224)

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
    dummy_input = torch.randn(2, 3, 224, 224)

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
    with torch.no_grad():
        run_all_tests()