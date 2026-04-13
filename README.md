# TransUNet

PyTorch implementation of TransUNet for multi-organ CT segmentation on the Synapse dataset.

## Overview

Hybrid encoder-decoder architecture combining a ResNet50 CNN backbone with a ViT-B/16 Transformer encoder and a Cascaded Upsampler (CUP) decoder with skip connections. Three model variants are provided: a fully pretrained version loaded from the official ImageNet21k checkpoint (`CheckpointNet`), a torchvision-pretrained version (`PT_TransUNet`), and a randomly initialized baseline (`NPT_TransUNet`).

## Requirements

Python 3.10+, CUDA GPU, and the packages listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Synapse multi-organ segmentation dataset. Registration at [synapse.org](https://www.synapse.org) is required. Set your personal access token in a `.env` file:
SYNAPSE_TOKEN=your_token_here

Then run the full setup pipeline (download, extraction, preprocessing):

```bash
python src/setup.py
```

Preprocessing applies HU windowing to [-125, 275], normalizes to [0, 1], converts training cases to 2D slices (`.npz`) and validation/test cases to 3D volumes (`.h5`).

## Pretrained Weights

`CheckpointNet` requires `R50+ViT-B_16.npz` from the [official TransUNet repository](https://github.com/Beckschen/TransUNet). Place it at: PreTrainedModels/imagenet21k/R50+ViT-B_16.npz

## Training

All hyperparameters are controlled through `src/config.yaml`. The most relevant
options are:

| Parameter              | Default | Description                                      |
|------------------------|---------|--------------------------------------------------|
| `model_name`           |         | Model variant to use                             |
| `checkpoint_net`       | true    | Load encoder from `.npz` checkpoint              |
| `lr`                   | 0.01    | Base learning rate (polynomial decay)            |
| `batch`                | 24      | Training batch size                              |
| `n_epoch_sy`           | 152     | Total training epochs                            |
| `validation_frequency` | 8       | Volumetric validation every N epochs             |

Configure `src/config.yaml`, then:

```bash
python src/train.py src/config.yaml
```

The training loop uses a combined Dice + Cross-Entropy loss (0.5/0.5), polynomial LR decay, and mixed-precision (AMP). TensorBoard logs are written to `tensorboard/<model_name>/`.

## Evaluation

```bash
python src/test.py src/config.yaml
```

Runs slice-by-slice volumetric inference and reports per-organ Dice and HD95 for each test case, comparing `scipy.ndimage.zoom` and `torchvision v2.Resize` upsampling strategies.

## Results

Best validation result (CheckpointNet, scipy resize, 152 epochs):

| Organ         | Dice   | HD95 (mm) |
|---------------|--------|-----------|
| Aorta         | 0.8323 | 7.02      |
| Gallbladder   | 0.5019 | 28.12     |
| Left Kidney   | 0.7981 | 47.44     |
| Right Kidney  | 0.7510 | 24.73     |
| Liver         | 0.9301 | 36.03     |
| Pancreas      | 0.5339 | 15.41     |
| Spleen        | 0.8570 | 25.91     |
| Stomach       | 0.7427 | 16.32     |
| Mean          | 0.7434 | 25.12     |

Scores are not directly comparable to the paper due to a different train/validation split.

## Project Structure

```
src/
  config.yaml       # hyperparameters and paths
  transUNet.py      # model definitions
  dataset.py        # dataset and augmentation
  train.py          # training loop
  test.py           # volumetric inference
  utils.py          # preprocessing and metrics
  visualizer.py     # model summary and visualization
  setup.py          # dataset setup entry point
  PreTrainedModels/
    imagenet21k/
      R50+ViT-B_16.npz   # not included, download separately
```
## Additive info
if you are interested on ulterior info you could look doc/info.md, where you can find my notes about experimental choices
## Reference

Chen et al., [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306), 2021.
