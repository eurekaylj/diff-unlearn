# DiffUnlearn

Code for paper "DiffUnlearn: Selectively Unlearnable Examples via Diffusion Models".

## Installation

Install the dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Unlearnable Images

Use the generation component corresponding to your surrogate:

- **CLIP surrogate:** [code/selective_diffunlearn.py](code/selective_diffunlearn.py).
- **Classification surrogate:** [code/cifar_selective_generation.py](code/cifar_selective_generation.py).
- **Loss calibration:** [code/loss_calibration.py](code/loss_calibration.py).

These modules implement diffusion inversion, target attention, latent updates, and perturbation projection.

### 2. Train the Diffusion UNet

Run the training script with a local diffusion model and an image-caption manifest. For example:

```bash
python code/train_diffusion.py \
  --model-dir /path/to/sd2 \
  --manifest data/flickr30k/train.jsonl \
  --output output/diffusion/flickr30k \
  --resolution 512 \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --caption-dropout 0.1 \
  --device cuda
```

The script updates the UNet while freezing the VAE and text encoder.

### 3. Inspect Samples and Check Files

Ten generated unlearnable examples are available in [samples/](samples/). All images use an L-infinity budget of `8/255`, and retain their original image dimensions.
